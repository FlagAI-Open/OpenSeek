"""
main.py — Task 5 (tweet_sadness_detection) 预测入口

依赖：
- ``method.py``                     : Task 5 示例选择器 + 5 套 Prompt + 标签校验 + 多数票投票
- ``common.llm_client.annotate_nvidia``: 共享的 vLLM/OpenAI 调用封装
- ``common.paths``                  : 统一路径中心（数据/模型/输出）

运行示例：
    conda activate flagscale
    bash run.sh                           # 默认：全量 500 样本
    python main.py --max_samples 20       # 快速测试 20 条

技术方案要点：
- 5 轮正交多视角投票（A/B/C/D 四轴），每轮 ICL 上下文稳定 ≥ 30K tokens（赛题硬约束）
- ``temperature=0`` + ``enable_thinking=False`` → 完全确定性可复现
- 多样性来自 prompt / 示例侧的正交设计，不引入采样扰动
"""

import json
import os
import sys
import argparse
import time
import logging
from tqdm import tqdm
from transformers import AutoTokenizer

# 让 main.py 既能直接 `python main.py` 运行，又能 `import` 同目录的 method 与
# 上级目录的 common.*。
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task5
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for p in (_CUR_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from method import (  # noqa: E402  (sys.path 设置后才能 import)
    Task5ExampleSelector,
    build_task5_prompt,
    extract_label_set,
    validate_prediction,
    majority_vote,
    TASK5_MAX_TOKENS,
    TASK5_TIMEOUT,
    TASK5_TEMPERATURE,
    TASK5_REPETITION_PENALTY,
    TASK5_ENABLE_THINKING,
    TASK5_N_ROUNDS,
    TASK5_MULTI_VIEW_CONFIGS,
    TASK5_MAX_INPUT_LENGTH,
    TASK5_MIN_INPUT_LENGTH,
)
from common.llm_client import annotate_nvidia  # noqa: E402
from common.paths import (  # noqa: E402
    FINAL_OUTPUT_DIR,
    MODEL_DIR,
    task_data_file,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---- 默认路径与配置（统一从 common.paths 派生）----
TASK_ID = 5
DATA_FILE_RAW  = task_data_file(TASK_ID)
OUTPUT_PREFIX  = FINAL_OUTPUT_DIR + '/'
TOKENIZER_PATH = MODEL_DIR

MAX_CONTEXT_TOKENS = TASK5_MAX_INPUT_LENGTH   # 31000
MIN_CONTEXT_TOKENS = TASK5_MIN_INPUT_LENGTH   # 30000


def parser_args():
    parser = argparse.ArgumentParser(
        description='Task 5 预测（tweet_sadness_detection + 5 轮正交投票）')
    parser.add_argument('--data_file', type=str, default='',
                        help='显式指定数据文件；默认走 common.paths.task_data_file(5)')
    parser.add_argument('--output_prefix', type=str, default=OUTPUT_PREFIX)
    parser.add_argument('--tokenizer_path', type=str, default=TOKENIZER_PATH)
    parser.add_argument('--max_context_tokens', type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument('--min_context_tokens', type=int, default=MIN_CONTEXT_TOKENS,
                        help='ICL token 数下限（赛题硬约束 ≥ 30K）；低于该值会输出 WARNING')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max test samples (0=all). For quick testing.')
    parser.add_argument('--n_rounds', type=int, default=TASK5_N_ROUNDS,
                        help='多视角投票轮数（默认 5）')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'])
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='静音模式：仅打印汇总级日志')
    return parser.parse_args()


def run(args):
    # ---- 加载数据 ----
    data_file = args.data_file or DATA_FILE_RAW
    if not os.path.isabs(data_file):
        data_file = os.path.abspath(os.path.join(_CUR_DIR, data_file))
    with open(data_file, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict.get('task_name', 'tweet_sadness_detection')
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples']
    test_samples = task_dict['test_samples']

    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]
        logger.info(f"[快速测试] 仅评测前 {args.max_samples} 个样本")

    logger.info(f"Task {TASK_ID}: {task_name}")
    logger.info(f"ICL examples: {len(icl_examples)}, Test samples: {len(test_samples)}")
    logger.info(f"Max context tokens: {args.max_context_tokens}")
    logger.info(f"Min context tokens: {args.min_context_tokens}")
    logger.info(f"n_rounds: {args.n_rounds}")

    # 提取标签集合
    label_set = extract_label_set(icl_examples)
    logger.info(f"标签集: {label_set}")

    # ---- 初始化 ----
    tokenizer_path = args.tokenizer_path
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = os.path.abspath(os.path.join(_CUR_DIR, tokenizer_path))
    logger.info(f"Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    selector = Task5ExampleSelector(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        min_context_tokens=args.min_context_tokens,
    )
    logger.info(f"推理配置：enable_thinking={TASK5_ENABLE_THINKING}, "
                f"max_tokens={TASK5_MAX_TOKENS}/1024 (R0-R2/R3-R4), "
                f"timeout={TASK5_TIMEOUT}s, "
                f"repetition_penalty={TASK5_REPETITION_PENALTY}, "
                f"temperature={TASK5_TEMPERATURE}")

    # ---- 输出文件（自动版本号）----
    output_prefix = args.output_prefix
    if not os.path.isabs(output_prefix):
        output_prefix = os.path.abspath(os.path.join(_CUR_DIR, output_prefix)) + os.sep
    os.makedirs(output_prefix, exist_ok=True)
    version = 1
    output_file = f'{output_prefix}openseek-{TASK_ID}-v{version}.jsonl'
    while os.path.exists(output_file):
        version += 1
        output_file = f'{output_prefix}openseek-{TASK_ID}-v{version}.jsonl'
    with open(output_file, 'w'):
        pass
    logger.info(f"Output: {output_file} (v{version})")

    # ---- 日志文件（保存到 task5/logs/）----
    log_dir = os.path.join(_CUR_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'task_{TASK_ID}_v{version}.log')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log: {os.path.abspath(log_file)}")

    # ---- 主循环（5 轮正交多视角投票）----
    null_count = 0
    start_time = time.time()
    report_interval = max(1, len(test_samples) // 10)

    for i, test_sample in enumerate(tqdm(test_samples,
                                         desc=f'Task {TASK_ID}: {task_name}')):
        text2annotate = test_sample['input']
        sample_t0 = time.time()
        candidates = []

        for round_i in range(args.n_rounds):
            cfg = TASK5_MULTI_VIEW_CONFIGS[round_i % len(TASK5_MULTI_VIEW_CONFIGS)]
            pool_composition = cfg.get('pool_composition', 'full_top')
            label_balance    = cfg.get('label_balance', 'natural')
            order            = cfg.get('order', 'recency_up')
            prompt_style     = cfg.get('prompt_style', 'neutral')
            round_max_tokens = cfg.get('max_tokens', TASK5_MAX_TOKENS)

            # 示例选择（A/B/C 三轴正交配置）
            examples_str = selector.select(
                icl_examples, text2annotate,
                round_idx=round_i,
                pool_composition=pool_composition,
                label_balance=label_balance,
                order=order,
            )

            # 构建 Prompt（D 轴：prompt_style）
            input_prompt = build_task5_prompt(
                task_description=task_description,
                text2annotate=text2annotate,
                examples_str=examples_str,
                label_set=label_set,
                prompt_style=prompt_style,
            )

            # API 调用（temperature=0, thinking=False, 完全确定性）
            raw_pred = annotate_nvidia(
                input_prompt,
                max_tokens=round_max_tokens,
                timeout=TASK5_TIMEOUT,
                temperature=TASK5_TEMPERATURE,
                repetition_penalty=TASK5_REPETITION_PENALTY,
                enable_thinking=TASK5_ENABLE_THINKING,
            )
            pred = validate_prediction(raw_pred, label_set)

            if pred:
                candidates.append(pred)
                logger.debug(f"[{i+1}] Round {round_i}: style={prompt_style} pred='{pred}'")
            else:
                logger.debug(f"[{i+1}] Round {round_i}: style={prompt_style} 无答案")

        # 多数票决定最终预测（5 轮平权硬投票）
        prediction = majority_vote(candidates) if candidates else None
        sample_elapsed = time.time() - sample_t0

        # 结果不能为 null
        if prediction is None:
            prediction = ""
            null_count += 1
            logger.warning(f"[{i+1}/{len(test_samples)}] sample={test_sample['id']} "
                           f"input='{text2annotate[:80]}' "
                           f"prediction='' ({sample_elapsed:.1f}s)")
        else:
            logger.info(f"[{i+1}/{len(test_samples)}] sample={test_sample['id']} "
                        f"input='{text2annotate[:80]}' "
                        f"prediction='{prediction}' ({sample_elapsed:.1f}s)")

        # 增量写入（中断不丢）
        with open(output_file, 'a') as f:
            f.write(json.dumps({'test_sample_id': test_sample['id'],
                                'prediction': prediction}) + '\n')

        # 进度报告
        if (i + 1) % report_interval == 0 or i == 0:
            elapsed_so_far = time.time() - start_time
            avg = elapsed_so_far / (i + 1)
            remaining = avg * (len(test_samples) - i - 1)
            logger.info(
                f"[{i+1}/{len(test_samples)}] "
                f"null={null_count}/{i+1} ({null_count/(i+1)*100:.1f}%) | "
                f"{elapsed_so_far:.0f}s | {avg:.1f}s/样本 | 剩余 ~{remaining:.0f}s"
            )

    # ---- 统计 ----
    elapsed = time.time() - start_time
    total = len(test_samples)
    logger.info(f"完成! {elapsed:.1f}s, 均速 {elapsed/total:.1f}s/样本")
    logger.info(f"Null: {null_count}/{total} ({null_count/total*100:.1f}%)")

    logging.getLogger().removeHandler(fh)
    fh.close()


if __name__ == '__main__':
    args = parser_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    run(args)
