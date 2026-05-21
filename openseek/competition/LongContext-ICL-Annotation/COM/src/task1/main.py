"""
main.py — Task 1 (closest_integers) 预测入口

依赖：
- ``method.py``                     : Task 1 示例选择器 + Prompt
- ``common.llm_client.annotate_nvidia``: 共享的 vLLM/OpenAI 调用封装
- ``common.paths``                  : 统一路径中心（数据/模型/输出/CoT）

路径来源：所有路径均从 :mod:`common.paths` 取，可通过 ``FLAGOS_*``
环境变量覆盖（见 ``env/paths.sh``）。

运行示例：
    conda activate flagscale
    bash run.sh                           # 默认：全量 500 样本
    python main.py --max_samples 20       # 快速测试 20 条
"""

import json
import os
import sys
import argparse
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer

# 让 main.py 既能直接 `python main.py` 运行，又能 `import` 同目录的 method 与
# 上级目录的 common.*。
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task1
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for p in (_CUR_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from method import (  # noqa: E402  (sys.path 设置后才能 import)
    Task1ExampleSelector,
    build_task1_prompt,
)
from common.llm_client import annotate_nvidia  # noqa: E402
from common.paths import (  # noqa: E402
    FINAL_OUTPUT_DIR,
    MODEL_DIR,
    task_cot_dir,
    task_data_file,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---- 默认路径与配置（统一从 common.paths 派生）----
TASK_ID = 1
DATA_FILE_WITH_COT = task_cot_dir(TASK_ID) + '/openseek-1_closest_integers_with_cot.json'
DATA_FILE_RAW      = task_data_file(TASK_ID)
OUTPUT_PREFIX      = FINAL_OUTPUT_DIR + '/'
TOKENIZER_PATH     = MODEL_DIR

MAX_CONTEXT_TOKENS = 31000   # 单条 ICL prompt 的 token 上限（赛题硬约束 ≥ 30K，留 1K 余量）
MIN_CONTEXT_TOKENS = 30000   # 单条 ICL prompt 的 token 下限（赛题硬约束 ≥ 30K，必须达标）
MAX_TOKENS = 2048            # thinking 模式需要更多生成空间
TIMEOUT = 600                # 单次请求超时
TEMPERATURE = 0.0            # 确定性输出
REPETITION_PENALTY = 1.1     # 抑制死循环复读
ENABLE_THINKING = True       # 配合 think 示例使用 Qwen3 thinking 模式
DEFAULT_CONCURRENCY = 1      # 默认串行（vLLM 显存吃紧时建议保持 1）


def parser_args():
    parser = argparse.ArgumentParser(description='Task 1 预测（列表长度匹配 + CoT + think）')
    parser.add_argument('--data_file', type=str, default='',
                        help='显式指定数据文件；默认优先用 with_cot，不存在时回落到原始数据')
    parser.add_argument('--output_prefix', type=str, default=OUTPUT_PREFIX)
    parser.add_argument('--tokenizer_path', type=str, default=TOKENIZER_PATH)
    parser.add_argument('--max_context_tokens', type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument('--min_context_tokens', type=int, default=MIN_CONTEXT_TOKENS,
                        help='ICL token 数下限（赛题硬约束 ≥ 30K）；低于该值会输出 WARNING')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max test samples (0=all). For quick testing.')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'])
    parser.add_argument('--disable_thinking', action='store_true', default=False,
                        help='强制关闭 thinking 模式')
    parser.add_argument('--no_think_examples', action='store_true', default=False,
                        help='示例拼接不使用 think 字段（仅用 cot）')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'并发请求数（默认 {DEFAULT_CONCURRENCY}，设为 1 即串行）')
    parser.add_argument('--verbose_prompt', action='store_true', default=True,
                        help='开启详细 prompt / 输出日志（默认开启）')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='关闭详细 prompt 日志（与 --verbose_prompt 对立）')
    return parser.parse_args()


def run(args):
    # ---- 加载数据 ----
    data_file = args.data_file
    if not data_file:
        # 优先 with_cot，不存在则回落原始数据（这里都是绝对路径）
        with_cot_path = DATA_FILE_WITH_COT
        raw_path = DATA_FILE_RAW
        if os.path.exists(with_cot_path):
            data_file = with_cot_path
            logger.info(f"使用带 cot+think 的数据: {data_file}")
        else:
            data_file = raw_path
            logger.warning(f"with_cot 不存在，回落到原始数据: {data_file}")
    elif not os.path.isabs(data_file):
        data_file = os.path.abspath(os.path.join(_CUR_DIR, data_file))
    with open(data_file, 'r') as f:
        task_dict = json.load(f)

    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples']
    test_samples = task_dict['test_samples']

    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]
        logger.info(f"[快速测试] 仅评测前 {args.max_samples} 个样本")

    logger.info(f"ICL examples: {len(icl_examples)}, Test samples: {len(test_samples)}")
    logger.info(f"Max context tokens: {args.max_context_tokens}")
    logger.info(f"Min context tokens: {args.min_context_tokens}")

    # ---- 初始化 ----
    tokenizer_path = args.tokenizer_path
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = os.path.abspath(os.path.join(_CUR_DIR, tokenizer_path))
    logger.info(f"Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    selector = Task1ExampleSelector(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        min_context_tokens=args.min_context_tokens,
        use_think=not args.no_think_examples,
    )
    n_has_think = sum(1 for ex in icl_examples if ex.get('think'))
    n_has_cot = sum(1 for ex in icl_examples if ex.get('cot'))
    logger.info(f"示例池统计：think={n_has_think}/{len(icl_examples)}, "
                f"cot={n_has_cot}/{len(icl_examples)}, "
                f"use_think={not args.no_think_examples}")
    enable_thinking = ENABLE_THINKING and not args.disable_thinking
    logger.info(f"推理配置：enable_thinking={enable_thinking}, "
                f"max_tokens={MAX_TOKENS}, timeout={TIMEOUT}s")

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

    # ---- 日志文件（保存到 task1/logs/）----
    log_dir = os.path.join(_CUR_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'task_{TASK_ID}_v{version}.log')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log: {os.path.abspath(log_file)}")

    # ---- 主循环（并发）----
    null_count = 0
    start_time = time.time()
    concurrency = max(1, args.concurrency)
    verbose = args.verbose_prompt and not args.quiet
    logger.info(f"并发配置：concurrency={concurrency}, verbose_prompt={verbose}")

    # 结果按原顺序存储
    results = [None] * len(test_samples)
    write_lock = threading.Lock()
    stat_lock = threading.Lock()

    # 增量写入 jsonl（避免中断丢失）
    def append_result(r: dict):
        with write_lock:
            with open(output_file, 'a') as f:
                f.write(json.dumps({'test_sample_id': r['test_sample_id'],
                                    'prediction': r['prediction']}) + '\n')

    def process_sample(idx: int, test_sample: dict) -> dict:
        """单条样本处理（线程安全）"""
        nonlocal null_count
        text2annotate = test_sample['input']
        tag = f"[{idx+1}/{len(test_samples)}]"
        sample_t0 = time.time()

        # 示例选择
        examples_str = selector.select(icl_examples, text2annotate)
        n_examples = examples_str.count('\n# [')  # 估算示例数（每条以 # [ 开头）
        if n_examples == 0:
            n_examples = sum(1 for ln in examples_str.splitlines()
                             if ln.startswith('# ['))

        # Prompt 构建
        input_prompt = build_task1_prompt(task_description, text2annotate, examples_str)
        prompt_chars = len(input_prompt)
        try:
            prompt_tokens = len(tokenizer.encode(input_prompt, add_special_tokens=False))
        except Exception:
            prompt_tokens = -1

        # 详细日志：样本开始
        logger.info("=" * 80)
        logger.info(f"{tag} START id={test_sample['id']}")
        logger.info(f"{tag} INPUT: {text2annotate}")
        logger.info(f"{tag} examples={n_examples}, prompt_chars={prompt_chars}, "
                    f"prompt_tokens={prompt_tokens}, budget={args.max_context_tokens}")
        if verbose:
            head = input_prompt[:800]
            tail = input_prompt[-1200:]
            logger.info(f"{tag} PROMPT_HEAD (first 800 chars):\n{head}")
            logger.info(f"{tag} PROMPT_TAIL (last 1200 chars):\n{tail}")

        # API 调用（return_raw=True 拿回原始输出，便于 thinking 模式下兜底提取 <label>）
        try:
            prediction, raw = annotate_nvidia(
                input_prompt, max_tokens=MAX_TOKENS,
                timeout=TIMEOUT, temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                enable_thinking=enable_thinking,
                return_raw=True,
            )
        except Exception as e:
            logger.error(f"{tag} annotate_nvidia 异常: {e}")
            prediction, raw = None, ""

        # thinking 模式下 content 为空——从原始输出正则提取 <label>
        if prediction is None and raw:
            import re as _re
            m = _re.search(r'<label>\s*(.+?)\s*</label>', raw, _re.DOTALL)
            if m:
                prediction = m.group(1).strip()
                logger.info(f"{tag} Fallback 从原始输出提取 <label>: '{prediction}'")

        if prediction is None:
            prediction = ""

        sample_elapsed = time.time() - sample_t0

        # 详细日志：样本结果
        logger.info(f"{tag} RAW_LEN={len(raw)} RAW_PREVIEW: {raw[:300]!r}")
        if prediction == "":
            logger.warning(f"{tag} NULL 预测 ({sample_elapsed:.1f}s)")
        else:
            logger.info(f"{tag} prediction='{prediction}' ({sample_elapsed:.1f}s)")
        logger.info("=" * 80)

        # 线程安全地更新统计
        with stat_lock:
            if prediction == "":
                null_count += 1

        result = {
            'idx': idx,
            'test_sample_id': test_sample['id'],
            'prediction': prediction,
            'elapsed': sample_elapsed,
        }
        # 增量写入（中断不丢）
        append_result(result)
        return result

    # 并发执行
    if concurrency == 1:
        iterator = range(len(test_samples))
        for i in tqdm(iterator, desc=f'Task {TASK_ID}: closest_integers'):
            r = process_sample(i, test_samples[i])
            results[i] = r
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_sample, i, test_samples[i]): i
                for i in range(len(test_samples))
            }
            completed = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    logger.error(f"[{i+1}] worker 异常: {e}")
                    r = {'idx': i, 'test_sample_id': test_samples[i]['id'],
                         'prediction': '', 'elapsed': 0.0}
                    with stat_lock:
                        null_count += 1
                results[i] = r
                completed += 1
                if completed % max(1, len(test_samples) // 10) == 0 or completed == 1:
                    elapsed = time.time() - start_time
                    avg = elapsed / completed
                    remaining = avg * (len(test_samples) - completed)
                    logger.info(
                        f"[进度] {completed}/{len(test_samples)} | "
                        f"null={null_count} | "
                        f"{elapsed:.0f}s | {avg:.2f}s/样本均值 | 剩余 ~{remaining:.0f}s"
                    )

    # 最终确认：按原顺序重写 jsonl（并发时顺序不定）
    if concurrency > 1:
        with open(output_file, 'w') as f:
            for r in results:
                if r is None:
                    continue
                f.write(json.dumps({'test_sample_id': r['test_sample_id'],
                                    'prediction': r['prediction']}) + '\n')
        logger.info(f"已按顺序重写 {sum(1 for r in results if r)} 条到 {output_file}")
    else:
        logger.info(f"已增量写入 {sum(1 for r in results if r)} 条到 {output_file}")

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
