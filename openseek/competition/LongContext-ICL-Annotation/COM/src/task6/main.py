"""
main.py — Task 6 (mnli_same_genre_classification) 预测入口

依赖：
- ``method.py``                     : 6 轴 ICL 选择器 + 5 套 prompt + 白名单校验 + 平权投票
- ``common.llm_client.annotate_nvidia``: 共享 vLLM/OpenAI 调用封装
- ``common.paths``                  : 统一路径中心（数据/模型/输出）

ICL 数据：``cot_data/openseek-6_mnli_with_cot.json``（含 think + cot 字段，由 ``generate_cot.py`` 离线生成）。

运行示例：
    conda activate flagscale
    bash run.sh                           # 默认：全量 500 样本
    python main.py --max_samples 5        # 快速测试 5 条
    python main.py --concurrency 3        # 样本间并发 3 路（同样本 3 轮串行）

技术方案要点：
- 3 轮正交多视角硬投票（R0 中性锚 / R1 strict_n+全注入 / R2 极端保守）
- 全部 ``api_thinking=True`` + ``temperature=0`` → 确定性可复现
- 投票平票取 N（保守），全 null 兜底 N（MNLI 多数类常识）
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from transformers import AutoTokenizer

# 让 main.py 既能直接 `python main.py` 运行，又能 import 同目录的 method 与
# 上级目录的 common.*。
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task6
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for p in (_CUR_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from method import (  # noqa: E402  (sys.path 设置后才能 import)
    Task6ExampleSelector,
    build_task6_prompt,
    validate_task6_prediction,
    TASK6_LABEL_SET,
    TASK6_GENRE_SET,
    TASK6_N_ROUNDS,
    TASK6_MULTI_VIEW_CONFIGS,
    TASK6_MAX_TOKENS,
    TASK6_TIMEOUT,
    TASK6_TEMPERATURE,
    TASK6_REPETITION_PENALTY,
    TASK6_MAX_INPUT_LENGTH,
    TASK6_MIN_INPUT_LENGTH,
)
from common.llm_client import annotate_nvidia  # noqa: E402
from common.paths import (  # noqa: E402
    FINAL_OUTPUT_DIR,
    MODEL_DIR,
    task_data_file,
    task_cot_dir,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---- 默认路径与配置（统一从 common.paths 派生）----
TASK_ID = 6
DATA_FILE_RAW  = task_data_file(TASK_ID)
COT_DATA_FILE  = os.path.join(task_cot_dir(TASK_ID), 'openseek-6_mnli_with_cot.json')
OUTPUT_PREFIX  = FINAL_OUTPUT_DIR + '/'
TOKENIZER_PATH = MODEL_DIR

MAX_CONTEXT_TOKENS = TASK6_MAX_INPUT_LENGTH    # 31000
MIN_CONTEXT_TOKENS = TASK6_MIN_INPUT_LENGTH    # 30000

DEFAULT_CONCURRENCY = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Task 6 预测（mnli_same_genre_classification + 3 轮正交投票）')
    parser.add_argument('--data_file', type=str, default='',
                        help='显式指定原始数据文件；默认走 common.paths.task_data_file(6)')
    parser.add_argument('--cot_file', type=str, default='',
                        help='显式指定 cot_data 文件；默认走 task_cot_dir(6)/openseek-6_mnli_with_cot.json')
    parser.add_argument('--output_prefix', type=str, default=OUTPUT_PREFIX)
    parser.add_argument('--tokenizer_path', type=str, default=TOKENIZER_PATH)
    parser.add_argument('--max_context_tokens', type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument('--min_context_tokens', type=int, default=MIN_CONTEXT_TOKENS,
                        help='ICL token 数下限（赛题硬约束 ≥ 30K）；低于该值会输出 WARNING')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max test samples (0=all). For quick testing.')
    parser.add_argument('--n_rounds', type=int, default=TASK6_N_ROUNDS,
                        help='多视角投票轮数（默认 3）')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help='样本间并发数（同一样本的 3 轮始终串行）')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'])
    parser.add_argument('--quiet', action='store_true', default=False)
    return parser.parse_args()


def run(args):
    # ---- 加载原始测试数据 ----
    data_file = args.data_file or DATA_FILE_RAW
    if not os.path.isabs(data_file):
        data_file = os.path.abspath(os.path.join(_CUR_DIR, data_file))
    with open(data_file, 'r') as f:
        orig_dict = json.load(f)
    task_name = orig_dict.get('task_name', 'mnli_same_genre_classification')
    task_description = orig_dict['Definition'][0]
    test_samples = orig_dict['test_samples']

    # ---- 加载 ICL 池（cot_data，方案 D 干净 CoT）----
    cot_file = args.cot_file or COT_DATA_FILE
    if not os.path.isabs(cot_file):
        cot_file = os.path.abspath(os.path.join(_CUR_DIR, cot_file))
    if not os.path.exists(cot_file):
        raise FileNotFoundError(f'cot_data 文件不存在: {cot_file}')
    with open(cot_file, 'r') as f:
        cot_dict = json.load(f)
    icl_examples = cot_dict['examples']

    cot_nonempty = sum(1 for ex in icl_examples if (ex.get('cot') or '').strip())
    think_nonempty = sum(1 for ex in icl_examples if (ex.get('think') or '').strip())

    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]
        logger.info(f"[快速测试] 仅评测前 {args.max_samples} 个样本")

    n_rounds = min(args.n_rounds, len(TASK6_MULTI_VIEW_CONFIGS))
    concurrency = max(1, args.concurrency)

    logger.info(f"Task {TASK_ID}: {task_name}")
    logger.info(f"ICL examples: {len(icl_examples)}, Test samples: {len(test_samples)}")
    logger.info(f"示例池: cot={cot_nonempty}/{len(icl_examples)} "
                f"({cot_nonempty/len(icl_examples):.1%}), "
                f"think={think_nonempty}/{len(icl_examples)} "
                f"({think_nonempty/len(icl_examples):.1%})")
    logger.info(f"标签集: {TASK6_LABEL_SET}, Genre 白名单: {sorted(TASK6_GENRE_SET)}")
    logger.info(f"Max context tokens: {args.max_context_tokens}, "
                f"Min context tokens: {args.min_context_tokens}")
    logger.info(f"n_rounds: {n_rounds}, concurrency: {concurrency}")

    # ICL 标签分布
    label_dist = Counter(
        (ex['output'][0] if isinstance(ex['output'], list) else ex['output']).strip().upper()
        for ex in icl_examples
    )
    logger.info(f"ICL 标签分布: {dict(label_dist)}")

    # ---- 初始化 ----
    tokenizer_path = args.tokenizer_path
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = os.path.abspath(os.path.join(_CUR_DIR, tokenizer_path))
    logger.info(f"Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    selector = Task6ExampleSelector(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        min_context_tokens=args.min_context_tokens,
    )
    logger.info(f"推理配置：max_tokens={TASK6_MAX_TOKENS}, "
                f"timeout={TASK6_TIMEOUT}s, "
                f"repetition_penalty={TASK6_REPETITION_PENALTY}, "
                f"temperature={TASK6_TEMPERATURE}")
    for ri, cfg in enumerate(TASK6_MULTI_VIEW_CONFIGS[:n_rounds]):
        logger.info(
            f"  R{ri}: subset={cfg['subset_mode']:<18} balance={cfg['label_balance']:<8} "
            f"order={cfg['order']:<13} style={cfg['prompt_style']:<13} "
            f"cot={int(cfg['cot_inject'])} think={int(cfg['think_inject'])} "
            f"api_thinking={cfg['api_thinking']}"
        )

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

    # ---- 日志文件（保存到 task6/logs/）----
    log_dir = os.path.join(_CUR_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'task_{TASK_ID}_v{version}.log')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log: {os.path.abspath(log_file)}")

    # ---- 全局状态 ----
    null_round_count = 0
    fallback_count = 0
    tie_count = 0
    final_pred_stats: Counter = Counter()
    round_pred_stats = [Counter() for _ in range(n_rounds)]
    results = [None] * len(test_samples)

    write_lock = threading.Lock()
    stat_lock = threading.Lock()
    start_time = time.time()

    def append_record(record: dict):
        with write_lock:
            with open(output_file, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def process_sample(idx: int, test_sample: dict) -> dict:
        """单条样本处理（n_rounds 轮串行调用 + 投票）。"""
        nonlocal null_round_count, fallback_count, tie_count

        sid = test_sample['id']
        text2annotate = test_sample['input']
        stated_genre = Task6ExampleSelector._extract_genre(text2annotate)
        sample_t0 = time.time()
        candidates = []
        sample_null_rounds = 0

        for round_i in range(n_rounds):
            cfg = TASK6_MULTI_VIEW_CONFIGS[round_i]
            subset_mode   = cfg['subset_mode']
            label_balance = cfg['label_balance']
            order         = cfg['order']
            prompt_style  = cfg['prompt_style']
            cot_inject    = cfg['cot_inject']
            think_inject  = cfg['think_inject']
            api_thinking  = cfg['api_thinking']

            t0 = time.time()
            examples_str = selector.select(
                icl_examples, text2annotate,
                round_idx=round_i,
                subset_mode=subset_mode,
                label_balance=label_balance,
                order=order,
                cot_inject=cot_inject,
                think_inject=think_inject,
            )

            input_prompt = build_task6_prompt(
                task_description=task_description,
                text2annotate=text2annotate,
                examples_str=examples_str,
                prompt_style=prompt_style,
            )

            try:
                raw = annotate_nvidia(
                    input_prompt,
                    max_tokens=TASK6_MAX_TOKENS,
                    timeout=TASK6_TIMEOUT,
                    temperature=TASK6_TEMPERATURE,
                    repetition_penalty=TASK6_REPETITION_PENALTY,
                    enable_thinking=api_thinking,
                )
            except Exception as e:
                logger.error(f"[{idx+1}] R{round_i} annotate 异常: {e}")
                raw = None

            pred = validate_task6_prediction(raw, stated_genre=stated_genre) if raw else None
            round_elapsed = time.time() - t0

            with stat_lock:
                if pred:
                    round_pred_stats[round_i][pred] += 1
                else:
                    round_pred_stats[round_i]['NULL'] += 1

            if pred:
                candidates.append(pred)
                logger.info(f"[{idx+1}] R{round_i} style={prompt_style} -> '{pred}' "
                            f"({round_elapsed:.1f}s)")
            else:
                sample_null_rounds += 1
                logger.warning(f"[{idx+1}] R{round_i} style={prompt_style} 无有效预测 "
                               f"({round_elapsed:.1f}s)")

        # ---- 投票 ----
        is_tie = False
        is_fallback = False
        if candidates:
            counter = Counter(candidates)
            most_common = counter.most_common()
            if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
                prediction = 'N'
                is_tie = True
                logger.warning(f"[{idx+1}] 投票平票 {dict(counter)} -> 取 N（保守）")
            else:
                prediction = most_common[0][0]
        else:
            prediction = 'N'
            is_fallback = True
            logger.warning(f"[{idx+1}] {n_rounds} 轮全 null -> 兜底 N（MNLI 多数类）")

        sample_elapsed = time.time() - sample_t0

        with stat_lock:
            null_round_count += sample_null_rounds
            if is_fallback:
                fallback_count += 1
            if is_tie:
                tie_count += 1
            final_pred_stats[prediction] += 1

        votes_str = dict(Counter(candidates)) if candidates else 'empty'
        logger.info(f"[{idx+1}/{len(test_samples)}] sample={sid} "
                    f"input='{text2annotate[:80]}' prediction='{prediction}' "
                    f"votes={votes_str} ({sample_elapsed:.1f}s)")

        record = {'test_sample_id': sid, 'prediction': prediction}
        append_record(record)
        return {'idx': idx, 'record': record}

    # ---- 执行 ----
    if concurrency == 1:
        for i in tqdm(range(len(test_samples)), desc=f'Task {TASK_ID}: {task_name}'):
            r = process_sample(i, test_samples[i])
            results[i] = r['record']
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(process_sample, i, test_samples[i]): i
                       for i in range(len(test_samples))}
            completed = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    r = fut.result()
                    results[i] = r['record']
                except Exception as e:
                    sid = test_samples[i]['id']
                    logger.error(f"[{i+1}] worker 异常: {e}")
                    results[i] = {'test_sample_id': sid, 'prediction': 'N'}
                    with stat_lock:
                        fallback_count += 1
                    append_record(results[i])
                completed += 1
                report_every = max(1, len(test_samples) // 20)
                if completed % report_every == 0 or completed == 1 or completed == len(test_samples):
                    elapsed = time.time() - start_time
                    avg = elapsed / completed
                    remaining = avg * (len(test_samples) - completed)
                    with stat_lock:
                        logger.info(
                            f"[进度] {completed}/{len(test_samples)} | "
                            f"null_rounds={null_round_count} fallback={fallback_count} "
                            f"tie={tie_count} | "
                            f"{elapsed:.0f}s | {avg:.2f}s/样本 | 剩余 ~{remaining:.0f}s"
                        )
        # 并发完成后按顺序重写（保证有序）
        with open(output_file, 'w') as f:
            for r in results:
                if r is None:
                    continue
                f.write(json.dumps(r) + '\n')

    # ---- 统计 ----
    elapsed_total = time.time() - start_time
    n_total = len(test_samples)
    logger.info(f"完成! {elapsed_total:.1f}s, 均速 {elapsed_total/n_total:.1f}s/样本")
    logger.info(f"单轮 null 总次数: {null_round_count}/{n_total * n_rounds} "
                f"({null_round_count/(n_total*n_rounds)*100:.1f}%)")
    logger.info(f"全 null 兜底 N 的样本数: {fallback_count}/{n_total} "
                f"({fallback_count/n_total*100:.1f}%)")
    logger.info(f"投票平票取 N 的样本数: {tie_count}/{n_total} "
                f"({tie_count/n_total*100:.1f}%)")
    logger.info(f"最终预测分布: {dict(final_pred_stats)}")
    for ri, stats in enumerate(round_pred_stats):
        logger.info(f"  R{ri} 预测分布: {dict(stats)}")

    logging.getLogger().removeHandler(fh)
    fh.close()


if __name__ == '__main__':
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    run(args)
