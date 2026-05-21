"""
main.py — Task 7 (jeopardy_answer_generation) 预测入口

依赖：
- ``method.py``                     : 三策略 ICL + 三 Prompt + ClueTrap + Category + retry + 验证器
- ``common.llm_client.annotate_nvidia``: 共享 vLLM/OpenAI 调用封装
- ``common.paths``                  : 统一路径中心（数据/模型/输出）

ICL 数据：``data/openseek-7_jeopardy_answer_generation_all.json``（5499 条原始 examples）。

运行示例：
    conda activate flagscale
    bash run.sh                                    # 默认：全量 500 样本（ensemble）
    python main.py --max_samples 5                 # 快速测试 5 条
    python main.py --mode single --prompt_variant A   # 单 Prompt 模式（不调验证器）

技术方案要点：
- 三策略检索（A: Category+Clue BM25 / B: 纯 Clue BM25 / C: Category+随机打散）
- 三路答案 → ClueTrap 整词过滤 + Category 引号字母硬约束过滤
- 全剔时触发 retry（带 30K ICL + 排除原答案）
- 词边界子串去重 + 验证器裁决（多候选时调用）
- 思考模式 fallback：rescue_from_thinking → /no_think 降级重试
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer

# 让 main.py 既能直接 ``python main.py`` 运行，又能 import 同目录的 method 与
# 上级目录的 common.*。
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task7
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for p in (_CUR_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from method import (  # noqa: E402  (sys.path 设置后才能 import)
    Task7ExampleSelector,
    build_task7_prompt,
    build_verifier_prompt,
    build_task7_retry_prompt,
    candidate_in_clue_trap,
    parse_category_letter_constraint,
    candidate_satisfies_letter,
    postprocess_task7,
    validate_prediction,
    rescue_from_thinking,
    dedupe_candidates,
    TASK7_MAX_TOKENS,
    TASK7_TIMEOUT,
    TASK7_ENABLE_THINKING,
    TASK7_REP_PENALTY,
    TASK7_TEMPERATURE,
    TASK7_MAX_INPUT_LENGTH,
    TASK7_MIN_INPUT_LENGTH,
    TASK7_VERIFIER_MAX_TOKENS,
    TASK7_NOTHINK_MAX_TOKENS,
    TASK7_NOTHINK_TIMEOUT,
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
TASK_ID = 7
DATA_FILE_RAW  = task_data_file(TASK_ID)
OUTPUT_PREFIX  = FINAL_OUTPUT_DIR + '/'
TOKENIZER_PATH = MODEL_DIR

MAX_CONTEXT_TOKENS = TASK7_MAX_INPUT_LENGTH    # 31000
MIN_CONTEXT_TOKENS = TASK7_MIN_INPUT_LENGTH    # 30000


def parse_args():
    parser = argparse.ArgumentParser(
        description='Task 7 预测（jeopardy_answer_generation + 三策略 ensemble）')
    parser.add_argument('--data_file', type=str, default='',
                        help='显式指定原始数据文件；默认走 common.paths.task_data_file(7)')
    parser.add_argument('--output_prefix', type=str, default=OUTPUT_PREFIX)
    parser.add_argument('--tokenizer_path', type=str, default=TOKENIZER_PATH)
    parser.add_argument('--max_context_tokens', type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument('--min_context_tokens', type=int, default=MIN_CONTEXT_TOKENS,
                        help='ICL token 数下限（赛题硬约束 ≥ 30K）；低于该值会输出 WARNING')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max test samples (0=all). For quick testing.')
    parser.add_argument('--mode', type=str, default='ensemble',
                        choices=['ensemble', 'single'],
                        help='ensemble: 每条样本跑 A+B+C + 硬过滤 + 验证器; '
                             'single: 只跑 --prompt_variant 指定的单个 Prompt')
    parser.add_argument('--prompt_variant', type=str, default='A',
                        choices=['A', 'B', 'C'],
                        help='单 Prompt 模式下使用的变体（仅 --mode single 时生效）')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'])
    return parser.parse_args()


def _call_variant(prompt: str, variant_name: str) -> str:
    """调用单个 Prompt 变体（含 fallback），返回后处理后的答案（空串表示失败）。"""
    raw = annotate_nvidia(
        prompt,
        max_tokens=TASK7_MAX_TOKENS,
        timeout=TASK7_TIMEOUT,
        temperature=TASK7_TEMPERATURE,
        repetition_penalty=TASK7_REP_PENALTY,
        enable_thinking=TASK7_ENABLE_THINKING,
    )
    pred = validate_prediction(raw)
    if pred is None and raw:
        rescued = rescue_from_thinking(raw)
        if rescued:
            pred = rescued
            logger.info(f"    [{variant_name}] 思考链抢救 → '{pred}'")
    if pred is None:
        logger.warning(f"    [{variant_name}] 无输出，降级 /no_think 重试")
        raw_retry = annotate_nvidia(
            prompt,
            max_tokens=TASK7_NOTHINK_MAX_TOKENS,
            timeout=TASK7_NOTHINK_TIMEOUT,
            temperature=TASK7_TEMPERATURE,
            enable_thinking=False,
        )
        pred = validate_prediction(raw_retry)
    if pred:
        before = pred
        pred = postprocess_task7(pred)
        if before != pred:
            logger.debug(f"    [{variant_name}] 后处理: '{before}' → '{pred}'")
    return pred or ""


def _ensemble_predict(selector: Task7ExampleSelector,
                      icl_examples: list,
                      task_description: str,
                      text2annotate: str,
                      idx: int) -> dict:
    """三策略 ensemble + 两道硬过滤 + retry + 验证器。

    Returns:
        dict: prediction / ans_a / ans_b / ans_c / cat_letter / clue_trapped /
              cat_violated / retry_fired
    """
    import re as _re

    cat_m = _re.search(r'Category:\s*(.+?)(?:\n|$)', text2annotate)
    clue_m = _re.search(r'Clue:\s*(.+?)(?:\n|$)', text2annotate, _re.DOTALL)
    category_str = cat_m.group(1).strip() if cat_m else "Unknown"
    clue_str = clue_m.group(1).strip() if clue_m else text2annotate.strip()

    # ---- 三策略检索 ----
    examples_a = selector.select_by_strategy('A', icl_examples, text2annotate)
    examples_b = selector.select_by_strategy('B', icl_examples, text2annotate)
    examples_c = selector.select_by_strategy('C', icl_examples, text2annotate)

    # ---- 三路调用（统一使用 variant=A 的 prompt 模板，仅 examples 不同） ----
    t_a = time.time()
    ans_a = _call_variant(
        build_task7_prompt(task_description, text2annotate, examples_a, 'A'), 'A')
    logger.info(f"  Prompt A (Cat+Clue BM25) → '{ans_a}' ({time.time()-t_a:.1f}s)")

    t_b = time.time()
    ans_b = _call_variant(
        build_task7_prompt(task_description, text2annotate, examples_b, 'A'), 'B')
    logger.info(f"  Prompt B (Clue-only BM25) → '{ans_b}' ({time.time()-t_b:.1f}s)")

    t_c = time.time()
    ans_c = _call_variant(
        build_task7_prompt(task_description, text2annotate, examples_c, 'A'), 'C')
    logger.info(f"  Prompt C (Cat+shuffled) → '{ans_c}' ({time.time()-t_c:.1f}s)")

    raw_pool = [ans_a, ans_b, ans_c]
    non_empty = [a for a in raw_pool if a]

    # ---- 过滤 1：ClueTrap（候选作整词出现在 Clue 中即剔除） ----
    clue_trapped = [a for a in non_empty
                    if candidate_in_clue_trap(a, text2annotate)]
    pool_after_trap = [a for a in non_empty if a not in clue_trapped]
    if clue_trapped:
        logger.info(f"  [ClueTrap] 剔除 {len(clue_trapped)} 个: {clue_trapped}")

    # ---- 过滤 2：Category 引号字母硬约束 ----
    cat_letter = parse_category_letter_constraint(category_str)
    cat_violated: list = []
    pool_before_cat = pool_after_trap[:]
    pool = pool_after_trap[:]
    if cat_letter and pool_after_trap:
        cat_violated = [a for a in pool_after_trap
                        if not candidate_satisfies_letter(a, cat_letter)]
        pool = [a for a in pool_after_trap if a not in cat_violated]
        if cat_violated:
            logger.info(f"  [Category] 约束='{cat_letter}' "
                        f"剔除 {len(cat_violated)} 个: {cat_violated}")

    # ---- 全剔时触发重调用（一次性决定最终答案） ----
    retry_fired: str = ''
    prediction: str = ''

    if non_empty and not pool_after_trap:
        retry_fired = 'clue_trap'
        t_r = time.time()
        retry_prompt = build_task7_retry_prompt(
            task_description, text2annotate, examples_a,
            excluded=non_empty, exclude_reason='clue_trap')
        prediction = _call_variant(retry_prompt, 'RETRY-CT')
        logger.info(f"  [RETRY/CLUE_TRAP] 三路全抄 Clue → 重调用 → '{prediction}' "
                    f"({time.time()-t_r:.1f}s)")
    elif pool_before_cat and not pool:
        retry_fired = 'category_violation'
        t_r = time.time()
        retry_prompt = build_task7_retry_prompt(
            task_description, text2annotate, examples_a,
            excluded=pool_before_cat, exclude_reason='category_violation')
        prediction = _call_variant(retry_prompt, 'RETRY-CAT')
        logger.info(f"  [RETRY/CAT] 三路全违反 '{cat_letter}' → 重调用 → '{prediction}' "
                    f"({time.time()-t_r:.1f}s)")
    else:
        # ---- 决策：一致跳过验证器；分歧用词边界去重 + 验证器 ----
        if pool and all(a == pool[0] for a in pool):
            prediction = pool[0]
            logger.info(f"  [一致] pool={pool} 全相同，跳过验证器 → '{prediction}'")
        elif not pool:
            prediction = ''
            logger.warning(f"  [EMPTY] 三路全 empty，pool 为空")
        else:
            unique_candidates = dedupe_candidates(pool) or ['(no answer)']
            if len(unique_candidates) == 1:
                prediction = unique_candidates[0]
                logger.info(f"  [分歧→一致] pool={pool}，"
                            f"词边界去重后一致 → '{prediction}'，跳过验证器")
            else:
                logger.info(f"  [{len(unique_candidates)} 候选] pool={pool}，调验证器")
                t_v = time.time()
                verify_prompt = build_verifier_prompt(
                    category_str, clue_str, unique_candidates,
                    examples_str=examples_a)
                raw_v = annotate_nvidia(
                    verify_prompt,
                    max_tokens=TASK7_VERIFIER_MAX_TOKENS,
                    timeout=TASK7_TIMEOUT,
                    temperature=TASK7_TEMPERATURE,
                    repetition_penalty=TASK7_REP_PENALTY,
                    enable_thinking=TASK7_ENABLE_THINKING,
                )
                prediction = validate_prediction(raw_v)
                if prediction is None and raw_v:
                    prediction = rescue_from_thinking(raw_v)
                if prediction:
                    # 验证器输出单字母时映射到对应候选
                    labels = ['a', 'b', 'c']
                    letter_map = {labels[i]: unique_candidates[i]
                                  for i in range(len(unique_candidates))}
                    pred_lower = prediction.strip().lower()
                    if pred_lower in letter_map:
                        mapped = letter_map[pred_lower]
                        logger.info(f"  [验证器] 输出字母 '{pred_lower.upper()}' → 展开为 '{mapped}'")
                        prediction = mapped
                    else:
                        before = prediction
                        prediction = postprocess_task7(prediction)
                        if before != prediction:
                            logger.debug(f"  [验证器] 后处理: '{before}' → '{prediction}'")
                    logger.info(f"  [验证器] → '{prediction}' ({time.time()-t_v:.1f}s)")
                else:
                    # 多数投票降级
                    votes = pool[:]
                    prediction = Counter(votes).most_common(1)[0][0] if votes else ''
                    logger.warning(f"  [验证器] 无输出，多数投票降级 → '{prediction}'")

    prediction = prediction or ''
    return {
        'prediction': prediction,
        'ans_a': ans_a or '',
        'ans_b': ans_b or '',
        'ans_c': ans_c or '',
        'cat_letter': cat_letter or '',
        'clue_trapped': clue_trapped,
        'cat_violated': cat_violated,
        'retry_fired': retry_fired,
    }


def _single_predict(selector: Task7ExampleSelector,
                    icl_examples: list,
                    task_description: str,
                    text2annotate: str,
                    prompt_variant: str) -> str:
    """单 Prompt 模式（向后兼容）。"""
    examples_str = selector.select_by_strategy('A', icl_examples, text2annotate)
    input_prompt = build_task7_prompt(
        task_description, text2annotate, examples_str, prompt_variant)
    return _call_variant(input_prompt, prompt_variant)


def run(args):
    # ---- 加载原始测试数据 ----
    data_file = args.data_file or DATA_FILE_RAW
    if not os.path.isabs(data_file):
        data_file = os.path.abspath(os.path.join(_CUR_DIR, data_file))
    with open(data_file, 'r') as f:
        orig_dict = json.load(f)
    task_name = orig_dict.get('task_name', 'jeopardy_answer_generation')
    task_description = orig_dict['Definition'][0]
    test_samples = orig_dict['test_samples']
    icl_examples = orig_dict['examples']

    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]
        logger.info(f"[快速测试] 仅评测前 {args.max_samples} 个样本")

    logger.info(f"Task {TASK_ID}: {task_name}")
    logger.info(f"ICL examples: {len(icl_examples)}, Test samples: {len(test_samples)}")
    logger.info(f"Mode: {args.mode}, "
                f"Prompt variant: {args.prompt_variant if args.mode=='single' else 'A+B+C+Verify'}")
    logger.info(f"Max context tokens: {args.max_context_tokens}, "
                f"Min context tokens: {args.min_context_tokens}")

    # ---- 初始化 ----
    tokenizer_path = args.tokenizer_path
    if not os.path.isabs(tokenizer_path):
        tokenizer_path = os.path.abspath(os.path.join(_CUR_DIR, tokenizer_path))
    logger.info(f"Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    selector = Task7ExampleSelector(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        min_context_tokens=args.min_context_tokens,
    )
    logger.info(f"推理配置：max_tokens={TASK7_MAX_TOKENS}, "
                f"timeout={TASK7_TIMEOUT}s, "
                f"enable_thinking={TASK7_ENABLE_THINKING}, "
                f"repetition_penalty={TASK7_REP_PENALTY}, "
                f"temperature={TASK7_TEMPERATURE}")

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

    # ---- 日志文件（保存到 task7/logs/）----
    log_dir = os.path.join(_CUR_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'task_{TASK_ID}_v{version}.log')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log: {os.path.abspath(log_file)}")

    # ---- 全局状态 ----
    null_count = 0
    retry_count: Counter = Counter()
    final_pred_stats: Counter = Counter()
    start_time = time.time()
    report_interval = max(1, len(test_samples) // 10)
    n_total = len(test_samples)

    for i, test_sample in enumerate(tqdm(test_samples,
                                          desc=f'Task {TASK_ID}: {task_name}')):
        sid = test_sample['id']
        text2annotate = test_sample['input']
        sample_t0 = time.time()
        record: dict = {'test_sample_id': sid}

        if args.mode == 'ensemble':
            logger.info(f"[{i+1}/{n_total}] sample={sid}")
            logger.info(f"  input='{text2annotate[:100]}'")
            res = _ensemble_predict(selector, icl_examples, task_description,
                                    text2annotate, i)
            prediction = res['prediction']
            record['prediction'] = prediction
            record['ans_a'] = res['ans_a']
            record['ans_b'] = res['ans_b']
            record['ans_c'] = res['ans_c']
            record['cat_letter'] = res['cat_letter']
            record['clue_trapped'] = res['clue_trapped']
            record['cat_violated'] = res['cat_violated']
            record['retry_fired'] = res['retry_fired']
            if res['retry_fired']:
                retry_count[res['retry_fired']] += 1
            logger.info(f"  >>> FINAL: '{prediction}' | retry={res['retry_fired'] or 'none'} "
                        f"| 总耗时: {time.time()-sample_t0:.1f}s")
        else:
            prediction = _single_predict(
                selector, icl_examples, task_description,
                text2annotate, args.prompt_variant)
            record['prediction'] = prediction
            logger.info(f"[{i+1}/{n_total}] sample={sid} "
                        f"prompt={args.prompt_variant} prediction='{prediction}' "
                        f"({time.time()-sample_t0:.1f}s)")

        if not prediction:
            null_count += 1
            logger.warning(f"[{i+1}/{n_total}] sample={sid} → EMPTY")

        final_pred_stats[prediction or '<EMPTY>'] += 1

        with open(output_file, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # ---- 进度报告 ----
        if (i + 1) % report_interval == 0 or i == 0 or (i + 1) == n_total:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = avg * (n_total - i - 1)
            logger.info(
                f"[进度] {i+1}/{n_total} | null={null_count}/{i+1} "
                f"({null_count/(i+1)*100:.1f}%) | "
                f"retry={dict(retry_count)} | "
                f"{elapsed:.0f}s | {avg:.1f}s/样本 | 剩余 ~{remaining:.0f}s"
            )

    # ---- 统计 ----
    elapsed_total = time.time() - start_time
    logger.info(f"完成! {elapsed_total:.1f}s, 均速 {elapsed_total/n_total:.1f}s/样本")
    logger.info(f"Null 总数: {null_count}/{n_total} ({null_count/n_total*100:.1f}%)")
    if args.mode == 'ensemble':
        logger.info(f"Retry 触发统计: {dict(retry_count)}")
    top_preds = final_pred_stats.most_common(10)
    logger.info(f"最终预测分布 Top10: {top_preds}")

    logging.getLogger().removeHandler(fh)
    fh.close()


if __name__ == '__main__':
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    run(args)
