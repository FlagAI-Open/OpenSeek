"""
main.py — Task 4 (conala_concat_strings) 预测入口

依赖：
- ``method.py``                     : Task 4 示例选择器 + Prompt + 后处理
- ``common.llm_client.annotate_nvidia``: 共享的 vLLM/OpenAI 调用封装
- ``common.paths``                  : 统一路径中心（数据/模型/输出/CoT）

运行示例::

    conda activate flagscale
    bash run.sh                           # 默认：全量 500 样本
    python main.py --max_samples 20       # 快速测试 20 条
"""

import json
import os
import re
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
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task4
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for p in (_CUR_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from method import (  # noqa: E402  (sys.path 设置后才能 import)
    Task4ExampleSelector,
    build_task4_prompt,
    postprocess_task4,
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

# ---- Label 包装剥离模式（thinking 模式 artifact 容错） ----
# 通用标签：左括号 [<\[] + label + 右括号 [>\]]，兼容所有畸形：
#   <label>...</label>  [label]...[/label]  [label>...</label>
#   <label]...[/label>  [label>...[/label>  等 16 种组合
_LABEL_OPEN = r'[<\[]\s*label\s*[>\]]'
_LABEL_CLOSE = r'[<\[]\s*/\s*label\s*[>\]]'

_LABEL_PATTERNS = [
    # 通用配对：开标签 + 内容 + 闭标签
    re.compile(rf'{_LABEL_OPEN}\s*(.+?)\s*{_LABEL_CLOSE}', re.DOTALL | re.IGNORECASE),
    # \boxed{...}
    re.compile(r'\\boxed\{(.+?)\}', re.DOTALL),
]

# 单边残留 prefix/suffix（配对正则都没命中时的兜底）
_LABEL_PREFIX = re.compile(rf'^\s*{_LABEL_OPEN}\s*', re.IGNORECASE)
_LABEL_SUFFIX = re.compile(rf'\s*{_LABEL_CLOSE}\s*$', re.IGNORECASE)


def extract_label_content(text: str):
    """从任意文本里抽出 label 包装内的纯文本；未命中返回 None。"""
    if not text:
        return None
    for pat in _LABEL_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip('\n\r\t ')
    return None


def strip_label_wrapper(prediction: str) -> str:
    """剥掉 label 外壳：先试正规配对，失败则独立剥两边残留 prefix/suffix。"""
    if not prediction:
        return prediction
    inner = extract_label_content(prediction)
    if inner is not None:
        return inner
    # 单边剥离（防止 [label] / </label> 孤出等畸形）
    stripped = prediction.strip()
    stripped = _LABEL_PREFIX.sub('', stripped)
    stripped = _LABEL_SUFFIX.sub('', stripped)
    return stripped


# ---- 默认路径与配置（统一从 common.paths 派生）----
TASK_ID = 4
DATA_FILE_WITH_COT = task_cot_dir(TASK_ID) + '/openseek-4_conala_concat_strings_with_cot.json'
DATA_FILE_RAW      = task_data_file(TASK_ID)
OUTPUT_PREFIX      = FINAL_OUTPUT_DIR + '/'
TOKENIZER_PATH     = MODEL_DIR

MAX_CONTEXT_TOKENS = 31000   # 单条 ICL prompt 的 token 上限（赛题硬约束 ≥ 30K，留 1K 余量）
MIN_CONTEXT_TOKENS = 30000   # 单条 ICL prompt 的 token 下限（赛题硬约束 ≥ 30K，必须达标）
MAX_TOKENS = 4096            # 拼接结果 + thinking 模式需更多生成空间
TIMEOUT = 600                # 单次请求超时
TEMPERATURE = 0.0            # 确定性输出
REPETITION_PENALTY = 1.1     # 抑制死循环复读
ENABLE_THINKING = True       # 配合 think 示例使用 Qwen3 thinking 模式
DEFAULT_CONCURRENCY = 1      # 默认串行（vLLM 显存吃紧时建议保持 1）


def _validate_concat(prediction):
    """轻量校验：非空字符串即采用，否则置空。

    Task 4 输出空间是任意字符串拼接，没有结构化校验项，仅做空值/纯空白过滤。
    """
    if prediction is None:
        return None
    s = prediction.strip('\n\r\t ')
    if not s:
        return None
    return s


def parser_args():
    parser = argparse.ArgumentParser(description='Task 4 预测（conala_concat_strings + CoT + think）')
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
    parser.add_argument('--verbose_prompt', action='store_true', default=False,
                        help='开启详细 prompt 日志（打印 PROMPT_HEAD/TAIL，默认关闭）')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='静音模式：仅打印汇总级日志')
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
    selector = Task4ExampleSelector(
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
                f"max_tokens={MAX_TOKENS}, timeout={TIMEOUT}s, "
                f"repetition_penalty={REPETITION_PENALTY}")

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

    # ---- 日志文件（保存到 task4/logs/）----
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
    postproc_count = 0
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
        nonlocal null_count, postproc_count
        text2annotate = test_sample['input']
        tag = f"[{idx+1}/{len(test_samples)}]"
        sample_t0 = time.time()

        # 示例选择
        examples_str = selector.select(icl_examples, text2annotate)
        n_examples = sum(1 for ln in examples_str.splitlines()
                         if ln.startswith('# ['))

        # Prompt 构建
        input_prompt = build_task4_prompt(task_description, text2annotate, examples_str)
        prompt_chars = len(input_prompt)
        try:
            prompt_tokens = len(tokenizer.encode(input_prompt, add_special_tokens=False))
        except Exception:
            prompt_tokens = -1

        # 详细日志：样本开始（单行，避免并发交错）
        logger.info(f"{tag} START id={test_sample['id']} "
                    f"examples={n_examples} prompt_chars={prompt_chars} "
                    f"prompt_tokens={prompt_tokens} budget={args.max_context_tokens}")
        logger.info(f"{tag} INPUT: {text2annotate}")
        if verbose:
            head = input_prompt[:400].replace('\n', ' ⏎ ')
            tail = input_prompt[-600:].replace('\n', ' ⏎ ')
            logger.info(f"{tag} PROMPT_HEAD: {head}")
            logger.info(f"{tag} PROMPT_TAIL: {tail}")

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

        # thinking 模式下 content 可能为空——从原始输出兼容提取 label
        if prediction is None and raw:
            inner = extract_label_content(raw)
            if inner is not None:
                prediction = inner
                logger.info(f"{tag} Fallback 从原始输出提取 label: '{prediction[:120]}'")

        # label 外壳剥离（兼容 [label]...[/label] / \boxed{...} 等 artifact）
        if prediction:
            before_strip = prediction
            prediction = strip_label_wrapper(prediction)
            if before_strip != prediction:
                logger.info(f"{tag} 剥离 label 外壳: "
                            f"'{before_strip[:80]}' → '{prediction[:80]}'")

        # 后处理 — 智能去空格（模型带空格拼接，代码去空格）
        local_postproc = False
        if prediction:
            before = prediction
            prediction = postprocess_task4(prediction, text2annotate)
            if before != prediction:
                local_postproc = True
                logger.info(f"{tag} Task4 后处理: "
                            f"'{before[:80]}' → '{prediction[:80]}'")

        # 轻量校验：非空字符串
        prediction = _validate_concat(prediction)
        if prediction is None:
            prediction = ""

        sample_elapsed = time.time() - sample_t0

        # 详细日志：样本结果（单行）
        if prediction == "":
            logger.warning(f"{tag} ⚠️  null预测 raw_len={len(raw)} "
                           f"raw_tail={raw[-180:]!r} ({sample_elapsed:.1f}s)")
        else:
            logger.info(f"{tag} pred='{prediction[:120]}' raw_len={len(raw)} "
                        f"({sample_elapsed:.1f}s)")

        # 线程安全地更新统计
        with stat_lock:
            if prediction == "":
                null_count += 1
            if local_postproc:
                postproc_count += 1

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
        for i in tqdm(iterator, desc=f'Task {TASK_ID}: concat_strings'):
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
                        f"null={null_count} postproc={postproc_count} | "
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
    logger.info(f"后处理触发: {postproc_count}/{total}")

    logging.getLogger().removeHandler(fh)
    fh.close()


if __name__ == '__main__':
    args = parser_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    run(args)
