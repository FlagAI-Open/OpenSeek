#!/usr/bin/env python3
"""generate_cot.py — Task 1 离线 CoT + think 生成

每条 ICL 示例产出两个字段：
- ``cot``  ：用 ``method.generate_cot_task1`` 程序化生成（排序 → 逐对 diff → Min → label，
  100% 正确）。
- ``think``：用 Qwen3-4B thinking 模式自我推理产出。
  - **不在 prompt 中暴露示例答案**，避免推理阶段出现 ``"The user said the answer is X"``
    类锚点句进而引发测试样本死循环。
  - 从 ``content`` 中提取 ``<label>X</label>``，与示例标注答案一致才接受；不一致最多重试 3 次
    （温度阶梯 0.0 → 0.3 → 0.6）。
  - 重试耗尽 → ``think`` 留空，该示例只保留 ``cot``，不污染 ICL。
  - ``think`` 取自 ``reasoning_content``（纯推理，不含答案暗示）。

输出格式（与其它 task cot_data 对齐）::

    {
        "id": "...",
        "input": "[59, 26, -96, -30]",
        "output": ["33"],
        "cot": "\\nSorted (4): [-96, -30, 26, 59]\\n...\\n<label>33</label>",
        "think": "<Qwen3-4B reasoning content (纯思考, 不含答案暗示)>"
    }

注：
- 此脚本依赖 vLLM 服务（端点见 :data:`common.paths.VLLM_BASE_URL`）已启动；
- 因为需要拿原始的 ``reasoning_content`` 作为 think 字段，这里直接使用 OpenAI SDK，
  不复用 ``common.llm_client.annotate_nvidia``（后者已抽走原始字段）。

运行：
    conda activate flagscale
    python generate_cot.py --limit 10          # 快速测试
    python generate_cot.py                     # 全量（5500 条）
    python generate_cot.py --resume            # 续跑
"""

import json
import re
import os
import sys
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from openai import OpenAI

# 同目录 method 提供 generate_cot_task1；上级 common 提供路径中心
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))
for _p in (_CUR_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from method import generate_cot_task1  # noqa: E402
from common.paths import (  # noqa: E402
    VLLM_BASE_URL as DEFAULT_BASE_URL,
    VLLM_MODEL_ID as DEFAULT_MODEL,
    task_cot_dir,
    task_data_file,
    task_log_dir,
)

_client = OpenAI(api_key="EMPTY", base_url=DEFAULT_BASE_URL)


# ============================================================
# Open-ended prompt（不透露答案）
# ============================================================

def build_open_prompt(input_text: str) -> str:
    """只给输入 list，让模型自己推到答案，最后以 ``<label>X</label>`` 收尾。

    - 模型用 thinking 模式产出 reasoning_content（即 think 字段）
    - content 必须含 ``<label>X</label>``，``X`` 会被程序提取并与示例标注答案对比
    - prompt 中不出现答案，确保 think 无 'user said answer is ...' 污染
    """
    return (
        "### Task\n"
        "Given a list of integers, find the MINIMUM absolute difference "
        "between any two integers in the list. Think step by step.\n\n"
        "### Reasoning pattern\n"
        "1. Sort the list in ascending order.\n"
        "2. For each consecutive pair (a, b) in the sorted list, compute |b - a|.\n"
        "3. The minimum of these differences is the answer.\n\n"
        "### Output format (MUST end with the label tag)\n"
        "Sorted (N): [sorted list]\n"
        "|a[1] - a[0]| = diff\n"
        "... (one line per consecutive pair)\n"
        "Min = minimum_difference\n"
        "<label>minimum_difference</label>\n\n"
        "### Input\n"
        f"{input_text}\n"
    )


# ============================================================
# Qwen3-4B 调用（thinking mode）
# ============================================================

def call_qwen3_4b(prompt: str, temperature: float = 0.0,
                  max_tokens: int = 2048,
                  repetition_penalty: float = 1.1
                  ) -> tuple[Optional[str], Optional[str]]:
    """返回 (content, reasoning_content)"""
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": True},
        "repetition_penalty": repetition_penalty,
    }
    try:
        response = _client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
            timeout=300,
        )
        message = response.choices[0].message
        content = (message.content or "").strip()
        reasoning = (getattr(message, 'reasoning_content', None) or "").strip()
        return content, reasoning
    except Exception as e:
        logger.warning(f"API 调用失败: {e}")
        return None, None


# ============================================================
# think 字段提取与答案校验
# ============================================================

def pick_think_text(content: str, reasoning: str) -> str:
    """优先使用 reasoning_content，其次 content 中去掉 <think> 标签后的正文。"""
    if reasoning:
        return reasoning
    if not content:
        return ""
    m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content


def extract_label_answer(content: str, reasoning: str) -> Optional[str]:
    """从 content 中提取 <label>X</label>；降级到 reasoning 尾部的 label / Min = X 等形式。"""
    for src in (content or "", reasoning or ""):
        if not src:
            continue
        m = re.search(r'<label>\s*(-?\d+)\s*</label>', src)
        if m:
            return m.group(1)
    # 兜底：最后一个 Min = N
    for src in (content or "", reasoning or ""):
        if not src:
            continue
        ms = re.findall(r'Min\s*=\s*(-?\d+)', src)
        if ms:
            return ms[-1]
    return None


def scrub_think(think: str) -> str:
    """二次防护：删掉任何可能暗示 'user said answer is X' 的锚点句（理论上不应该出现）。"""
    if not think:
        return think
    patterns = [
        r'(?i)the\s+user\s+(said|mentioned|told|provided|claimed).*?(\.|\n|$)',
        r"(?i)(the|user's)\s+answer\s+is\s+-?\d+.*?(\.|\n|$)",
        r'(?i)user\s+mentioned\s+that.*?(\.|\n|$)',
    ]
    cleaned = think
    for p in patterns:
        cleaned = re.sub(p, '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def validate_think(think: str) -> bool:
    """think 可用性校验：非空、不过短（防空响应）、含推理关键字。
    不设上限：由上层答案校验作为金标准，长度不重要。
    """
    if not think:
        return False
    if len(think) < 40:
        return False
    lower = think.lower()
    keywords = ('sort', 'sorted', 'diff', 'minimum', 'smallest', 'absolute', 'closest')
    return any(kw in lower for kw in keywords)


# ============================================================
# 单条样本处理
# ============================================================

def generate_for_example(ex: dict, max_retries: int = 3) -> dict:
    """不透露答案，让模型自己推，验证答案正确才采纳 think。

    返回 dict 带 'cot', 'think', 'attempts', 'verified'（便于统计）。
    """
    gt = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
    gt_str = str(gt).strip()
    input_text = ex['input']

    # cot 程序化生成，100% 正确
    cot_body = generate_cot_task1(input_text, gt)

    # think 由模型产出（prompt 不暴露答案 + 输出答案校验）
    prompt = build_open_prompt(input_text)
    temps = [0.0, 0.3, 0.6]
    think = ""
    verified = False
    attempts = 0
    last_pred = None
    for attempt in range(max_retries):
        attempts = attempt + 1
        temp = temps[attempt] if attempt < len(temps) else 0.7
        content, reasoning = call_qwen3_4b(prompt, temperature=temp)
        if content is None and reasoning is None:
            time.sleep(1)
            continue
        pred = extract_label_answer(content or "", reasoning or "")
        last_pred = pred
        candidate = pick_think_text(content or "", reasoning or "")
        candidate = scrub_think(candidate)
        if pred is not None and pred == gt_str and validate_think(candidate):
            think = candidate
            verified = True
            break
        logger.warning(
            f"  [attempt {attempt+1}/{max_retries}] pred={pred!r} vs answer={gt_str!r} "
            f"(len={len(candidate)})"
        )

    return {'cot': cot_body, 'think': think, 'attempts': attempts,
            'verified': verified, 'last_pred': last_pred}


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Task 1 CoT + Think 生成')
    parser.add_argument('--limit', type=int, default=0,
                        help='仅处理前 N 条（0=全量）')
    parser.add_argument('--data', type=str,
                        default=task_data_file(1),
                        help='输入原始数据路径（默认指向 common.paths.task_data_file(1)）')
    parser.add_argument('--output', type=str,
                        default=task_cot_dir(1) + '/openseek-1_closest_integers_with_cot.json',
                        help='输出带 cot + think 的数据路径')
    parser.add_argument('--resume', action='store_true',
                        help='从已有输出续跑（跳过已有非空 think 的示例）')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='并发请求数（默认 1 = 串行）')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='每完成 N 条保存一次')
    args = parser.parse_args()

    # 路径绝对化（兼容用户传入相对路径）
    data_path = args.data if os.path.isabs(args.data) else \
        os.path.abspath(os.path.join(_CUR_DIR, args.data))
    out_path = args.output if os.path.isabs(args.output) else \
        os.path.abspath(os.path.join(_CUR_DIR, args.output))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 日志文件
    log_dir = task_log_dir(1)
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + '/generate_task1_cot.log'
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)
    logger.info(f"Log: {log_file}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {out_path}")

    with open(data_path) as f:
        data = json.load(f)
    examples = data['examples']
    total = len(examples)
    logger.info(f"加载 {total} 条示例")

    # 续跑：加载已有输出
    existing = {}
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            old = json.load(f)
        for ex in old.get('examples', []):
            if ex.get('think'):
                existing[ex['id']] = ex
        logger.info(f"续跑: 已有 {len(existing)} 条 think")

    processed = success = failed = skipped = 0
    t_start = time.time()
    save_lock = threading.Lock()

    pending = []
    for ex in examples:
        if ex['id'] in existing:
            ex['cot'] = existing[ex['id']].get('cot', '')
            ex['think'] = existing[ex['id']].get('think', '')
            skipped += 1
            continue
        pending.append(ex)
        if args.limit > 0 and len(pending) >= args.limit:
            break

    to_process = len(pending)
    logger.info(f"实际需处理 {to_process} 条，跳过 {skipped} 条，"
                f"并发数={args.concurrency}")

    def process_one(ex):
        t0 = time.time()
        result = generate_for_example(ex)
        dt = time.time() - t0
        return ex, result, dt

    def save_progress():
        with save_lock:
            out = dict(data)
            out['examples'] = examples
            tmp_path = out_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, out_path)

    verified_count = 0
    if args.concurrency <= 1:
        for ex in pending:
            gt = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            logger.info(f"[{processed+1}/{to_process}] id={ex['id']} "
                        f"input={ex['input'][:60]} answer={gt}")
            _, result, dt = process_one(ex)
            ex['cot'] = result['cot']
            ex['think'] = result['think']
            if result['think']:
                success += 1
                if result.get('verified'):
                    verified_count += 1
                logger.info(f"  OK think ({len(result['think'])} chars, "
                            f"attempts={result.get('attempts')}, {dt:.1f}s)")
            else:
                failed += 1
                logger.warning(f"  FAIL think 降级（验证未过, last_pred="
                               f"{result.get('last_pred')!r}, {dt:.1f}s）")
            processed += 1
            if processed % args.save_interval == 0:
                save_progress()
                elapsed = time.time() - t_start
                avg = elapsed / processed
                logger.info(f"  saved (success={success} verified={verified_count} "
                            f"degrade={failed}) | {avg:.1f}s/条 | "
                            f"ETA={avg*(to_process-processed)/60:.0f}min")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_one, ex): ex for ex in pending}
            for fut in as_completed(futures):
                try:
                    ex, result, dt = fut.result()
                except Exception as e:
                    ex = futures[fut]
                    logger.error(f"  worker 异常 id={ex['id']}: {e}")
                    ex['cot'] = ex.get('cot', '')
                    ex['think'] = ''
                    failed += 1
                    processed += 1
                    continue
                ex['cot'] = result['cot']
                ex['think'] = result['think']
                gt = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
                if result['think']:
                    success += 1
                    if result.get('verified'):
                        verified_count += 1
                    logger.info(f"[{processed+1}/{to_process}] id={ex['id'][:20]} "
                                f"answer={gt} OK ({len(result['think'])}ch, "
                                f"attempts={result.get('attempts')}, {dt:.1f}s)")
                else:
                    failed += 1
                    logger.warning(f"[{processed+1}/{to_process}] id={ex['id'][:20]} "
                                   f"answer={gt} FAIL last_pred="
                                   f"{result.get('last_pred')!r} ({dt:.1f}s)")
                processed += 1
                if processed % args.save_interval == 0:
                    save_progress()
                    elapsed = time.time() - t_start
                    avg = elapsed / processed
                    remaining = to_process - processed
                    logger.info(f"  saved (success={success} verified={verified_count} "
                                f"degrade={failed}) | {avg:.1f}s/条均值 | "
                                f"ETA={avg*remaining/60:.0f}min")

    # 最终保存
    out = dict(data)
    out['examples'] = examples
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"完成! {elapsed:.0f}s, 共处理 {processed} 条")
    logger.info(f"  think 验证通过: {verified_count}")
    logger.info(f"  think 非空总数: {success}")
    logger.info(f"  think 降级（空）: {failed}")
    logger.info(f"  跳过: {skipped}")
    if processed > 0:
        logger.info(f"  think 验证通过率: {verified_count/processed*100:.1f}%")
    logger.info(f"  输出: {out_path}")


if __name__ == '__main__':
    main()
