#!/usr/bin/env python3
"""generate_cot.py — Task 3 离线 CoT + Think 生成

设计要点：
- **Prompt 不暴露 GT**：只给输入 list 与 Collatz 规则，避免 think
  里出现 ``"the user said the answer is [...]"`` 类锚点句污染。
- **答案校验（金标准）**：从 ``<label>[...]</label>`` 提取整数列表，与 GT 列表
  逐元素严格比对一致才采纳该次 think；否则继续下一轮重试。
- **3 次重试 + 温度阶梯**：0.0 → 0.3 → 0.6（验证失败自动升温）
- **scrub_think**：清洗残留 ``"user said"`` / ``"answer is"`` 锚点
- **validate_think**：长度 ≥ 40 + 命中 even/odd/collatz/half/triple/3n+1 任一关键词

输出格式（与其它 task cot_data 对齐）::

    {
        "id": "...",
        "input": "[72, 29, 49]",
        "output": ["[36, 88, 148]"],
        "cot":   "72 even → 36; 29 odd → 88; 49 odd → 148\\n<label>[36, 88, 148]</label>",
        "think": "<Qwen3-4B reasoning content（已清洗）>"
    }

注：
- 此脚本依赖 vLLM 服务（端点见 :data:`common.paths.VLLM_BASE_URL`）已启动；
- 因为需要拿原始的 ``reasoning_content`` 作为 think 字段，这里直接使用 OpenAI SDK，
  不复用 ``common.llm_client.annotate_nvidia``（后者已抽走原始字段）。

运行：
    conda activate flagscale
    python generate_cot.py --limit 10          # 快速测试
    python generate_cot.py                     # 全量（3997 条）
    python generate_cot.py --resume            # 续跑
    python generate_cot.py --concurrency 3     # 并发
"""

import json
import re
import os
import sys
import time
import ast
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from openai import OpenAI

# 同目录提供 method 中的程序化 cot 生成器；上级 common 提供路径中心
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))
for _p in (_CUR_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from method import generate_cot_task3  # noqa: E402
from common.paths import (  # noqa: E402
    VLLM_BASE_URL as DEFAULT_BASE_URL,
    VLLM_MODEL_ID as DEFAULT_MODEL,
    task_cot_dir,
    task_data_file,
    task_log_dir,
)

_client = OpenAI(api_key="EMPTY", base_url=DEFAULT_BASE_URL)


# ============================================================
# Prompt 构造（不暴露 GT）
# ============================================================

def build_open_prompt(input_text: str) -> str:
    """只给输入 list，让模型自己推到结果列表，最后以 ``<label>[...]</label>`` 收尾。

    - 模型用 thinking mode 产出 ``reasoning_content``（即 think 字段）
    - ``content`` 必须含 ``<label>[...]</label>``，会被解析并与 GT 列表逐元素比对
    - 绝不暴露 GT，确保 think 无 "user said answer is ..." 类污染
    """
    return (
        "### Task\n"
        "Given a list of integers, apply the Collatz step to EACH element "
        "independently and output the resulting list. Think step by step.\n\n"
        "### Reasoning pattern\n"
        "1. For each element n in the input list (keep the ORIGINAL order):\n"
        "   - If n is even → n // 2\n"
        "   - If n is odd  → 3 * n + 1\n"
        "2. Collect the per-element results into the output list.\n"
        "3. Verify the output list has the SAME length as the input list.\n\n"
        "### Output format (MUST end with the label tag)\n"
        "n1 even/odd → result1; n2 even/odd → result2; ... ; nN even/odd → resultN\n"
        "<label>[result1, result2, ..., resultN]</label>\n\n"
        "### Input\n"
        f"{input_text}\n"
    )


# ============================================================
# Qwen3-4B 调用（thinking mode）
# ============================================================

def call_qwen3_4b(prompt: str, temperature: float = 0.0,
                  max_tokens: int = 2048,
                  repetition_penalty: float = 1.1
                  ) -> tuple:
    """返回 (content, reasoning_content)；失败返回 (None, None)。"""
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


def _parse_list(text: str) -> Optional[list]:
    """将字符串解析为整数列表；失败返回 None。"""
    if not text:
        return None
    try:
        obj = ast.literal_eval(text.strip())
        if isinstance(obj, list):
            return [int(x) for x in obj]
    except Exception:
        pass
    return None


def extract_label_answer(content: str, reasoning: str) -> Optional[list]:
    """从 content 中提取 ``<label>[...]</label>``；降级到 reasoning 中的同样标签。

    返回解析后的整数列表；失败返回 None。
    """
    for src in (content or "", reasoning or ""):
        if not src:
            continue
        # 允许 <label>...</label> 跨行匹配
        m = re.search(r'<label>\s*(\[[^<]*?\])\s*</label>', src, re.DOTALL)
        if m:
            parsed = _parse_list(m.group(1))
            if parsed is not None:
                return parsed
    # 兜底：最后一个裸列表 [a, b, c]
    for src in (content or "", reasoning or ""):
        if not src:
            continue
        ms = re.findall(r'\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]', src)
        if ms:
            parsed = _parse_list(ms[-1])
            if parsed is not None:
                return parsed
    return None


def scrub_think(think: str) -> str:
    """二次防护：删掉任何可能暗示 ``"user said answer is X"`` 的锚点句。"""
    if not think:
        return think
    patterns = [
        r'(?i)the\s+user\s+(said|mentioned|told|provided|claimed).*?(\.|\n|$)',
        r'(?i)(the|user\'s)\s+(answer|result)\s+is\s+\[.*?\].*?(\.|\n|$)',
        r'(?i)user\s+mentioned\s+that.*?(\.|\n|$)',
    ]
    cleaned = think
    for p in patterns:
        cleaned = re.sub(p, '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def validate_think(think: str) -> bool:
    """think 可用性校验：非空、不过短、含推理关键字。

    不设上限：由上层 ``pred==GT`` 校验作为金标准，长度不重要。
    """
    if not think:
        return False
    if len(think) < 40:
        return False
    lower = think.lower()
    keywords = ('even', 'odd', 'collatz', 'half', 'triple',
                'divid', 'multipl', 'divide by two', '3n+1', '3 * n + 1')
    return any(kw in lower for kw in keywords)


# ============================================================
# 单条样本处理
# ============================================================

def generate_for_example(ex: dict, max_retries: int = 3) -> dict:
    """不暴露 GT 让模型自推，验证答案列表完全一致才采纳 think。"""
    gt_raw = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
    gt_list = _parse_list(str(gt_raw))
    input_text = ex['input']

    # cot 程序化生成，100% 正确
    cot_body = generate_cot_task3(input_text, gt_raw)

    # think 由模型产出（无 GT 暗示 + 答案校验）
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
        if (pred is not None and gt_list is not None
                and pred == gt_list and validate_think(candidate)):
            think = candidate
            verified = True
            break
        logger.warning(
            f"  [attempt {attempt+1}/{max_retries}] pred={pred} vs GT={gt_list} "
            f"(think_len={len(candidate)})"
        )

    return {'cot': cot_body, 'think': think, 'attempts': attempts,
            'verified': verified, 'last_pred': last_pred}


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Task 3 CoT + Think 生成')
    parser.add_argument('--limit', type=int, default=0,
                        help='仅处理前 N 条（0=全量）')
    parser.add_argument('--data', type=str, default='',
                        help='输入原始数据路径（默认 common.paths.task_data_file(3)）')
    parser.add_argument('--output', type=str, default='',
                        help='输出带 cot+think 的数据路径（默认 task3/cot_data/openseek-3_*_with_cot.json）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有输出续跑（跳过已有非空 think 的示例）')
    parser.add_argument('--concurrency', type=int, default=3,
                        help='并发请求数（默认 3）')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='每完成 N 条保存一次')
    args = parser.parse_args()

    data_path = args.data or task_data_file(3)
    out_path = args.output or os.path.join(
        task_cot_dir(3), 'openseek-3_collatz_conjecture_with_cot.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    log_dir = task_log_dir(3)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'generate_cot.log')
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
                        f"input={ex['input'][:60]} GT={gt[:60]}")
            _, result, dt = process_one(ex)
            ex['cot'] = result['cot']
            ex['think'] = result['think']
            if result['think']:
                success += 1
                if result.get('verified'):
                    verified_count += 1
                logger.info(f"  ✅ think ({len(result['think'])} chars, "
                            f"attempts={result.get('attempts')}, {dt:.1f}s)")
            else:
                failed += 1
                logger.warning(f"  ❌ think 降级（验证未过, last_pred="
                               f"{result.get('last_pred')}, {dt:.1f}s）")
            processed += 1
            if processed % args.save_interval == 0:
                save_progress()
                elapsed = time.time() - t_start
                avg = elapsed / processed
                logger.info(f"  💾 已保存 (成功={success} 验证通过={verified_count} "
                            f"降级={failed}) | {avg:.1f}s/条 | "
                            f"ETA={avg*(to_process-processed)/60:.0f}min")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_one, ex): ex for ex in pending}
            for fut in as_completed(futures):
                try:
                    ex, result, dt = fut.result()
                except Exception as e:
                    ex = futures[fut]
                    logger.error(f"  ❌ worker 异常 id={ex['id']}: {e}")
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
                                f"GT={gt[:40]} ✅ ({len(result['think'])}ch, "
                                f"attempts={result.get('attempts')}, {dt:.1f}s)")
                else:
                    failed += 1
                    logger.warning(f"[{processed+1}/{to_process}] id={ex['id'][:20]} "
                                   f"GT={gt[:40]} ❌ 降级 last_pred="
                                   f"{result.get('last_pred')} ({dt:.1f}s)")
                processed += 1
                if processed % args.save_interval == 0:
                    save_progress()
                    elapsed = time.time() - t_start
                    avg = elapsed / processed
                    remaining = to_process - processed
                    logger.info(f"  💾 已保存 (成功={success} 验证通过={verified_count} "
                                f"降级={failed}) | {avg:.1f}s/条均值 | "
                                f"ETA={avg*remaining/60:.0f}min")

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
