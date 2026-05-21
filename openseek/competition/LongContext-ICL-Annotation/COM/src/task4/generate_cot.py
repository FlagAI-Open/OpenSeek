#!/usr/bin/env python3
"""generate_cot.py — Task 4 离线 CoT + Think 生成

设计要点：
- **Prompt 不暴露 GT**：只给输入字符串列表与拼接规则，避免 think 中
  出现 ``"the user said the answer is ..."`` / ``"expected output is ..."``
  等锚点句污染。
- **答案校验（金标准）**：从 ``<label>...</label>`` 提取带空格预测，与代码
  按 ``' '.join(input_list)`` 计算的带空格 GT 逐字符严格比对，通过才采纳 think。
- **3 次重试 + 温度阶梯**：0.0 → 0.3 → 0.6
- **scrub_think**：清洗残留锚点（"the user" / "expected output" / "they want" / "answer is"）
- **validate_think**：长度 30–4000 + 命中拼接 / 大小写 / 空格相关关键词任一

输出格式（与其它 task cot_data 对齐）::

    {
        "id": "...",
        "input": "['get', 'B', 'have']",
        "output": ["getBhave"],
        "cot":   "[1] \"get\" → get\\n...\\n<label>get B have</label>",
        "think": "<Qwen3-4B reasoning content（已清洗）>"
    }

注：
- 此脚本依赖 vLLM 服务（端点见 :data:`common.paths.VLLM_BASE_URL`）已启动；
- 因为需要拿原始的 ``reasoning_content`` 作为 think 字段，这里直接使用 OpenAI SDK，
  不复用 ``common.llm_client.annotate_nvidia``（后者已抽走原始字段）。

运行::

    conda activate flagscale
    python generate_cot.py --limit 10          # 快速测试
    python generate_cot.py                     # 全量
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
from method import generate_cot_task4  # noqa: E402
from common.paths import (  # noqa: E402
    VLLM_BASE_URL as DEFAULT_BASE_URL,
    VLLM_MODEL_ID as DEFAULT_MODEL,
    task_cot_dir,
    task_data_file,
    task_log_dir,
)

_client = OpenAI(api_key="EMPTY", base_url=DEFAULT_BASE_URL)


# ============================================================
# 辅助工具
# ============================================================

def build_spaced_gt(input_text: str) -> Optional[str]:
    """从字符串列表输入构造带空格 GT（模型在线应当输出的版本）。

    ``['get', 'B', 'have']`` → ``'get B have'``

    含内部空格的元素原样保留：
    ``['hello world', 'foo']`` → ``'hello world foo'``
    """
    try:
        strings = ast.literal_eval(input_text.strip())
        if not isinstance(strings, list):
            return None
        return ' '.join(str(s) for s in strings)
    except Exception:
        return None


# ============================================================
# Prompt 构造（不暴露 GT）
# ============================================================

def build_open_prompt(input_text: str) -> str:
    """只给输入列表，让模型自己完成带空格拼接。

    关键约束（防 GT 污染）：
    - **绝不暴露**任何 "expected output / answer / result" 字样
    - 不使用 "they want / user said" 等指代性描述
    - 任务以第二人称直接下达，让模型视角是"我在做这个任务"而非"我在解释某个答案"
    """
    return (
        "Task: concatenate a list of strings into a single line where "
        "consecutive elements are joined by exactly ONE space.\n"
        "\n"
        "Rules:\n"
        "1. Take the first element as-is.\n"
        "2. For each following element, append it after exactly ONE space, "
        "copying every character verbatim — preserve case and punctuation.\n"
        "3. If an element itself contains spaces, keep those internal "
        "spaces intact; only one single space separates consecutive "
        "elements.\n"
        "4. Think step by step, then wrap the final spaced chain in "
        "<label>...</label>.\n"
        "\n"
        f"Input list: {input_text}\n"
    )


# ============================================================
# 答案提取（从 <label>...</label> 中取带空格预测）
# ============================================================

_LABEL_RE = re.compile(r'<label>(.*?)</label>', re.DOTALL)


def extract_spaced_prediction(content: str) -> Optional[str]:
    """从模型 content 中提取 ``<label>...</label>`` 内文本（不 strip 内部空格）。"""
    if not content:
        return None
    m = _LABEL_RE.search(content)
    if not m:
        return None
    # 只去除前后整段 whitespace，不动内部空格（这是带空格 GT 的核心）
    return m.group(1).strip('\n\r\t ')


# ============================================================
# think 二次清洗（删除潜在的 GT 锚点句）
# ============================================================

_SCRUB_PATTERNS = [
    re.compile(r'^.*\bthe user\b.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^.*\b(expected output|desired output|expected answer)\b.*$',
               re.IGNORECASE | re.MULTILINE),
    re.compile(r'^.*\b(they want|user said|user provided|user mentioned)\b.*$',
               re.IGNORECASE | re.MULTILINE),
    re.compile(r'^.*\bthe answer is\b.*$', re.IGNORECASE | re.MULTILINE),
]


def scrub_think(think: str) -> str:
    """删除 think 中残留的 GT 锚点句（兜底，理论上不应出现）。"""
    if not think:
        return think
    cleaned = think
    for pat in _SCRUB_PATTERNS:
        cleaned = pat.sub('', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


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
# think 字段提取与验证
# ============================================================

def pick_think_text(content: str, reasoning: str) -> str:
    """优先使用 reasoning_content，其次 content 中去掉 ``<think>`` 标签后的正文。"""
    if reasoning:
        return reasoning
    if not content:
        return ""
    m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content


# Task 4 语义相关关键词（命中任意一个即视为"在讲这个任务"）
_TASK4_THINK_KEYWORDS = (
    'concat', 'join', 'order', 'append',           # 拼接动作
    'preserve', 'exact', 'case', 'punctuation',    # 核心约束
    'space', 'between',                            # 带空格策略
    'element', 'each', 'string',                   # 通用领域词
)


def validate_think(think: str) -> bool:
    """宽松验证 think 字段可用性：
    - 非空
    - 长度 30–4000 字符
    - 命中 :data:`_TASK4_THINK_KEYWORDS` 中任一关键词
    """
    if not think:
        return False
    length = len(think)
    if length < 30 or length > 4000:
        return False
    lower = think.lower()
    return any(kw in lower for kw in _TASK4_THINK_KEYWORDS)


# ============================================================
# 单条样本处理
# ============================================================

_TEMPERATURE_LADDER = (0.0, 0.3, 0.6)


def generate_for_example(ex: dict, max_retries: int = 3) -> dict:
    """不暴露 GT 让模型自推，``<label>`` 预测与代码计算的带空格 GT 完全一致才采纳 think。"""
    gt = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
    input_text = ex['input']

    # cot 程序化生成，100% 正确（带空格版）
    cot_body = generate_cot_task4(input_text, gt)

    # 构造带空格 GT（代码端金标准，用于校验模型输出）
    spaced_gt = build_spaced_gt(input_text)
    if not spaced_gt:
        logger.warning(f"无法解析 input 为列表, 跳过 think 生成: id={ex.get('id')}")
        return {'cot': cot_body, 'think': ''}

    prompt = build_open_prompt(input_text)
    think = ""
    for attempt in range(max_retries):
        temp = _TEMPERATURE_LADDER[min(attempt, len(_TEMPERATURE_LADDER) - 1)]
        content, reasoning = call_qwen3_4b(prompt, temperature=temp)
        if content is None and reasoning is None:
            time.sleep(1)
            continue

        # 1. 金标准：<label> 中的带空格预测必须等于代码计算的 spaced_gt
        pred = extract_spaced_prediction(content or "")
        if pred != spaced_gt:
            logger.warning(
                f"  [attempt {attempt+1} T={temp}] 答案校验失败 "
                f"pred={repr(pred)[:80]} expect={repr(spaced_gt)[:80]}"
            )
            continue

        # 2. 提取 think 并做 scrub + validate
        candidate = pick_think_text(content or "", reasoning or "")
        candidate = scrub_think(candidate)
        if validate_think(candidate):
            think = candidate
            break
        logger.warning(
            f"  [attempt {attempt+1} T={temp}] think validate 失败 "
            f"(len={len(candidate)}): '{candidate[:80]}'"
        )

    return {'cot': cot_body, 'think': think}


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Task 4 CoT + Think 生成')
    parser.add_argument('--limit', type=int, default=0,
                        help='仅处理前 N 条（0=全量）')
    parser.add_argument('--data', type=str, default='',
                        help='输入原始数据路径（默认 common.paths.task_data_file(4)）')
    parser.add_argument('--output', type=str, default='',
                        help='输出带 cot+think 的数据路径（默认 task4/cot_data/openseek-4_*_with_cot.json）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有输出续跑（跳过已有非空 think 的示例）')
    parser.add_argument('--concurrency', type=int, default=3,
                        help='并发请求数（默认 3）')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='每完成 N 条保存一次')
    args = parser.parse_args()

    # 路径绝对化（默认走 common.paths）
    data_path = args.data or task_data_file(4)
    out_path = args.output or os.path.join(
        task_cot_dir(4), 'openseek-4_conala_concat_strings_with_cot.json')
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(_CUR_DIR, data_path))
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(os.path.join(_CUR_DIR, out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 日志文件
    log_dir = task_log_dir(4)
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

    # 构建待处理的 pending 列表：跳过已有 think 的；limit 控制上限
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

    if args.concurrency <= 1:
        for ex in pending:
            gt = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            logger.info(f"[{processed+1}/{to_process}] id={ex['id']} "
                        f"input={ex['input'][:60]} GT={gt[:40]}")
            _, result, dt = process_one(ex)
            ex['cot'] = result['cot']
            ex['think'] = result['think']
            if result['think']:
                success += 1
                logger.info(f"  ✅ think ({len(result['think'])} chars, {dt:.1f}s)")
            else:
                failed += 1
                logger.warning(f"  ❌ think 生成失败 ({dt:.1f}s)")
            processed += 1
            if processed % args.save_interval == 0:
                save_progress()
                elapsed = time.time() - t_start
                avg = elapsed / processed
                logger.info(f"  💾 已保存 (成功={success} 失败={failed}) "
                            f"| {avg:.1f}s/条 | ETA={avg*(to_process-processed)/60:.0f}min")
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
                    logger.info(f"[{processed+1}/{to_process}] id={ex['id'][:20]} "
                                f"GT={gt[:30]} ✅ ({len(result['think'])}ch, {dt:.1f}s)")
                else:
                    failed += 1
                    logger.warning(f"[{processed+1}/{to_process}] id={ex['id'][:20]} "
                                   f"GT={gt[:30]} ❌ ({dt:.1f}s)")
                processed += 1
                if processed % args.save_interval == 0:
                    save_progress()
                    elapsed = time.time() - t_start
                    avg = elapsed / processed
                    remaining = to_process - processed
                    logger.info(f"  💾 已保存 (成功={success} 失败={failed}) "
                                f"| {avg:.1f}s/条均值 | ETA={avg*remaining/60:.0f}min")

    # 最终保存
    out = dict(data)
    out['examples'] = examples
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"完成! {elapsed:.0f}s, 共处理 {processed} 条")
    logger.info(f"  成功: {success}")
    logger.info(f"  失败(think 空): {failed}")
    logger.info(f"  跳过: {skipped}")
    if processed > 0:
        logger.info(f"  成功率: {success/processed*100:.1f}%")
    logger.info(f"  输出: {out_path}")


if __name__ == '__main__':
    main()
