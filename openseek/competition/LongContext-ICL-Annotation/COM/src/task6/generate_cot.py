#!/usr/bin/env python3
"""generate_cot.py — Task 6 离线 CoT + Think 生成

设计要点：
- **Prompt 不暴露 GT**：只给两句话和 stated genre，让模型独立判断 sentence1 /
  sentence2 各自的 genre，避免 think 中出现 ``"label is N"`` 锚点污染。
- **答案校验（金标准）**：
  * 三行 ``stated genre / sentence1 genre / sentence2 genre`` 必须落在 10 类白名单内
  * 解析出的 stated genre 与 input 中的 stated 一致
  * ``<label>`` 与 gold label 一致
  * 语义一致性：
    - gold=Y: ``s1_genre == s2_genre == stated``
    - gold=N: 至少一个 ``s_genre != stated``
- **3 次重试 + 温度阶梯**：0.0 → 0.3 → 0.6
- **失败即空**：3 次重试均未通过校验时，``cot`` 与 ``think`` 一并置空
  （Selector 协议：空 CoT 不会注入 prompt，避免污染）

输出格式（与 Selector.select 自动注入兼容）::

    {
        "id": "openseek-6-xxx",
        "input": "Sentence 1: ... Sentence 2: ... Genre: xxx.",
        "output": ["Y" | "N"],
        "cot":   "stated genre: xxx\\nsentence1 genre: xxx\\nsentence2 genre: xxx",
        "think": "<Qwen3-4B reasoning content>"
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
    python generate_cot.py --concurrency 3     # 并发
"""

import json
import re
import os
import sys
import time
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from openai import OpenAI

# 同目录提供 method 常量；上级 common 提供路径中心
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))
for _p in (_CUR_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from method import TASK6_GENRE_SET  # noqa: E402
from common.paths import (  # noqa: E402
    VLLM_BASE_URL as DEFAULT_BASE_URL,
    VLLM_MODEL_ID as DEFAULT_MODEL,
    task_cot_dir,
    task_data_file,
    task_log_dir,
)

_client = OpenAI(api_key="EMPTY", base_url=DEFAULT_BASE_URL)


# ============================================================
# Genre 参考说明（用于 prompt）
# ============================================================

_GENRE_REFERENCE = (
    "- government: formal bureaucratic tone, agencies (GAO, EPA), regulations, federal programs\n"
    "- fiction: narrative storytelling, character names, dialogue tags (said, asked), emotional descriptions\n"
    "- telephone: spoken markers (yeah, uh, um, you know, gonna), conversational fragments, informal\n"
    "- travel: place descriptions, landmarks, tourism (island, temple, museum, beach), sightseeing\n"
    "- slate: opinion/analysis, journalistic commentary, cultural criticism, named public figures\n"
    "- 9/11: references to attacks, hijacking, FAA, NORAD, Pentagon, bin Laden, al Qaeda\n"
    "- oup: academic research, child development, textile/fabric industry, cognitive/social science\n"
    "- verbatim: linguistics discussion, word origins, etymology, language usage, dictionary\n"
    "- face-to-face: in-person conversation (yeah, okay, right, so, like), similar to telephone\n"
    "- letters: philanthropic appeals, donation requests, dear [recipient], fundraising, charity\n"
)


# ============================================================
# 输入解析辅助
# ============================================================

def extract_stated_genre(input_text: str) -> str:
    """从 input 末尾提取 Genre 值，如 'Genre: government.' -> 'government'。"""
    m = re.search(r'Genre:\s*([\w/\-]+)\.?\s*$', input_text, re.IGNORECASE)
    return m.group(1).strip().lower() if m else ''


def extract_sentences(input_text: str) -> tuple[str, str]:
    """提取 Sentence 1 / Sentence 2 纯文本。"""
    s1 = s2 = ''
    m1 = re.search(r'Sentence 1:\s*(.+?)(?=Sentence 2:|Genre:|$)', input_text,
                   re.DOTALL | re.IGNORECASE)
    if m1:
        s1 = m1.group(1).strip()
    m2 = re.search(r'Sentence 2:\s*(.+?)(?=Genre:|$)', input_text,
                   re.DOTALL | re.IGNORECASE)
    if m2:
        s2 = m2.group(1).strip()
    return s1, s2


# ============================================================
# Open-Ended Prompt 构造（不暴露 GT）
# ============================================================

def build_open_prompt(s1: str, s2: str, stated_genre: str) -> str:
    """让模型独立判断 sentence1/sentence2 各自属于哪个 genre。

    Y/N 样本共用同一 prompt；模型在推理阶段自行得出 <label>，
    脚本再用 gold 去校验是否一致。
    """
    return (
        "You are analyzing whether sentences belong to a stated genre.\n\n"

        "### Task\n"
        "Determine if two sentences both belong to the stated genre.\n"
        "For each sentence, identify 1-2 KEY linguistic/stylistic markers and pick "
        "the single genre that BEST describes it.\n\n"

        "### Genre Reference\n"
        f"{_GENRE_REFERENCE}\n"

        "### Genre Whitelist (STRICT)\n"
        "sentence1 genre and sentence2 genre MUST be EXACTLY one of the following 10 values:\n"
        "  face-to-face, government, letters, 9/11, slate, fiction, telephone, travel, oup, verbatim\n"
        "DO NOT invent new genre names such as 'fashion', 'history', 'personal', 'opinion',\n"
        "'narrative', 'conversation', 'academic', 'regulation', 'tourism', 'linguistics'.\n"
        "If a sentence seems ambiguous, pick the CLOSEST matching value from the 10 genres above.\n\n"

        "### Rules\n"
        f"- stated genre MUST be: {stated_genre}\n"
        "- For label Y: sentence1 genre AND sentence2 genre must both equal the stated genre\n"
        "- For label N: at least one of the two genres does NOT equal the stated genre\n\n"

        "### Text to Classify\n"
        f"Sentence 1: {s1}\n"
        f"Sentence 2: {s2}\n"
        f"Stated Genre: {stated_genre}\n\n"

        "### Output Format\n"
        "Output EXACTLY four lines, no explanation:\n"
        "stated genre: [one of the 10 genres]\n"
        "sentence1 genre: [one of the 10 genres]\n"
        "sentence2 genre: [one of the 10 genres]\n"
        "<label>N or Y</label>\n"
    )


# ============================================================
# Qwen3-4B 调用（thinking mode）
# ============================================================

def call_qwen3_4b(prompt: str, temperature: float = 0.0,
                  max_tokens: int = 4096,
                  repetition_penalty: float = 1.1) -> tuple:
    """返回 (content, reasoning_content)。失败时返回 (None, None)。"""
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
# 答案提取与校验
# ============================================================

_RE_STATED = re.compile(r'stated genre:\s*([\w/\-]+)', re.IGNORECASE)
_RE_S1 = re.compile(r'sentence1 genre:\s*([\w/\-]+)', re.IGNORECASE)
_RE_S2 = re.compile(r'sentence2 genre:\s*([\w/\-]+)', re.IGNORECASE)
_RE_LABEL = re.compile(r'<label>\s*([YN])\s*</label>', re.IGNORECASE)


def pick_think_text(content: str, reasoning: str) -> str:
    """优先使用 reasoning_content，其次 content 中 ``<think>...</think>`` 块。

    Task6 prompt 不透露 GT，reasoning 中不存在 GT 锚点污染，无需清洗。
    """
    if reasoning:
        return reasoning.strip()
    if not content:
        return ""
    m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def parse_four_lines(raw: str) -> tuple:
    """从模型输出解析 (stated_genre, s1_genre, s2_genre, label)。"""
    if not raw:
        return None, None, None, None
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    ms = _RE_STATED.search(clean)
    m1 = _RE_S1.search(clean)
    m2 = _RE_S2.search(clean)
    ml = _RE_LABEL.search(clean)
    return (
        ms.group(1).strip().lower() if ms else None,
        m1.group(1).strip().lower() if m1 else None,
        m2.group(1).strip().lower() if m2 else None,
        ml.group(1).upper() if ml else None,
    )


def verify_cot(stated_parsed: Optional[str], s1_genre: Optional[str],
               s2_genre: Optional[str], label_parsed: Optional[str],
               stated_gold: str, gold_label: str) -> tuple[bool, str]:
    """Y/N 统一校验：
      1. 四行齐全（stated / s1 / s2 / label）
      2. 三个 genre 都在 10 类白名单内
      3. stated genre 与 gold stated 一致
      4. <label> 与 gold_label 一致
      5. 语义一致性:
         - gold=Y: sentence1 genre == sentence2 genre == stated
         - gold=N: sentence1 genre / sentence2 genre 至少一个 != stated
    """
    if not all([stated_parsed, s1_genre, s2_genre, label_parsed]):
        return False, "四行格式缺失"
    for g, name in [(stated_parsed, 'stated'), (s1_genre, 'S1'), (s2_genre, 'S2')]:
        if g not in TASK6_GENRE_SET:
            return False, f"{name} genre '{g}' 不在白名单"
    if stated_parsed != stated_gold:
        return False, f"stated '{stated_parsed}' != gold '{stated_gold}'"
    if label_parsed != gold_label:
        return False, f"label '{label_parsed}' != gold '{gold_label}'"
    if gold_label == 'Y':
        if s1_genre != stated_gold or s2_genre != stated_gold:
            return False, (f"Y 语义违反: S1='{s1_genre}' S2='{s2_genre}' "
                           f"均应 == stated '{stated_gold}'")
    else:  # gold_label == 'N'
        if s1_genre == stated_gold and s2_genre == stated_gold:
            return False, "N 语义违反: 两 genre 均等于 stated"
    return True, ""


# ============================================================
# 单条样本处理
# ============================================================

_TEMPERATURE_LADDER = (0.0, 0.3, 0.6)


def format_cot_3lines(stated: str, s1_genre: str, s2_genre: str) -> str:
    """生成 3 行 CoT（不含 ``<label>``，由 Selector.select 自动补）。"""
    return (f"stated genre: {stated}\n"
            f"sentence1 genre: {s1_genre}\n"
            f"sentence2 genre: {s2_genre}")


def generate_for_example(ex: dict, max_retries: int = 3) -> dict:
    """返回 dict: ``{cot, think, attempts, verified, source}``。

    source: ``model`` / ``parse_fail`` / ``failed``。
    Y/N 都走模型推理，失败即空（``cot=''`` ``think=''``）。
    """
    gold = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
    gold = gold.strip().upper()
    input_text = ex['input']
    stated = extract_stated_genre(input_text)

    if not stated or stated not in TASK6_GENRE_SET:
        logger.warning(f"stated genre 解析失败: id={ex.get('id')} stated={stated!r}")
        return {'cot': '', 'think': '', 'attempts': 0,
                'verified': False, 'source': 'parse_fail'}

    if gold not in ('Y', 'N'):
        logger.warning(f"非法 gold label: id={ex.get('id')} gold={gold!r}")
        return {'cot': '', 'think': '', 'attempts': 0,
                'verified': False, 'source': 'parse_fail'}

    s1_text, s2_text = extract_sentences(input_text)
    prompt = build_open_prompt(s1_text, s2_text, stated)

    attempts = 0
    last_reason = ''
    for attempt in range(max_retries):
        attempts = attempt + 1
        temp = _TEMPERATURE_LADDER[min(attempt, len(_TEMPERATURE_LADDER) - 1)]
        content, reasoning = call_qwen3_4b(prompt, temperature=temp)
        if content is None:
            last_reason = 'API 调用失败'
            time.sleep(1)
            continue

        stated_p, s1_g, s2_g, label_p = parse_four_lines(content)
        ok, reason = verify_cot(stated_p, s1_g, s2_g, label_p, stated, gold)
        if not ok:
            last_reason = reason
            logger.debug(
                f"  [attempt {attempts}/{max_retries} T={temp}] 校验失败: {reason}"
            )
            continue

        cot = format_cot_3lines(stated, s1_g, s2_g)
        think = pick_think_text(content, reasoning)
        return {'cot': cot, 'think': think, 'attempts': attempts,
                'verified': True, 'source': 'model'}

    logger.info(
        f"  ❌ {max_retries} 次重试失败 id={ex.get('id','')[:20]} gold={gold} "
        f"stated={stated} last_reason='{last_reason}'"
    )
    return {'cot': '', 'think': '', 'attempts': attempts,
            'verified': False, 'source': 'failed'}


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Task 6 CoT + Think 离线生成')
    parser.add_argument('--limit', type=int, default=0,
                        help='仅处理前 N 条（0=全量）')
    parser.add_argument('--data', type=str, default='',
                        help='输入原始数据路径（默认使用 common.paths.task_data_file(6)）')
    parser.add_argument('--output', type=str, default='',
                        help='输出带 cot+think 的数据路径（默认 task_cot_dir(6)/openseek-6_*.json）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有输出续跑（跳过已有非空 cot 的示例）')
    parser.add_argument('--concurrency', type=int, default=3,
                        help='并发请求数（默认 3）')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='每完成 N 条保存一次')
    args = parser.parse_args()

    data_path = args.data or task_data_file(6)
    out_path = args.output or os.path.join(
        task_cot_dir(6), 'openseek-6_mnli_with_cot.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    log_dir = task_log_dir(6)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'generate_task6_cot.log')
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

    label_dist = Counter(
        (ex['output'][0] if isinstance(ex['output'], list) else ex['output']).strip().upper()
        for ex in examples
    )
    logger.info(f"加载 {total} 条示例 | 分布 {dict(label_dist)}")

    existing = {}
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            old = json.load(f)
        for ex in old.get('examples', []):
            if ex.get('cot'):
                existing[ex['id']] = ex
        logger.info(f"续跑: 已有 {len(existing)} 条 cot")

    processed = success = failed = skipped = 0
    verified_count = 0
    source_counter = Counter()
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

    def log_result(ex, result, dt, idx):
        nonlocal success, failed, verified_count
        gold = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
        if result['cot']:
            success += 1
            if result.get('verified'):
                verified_count += 1
            src = result.get('source', '?')
            mark = '✅' if result.get('verified') else '⚠️'
            logger.info(f"[{idx}/{to_process}] id={ex['id'][:20]} "
                        f"(GT={gold}, {src}) {mark} "
                        f"attempts={result.get('attempts', 0)}, {dt:.1f}s")
        else:
            failed += 1
            logger.warning(f"[{idx}/{to_process}] id={ex['id'][:20]} "
                           f"(GT={gold}) ❌ 降级 ({dt:.1f}s)")

    if args.concurrency <= 1:
        for ex in pending:
            processed += 1
            _, result, dt = process_one(ex)
            ex['cot'] = result['cot']
            ex['think'] = result['think']
            source_counter[result.get('source', '?')] += 1
            log_result(ex, result, dt, processed)
            if processed % args.save_interval == 0:
                save_progress()
                elapsed = time.time() - t_start
                avg = elapsed / processed
                logger.info(f"  💾 已保存 (成功={success} 验证通过={verified_count} "
                            f"降级={failed}) | {avg:.2f}s/条均值 | "
                            f"ETA={avg*(to_process-processed)/60:.1f}min")
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
                source_counter[result.get('source', '?')] += 1
                processed += 1
                log_result(ex, result, dt, processed)
                if processed % args.save_interval == 0:
                    save_progress()
                    elapsed = time.time() - t_start
                    avg = elapsed / processed
                    remaining = to_process - processed
                    logger.info(f"  💾 已保存 (成功={success} 验证通过={verified_count} "
                                f"降级={failed}) | {avg:.2f}s/条均值 | "
                                f"ETA={avg*remaining/60:.1f}min")

    out = dict(data)
    out['examples'] = examples
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"完成! {elapsed:.0f}s, 共处理 {processed} 条")
    logger.info(f"  cot 验证通过: {verified_count}")
    logger.info(f"  cot 非空总数: {success}")
    logger.info(f"  cot 降级（空）: {failed}")
    logger.info(f"  跳过（续跑）: {skipped}")
    logger.info(f"  来源分布: {dict(source_counter)}")
    if processed > 0:
        logger.info(f"  验证通过率: {verified_count/processed*100:.1f}%")
    logger.info(f"  输出: {out_path}")


if __name__ == '__main__':
    main()
