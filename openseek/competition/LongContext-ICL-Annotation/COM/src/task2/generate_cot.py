#!/usr/bin/env python3
"""generate_cot.py — Task 2 离线 CoT + Think 生成

设计要点：
- **Prompt 不暴露 GT**：只给句子和规则，避免 think 出现
  ``"the user said the answer is X"`` 锚点句污染。has/have/had 区分按"词性"
  判断（后接名词短语 → 实义动词 / 后接过去分词 → 助动词）。
- **答案校验（金标准）**：
  * ``<label>N</label>`` 的 N == GT
  * ``Nouns:/Verbs:`` 行列出的词数 == GT
  * verb 情况：列出的词不能落入助动词黑名单
- **3 次重试 + 温度阶梯**：0.0 → 0.3 → 0.6
- **scrub_think**：清洗残留锚点句
- **validate_think**：长度 40–8000 + 命中任务关键词任一

输出格式（与其它 task cot_data 对齐）::

    {
        "id": "...",
        "input": "Sentence: '...'. Count the number of nouns/verbs ...",
        "output": ["3"],
        "cot":   "Nouns: train, train, tracks",
        "think": "<Qwen3-4B reasoning content（已清洗）>"
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
from method import AUX_VERBS  # noqa: E402
from common.paths import (  # noqa: E402
    VLLM_BASE_URL as DEFAULT_BASE_URL,
    VLLM_MODEL_ID as DEFAULT_MODEL,
    task_cot_dir,
    task_data_file,
    task_log_dir,
)

_client = OpenAI(api_key="EMPTY", base_url=DEFAULT_BASE_URL)


# ============================================================
# 输入解析辅助
# ============================================================

def extract_sentence(input_text: str) -> str:
    """从 Task 2 输入中提取纯句子（两次 split，避免 girl's 等撇号截断）。"""
    for prefix, close in [("Sentence: '", "'. Count"), ("sentence: '", "'. Count"),
                          ('Sentence: "', '". Count'), ('sentence: "', '". Count')]:
        if prefix in input_text:
            return input_text.split(prefix, 1)[1].split(close, 1)[0]
    return input_text


def extract_target(input_text: str) -> str:
    """判断任务目标是 noun 还是 verb"""
    return 'noun' if 'noun' in input_text.lower() else 'verb'


# ============================================================
# Prompt 构造（不暴露 GT）
# ============================================================

_VERB_RULES = (
    "### Rules for counting verbs\n"
    "Count ONLY action/lexical verbs (e.g. run, eat, play, walk, catch).\n"
    "EXCLUDE these auxiliary/modal verbs:\n"
    "  is, are, was, were, am, be, been, being,\n"
    "  do, does, did,\n"
    "  will, would, shall, should,\n"
    "  can, could, may, might, must.\n"
    "\n"
    "Special case for has / have / had (decide by what follows them):\n"
    "  - COUNT as a VERB when followed by a NOUN or noun phrase\n"
    "    (main verb, meaning \"to possess/own\"):\n"
    "      \"I have a cat\"         -> have = verb (counted)\n"
    "      \"She has blue eyes\"    -> has  = verb (counted)\n"
    "      \"They had three dogs\"  -> had  = verb (counted)\n"
    "  - EXCLUDE when followed by a PAST PARTICIPLE\n"
    "    (auxiliary verb forming perfect tenses):\n"
    "      \"has eaten\"             -> has  = auxiliary (excluded)\n"
    "      \"have been working\"     -> have = auxiliary (excluded)\n"
    "      \"had finished the task\" -> had  = auxiliary (excluded)\n"
    "\n"
    "Additional inclusion rules (IMPORTANT, do NOT miss these):\n"
    "  - -ing forms (present participles / gerunds) ARE counted as verbs,\n"
    "    EVEN WHEN they modify a noun (adjectival use) or act as a noun:\n"
    "      \"boiling water\"             -> boiling  = verb (counted)\n"
    "      \"tasty looking food\"        -> looking  = verb (counted)\n"
    "      \"for loading passengers\"    -> loading  = verb (counted)\n"
    "      \"I enjoy swimming\"          -> swimming = verb (counted)\n"
    "      \"a running man\"             -> running  = verb (counted)\n"
    "  - Past participles that act as the MAIN verb in passive voice or\n"
    "    perfect tense ARE counted (the auxiliary is/are/was/were/being/been\n"
    "    is still EXCLUDED as listed above, but the participle itself IS):\n"
    "      \"is lowered\"        -> lowered = verb (counted), is    = excluded\n"
    "      \"are being canned\"  -> canned  = verb (counted), are/being = excluded\n"
    "      \"has been opened\"   -> opened  = verb (counted), has/been  = excluded\n"
    "      \"was broken\"        -> broken  = verb (counted), was   = excluded\n"
)

_NOUN_RULES = (
    "### Rules for counting nouns\n"
    "Count every noun in the sentence, including proper nouns.\n"
    "If the SAME noun appears multiple times, count each occurrence separately\n"
    "(e.g. \"a train on train tracks\" -> train, train, tracks -> 3).\n"
)


def build_open_prompt(input_text: str) -> str:
    """构造不暴露 GT 的 prompt：只给句子和规则，让模型自己数词 + 写 <label>N</label>"""
    sentence = extract_sentence(input_text)
    target = extract_target(input_text)
    target_cap = target.capitalize() + 's'
    rules = _VERB_RULES if target == 'verb' else _NOUN_RULES

    verb_skip_example = (
        "  Verbs: run, eat [skip: is, can]\n"
        "  <label>2</label>\n"
        "  Verbs: none [skip: is, are]\n"
        "  <label>0</label>\n"
    )
    noun_example = (
        "  Nouns: cat, dog, house\n"
        "  <label>3</label>\n"
        "  Nouns: none\n"
        "  <label>0</label>\n"
    )
    example_block = verb_skip_example if target == 'verb' else noun_example

    return (
        "### Task\n"
        f"Count the number of {target}s in the given sentence. Think step by step.\n\n"
        f"{rules}\n"
        "### Output format (MUST end with the label tag)\n"
        f"First, list the found words on one line as `{target_cap}: w1, w2, w3`.\n"
        + ("For verbs, append `[skip: ...]` if auxiliary/modal verbs appear in the sentence.\n"
           if target == 'verb' else "")
        + f"If no {target}s are found, write `{target_cap}: none`"
        + (" (still append `[skip: ...]` if auxiliaries are present)" if target == 'verb' else "")
        + ".\n"
        "Then output the count on a new line inside <label>...</label>.\n\n"
        "Example formats:\n"
        f"{example_block}\n"
        "### Sentence\n"
        f"'{sentence}'\n"
    )


# ============================================================
# Qwen3-4B 调用（thinking mode）
# ============================================================

def call_qwen3_4b(prompt: str, temperature: float = 0.0,
                  max_tokens: int = 4096,
                  repetition_penalty: float = 1.1) -> tuple:
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
# 答案提取与校验
# ============================================================

_LABEL_RE = re.compile(r'<label>\s*(\d+)\s*</label>', re.DOTALL)


def extract_label_count(content: str) -> Optional[int]:
    """从 content 中提取 <label>N</label> 的整数。"""
    if not content:
        return None
    m = _LABEL_RE.search(content)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def extract_cot_line(content: str, target: str) -> Optional[str]:
    """从 content 中提取 ``Nouns: ...`` 或 ``Verbs: ...`` 行。

    vLLM thinking mode 下 content 可能混合 ``<think>...</think>``，先剥掉再解析。
    """
    if not content:
        return None
    clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    target_cap = target.capitalize() + 's'
    pattern = rf'^{target_cap}:\s*.+$'
    for line in clean.split('\n'):
        line = line.strip()
        if re.match(pattern, line, re.IGNORECASE):
            return line
    fb = r'^[Nn]ouns:\s*.+$' if target == 'noun' else r'^[Vv]erbs:\s*.+$'
    for line in clean.split('\n'):
        line = line.strip()
        if re.match(fb, line):
            return line
    return None


def parse_words_from_cot(cot_line: str) -> list:
    """从 ``Nouns: a, b, c`` 或 ``Verbs: a, b [skip: is]`` 中提取主词列表。"""
    if not cot_line:
        return []
    main = cot_line.split('[skip:')[0].strip()
    main = re.sub(r'^[NnVv]\w+:\s*', '', main).strip()
    if not main or main.lower() == 'none':
        return []
    return [w.strip() for w in main.split(',') if w.strip()]


def verify_answer(cot_line: Optional[str], label_count: Optional[int],
                  gt_count: int, target: str) -> tuple[bool, str]:
    """三重校验：
      1. label_count == gt_count
      2. 列出的词数 == gt_count
      3. verb 情况：列出的词不能在助动词黑名单（has/have/had 允许）
    返回 (是否通过, 失败原因 for log)
    """
    if label_count is None:
        return False, "label 缺失"
    if label_count != gt_count:
        return False, f"label={label_count} != GT={gt_count}"

    words = parse_words_from_cot(cot_line or "")
    if gt_count == 0:
        if len(words) != 0:
            return False, f"GT=0 但列出 {len(words)} 个词"
        return True, ""

    if len(words) != gt_count:
        return False, f"列词数={len(words)} != GT={gt_count}"

    if target == 'verb':
        bad = [w for w in words if w.lower() in AUX_VERBS]
        if bad:
            return False, f"动词列表含助动词 {bad}"

    return True, ""


# ============================================================
# think 字段提取、清洗、验证
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


_SCRUB_PATTERNS = [
    re.compile(r'^.*\bthe user\b.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^.*\b(expected|desired)\s+(count|number|answer|output)\b.*$',
               re.IGNORECASE | re.MULTILINE),
    re.compile(r'^.*\b(they want|user said|user provided|user mentioned|user told)\b.*$',
               re.IGNORECASE | re.MULTILINE),
    re.compile(r'^.*\bthe (correct )?answer is\s+\d+.*$',
               re.IGNORECASE | re.MULTILINE),
]


def scrub_think(think: str) -> str:
    """清洗 think 中残留的 GT 锚点句（理论上不应出现，做兜底）。"""
    if not think:
        return think
    cleaned = think
    for pat in _SCRUB_PATTERNS:
        cleaned = pat.sub('', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


# Task 2 语义关键词（命中任一即视为"在讨论数词这件事"）
_TASK2_THINK_KEYWORDS = (
    'noun', 'verb',
    'count', 'number of', 'how many',
    'auxiliary', 'modal', 'lexical', 'action',
    'possess', 'own', 'have',
    'participle', 'perfect',
    'subject', 'object', 'noun phrase',
    'plural', 'singular',
)


def validate_think(think: str) -> bool:
    """宽松验证 think 字段可用性：
      - 非空
      - 40 <= 长度 <= 8000
      - 命中 _TASK2_THINK_KEYWORDS 中任一关键词
    """
    if not think:
        return False
    length = len(think)
    if length < 40 or length > 8000:
        return False
    lower = think.lower()
    return any(kw in lower for kw in _TASK2_THINK_KEYWORDS)


# ============================================================
# 单条样本处理
# ============================================================

_TEMPERATURE_LADDER = (0.0, 0.3, 0.6)


def generate_for_example(ex: dict, max_retries: int = 3) -> dict:
    """不透露 GT，答案校验（label + 词数 + 助动词黑名单）通过才采纳 think。"""
    gt_raw = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
    try:
        gt_count = int(gt_raw)
    except (ValueError, TypeError):
        logger.warning(f"GT 无法解析为整数: id={ex.get('id')} gt={gt_raw}")
        return {'cot': '', 'think': '', 'attempts': 0,
                'verified': False, 'last_pred': None}

    input_text = ex['input']
    target = extract_target(input_text)

    prompt = build_open_prompt(input_text)

    cot_final = ''
    think = ''
    verified = False
    attempts = 0
    last_pred = None

    for attempt in range(max_retries):
        attempts = attempt + 1
        temp = _TEMPERATURE_LADDER[min(attempt, len(_TEMPERATURE_LADDER) - 1)]
        content, reasoning = call_qwen3_4b(prompt, temperature=temp)
        if content is None and reasoning is None:
            time.sleep(1)
            continue

        cot_line = extract_cot_line(content or "", target)
        label_count = extract_label_count(content or "")
        last_pred = label_count

        ok, reason = verify_answer(cot_line, label_count, gt_count, target)
        if not ok:
            logger.warning(
                f"  [attempt {attempt+1}/{max_retries} T={temp}] 校验失败: {reason} "
                f"cot_line={repr(cot_line)[:80] if cot_line else None}"
            )
            continue

        candidate = pick_think_text(content or "", reasoning or "")
        candidate = scrub_think(candidate)
        if not validate_think(candidate):
            logger.warning(
                f"  [attempt {attempt+1}/{max_retries} T={temp}] think 验证失败 "
                f"(len={len(candidate)}): '{candidate[:80]}'"
            )
            continue

        cot_final = cot_line
        think = candidate
        verified = True
        break

    return {'cot': cot_final, 'think': think, 'attempts': attempts,
            'verified': verified, 'last_pred': last_pred}


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Task 2 CoT + Think 离线生成')
    parser.add_argument('--limit', type=int, default=0,
                        help='仅处理前 N 条（0=全量）')
    parser.add_argument('--data', type=str, default='',
                        help='输入原始数据路径（默认使用 common.paths.task_data_file(2)）')
    parser.add_argument('--output', type=str, default='',
                        help='输出带 cot+think 的数据路径（默认 task_cot_dir(2)/openseek-2_*.json）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有输出续跑（跳过已有非空 think 的示例）')
    parser.add_argument('--concurrency', type=int, default=3,
                        help='并发请求数（默认 3）')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='每完成 N 条保存一次')
    args = parser.parse_args()

    data_path = args.data or task_data_file(2)
    out_path = args.output or os.path.join(
        task_cot_dir(2), 'openseek-2_count_nouns_verbs_with_cot.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    log_dir = task_log_dir(2)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'generate_task2_cot.log')
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
    verified_count = 0
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

    if args.concurrency <= 1:
        for ex in pending:
            gt = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            target = extract_target(ex['input'])
            sentence = extract_sentence(ex['input'])
            logger.info(f"[{processed+1}/{to_process}] id={ex['id']} "
                        f"({target}, GT={gt}) {sentence[:70]}")
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
                target = extract_target(ex['input'])
                if result['think']:
                    success += 1
                    if result.get('verified'):
                        verified_count += 1
                    logger.info(f"[{processed+1}/{to_process}] id={ex['id'][:20]} "
                                f"({target}, GT={gt}) ✅ ({len(result['think'])}ch, "
                                f"attempts={result.get('attempts')}, {dt:.1f}s)")
                else:
                    failed += 1
                    logger.warning(f"[{processed+1}/{to_process}] id={ex['id'][:20]} "
                                   f"({target}, GT={gt}) ❌ 降级 last_pred="
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
