"""
task2 后处理：若题干里引用的英文句子含 ASCII 大写字母（A–Z），则用 `.lower()` 后的句子
重新走一遍与 ``infer_examples_main1_task2_spacy_v3`` 相同的 prompt（零样本 + spaCy v3），
覆盖 ``model_output`` / ``is_match``。

用法（在仓库根目录执行）::

    python src/postprocess_main1_task2_uppercase_rerun_spacy_v3.py \\
        --input_jsonl examples_main1/openseek-2-examples-main1-compare_spacy_v3_zeroshot.jsonl \\
        --output_jsonl examples_main1/openseek-2-examples-main1-compare_spacy_v3_zeroshot_lowercase_rerun.jsonl

依赖：与 ``infer_examples_main1_task2_spacy_v3.py`` 相同（``method_hyb.annotate_nvidia``、DashScope 等）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from method_hyb import annotate_nvidia as annotate

from infer_examples_main1_task2_spacy_v3 import (
    TASK_DATA_FILES,
    REPO_ROOT,
    build_prompt,
    register_task_prompt,
    _extract_output,
    _normalize_text,
    _prompt_openseek_2_spacy_v3,
)

TASK_ID = 2

_RE_SENT_SINGLE = re.compile(r"(Sentence:\s*')(.*)('\.\s*Count\b)", flags=re.DOTALL)
_RE_SENT_DOUBLE = re.compile(r'(Sentence:\s*")(.*)("\.\s*Count\b)', flags=re.DOTALL)


def _contains_ascii_uppercase(s: str) -> bool:
    return any("A" <= c <= "Z" for c in s)


def build_input_with_lowercase_sentence(full_input: str) -> tuple[str, bool]:
    """
    若引用句中含大写，则仅将该句改为 `.lower()`，其余题干不变。
    返回 (new_input, changed)。
    """
    text = str(full_input).strip()
    for pat in (_RE_SENT_SINGLE, _RE_SENT_DOUBLE):
        m = pat.search(text)
        if not m:
            continue
        body = m.group(2)
        if not _contains_ascii_uppercase(body):
            return text, False
        lowered = body.lower()
        new_text = text[: m.start(2)] + lowered + text[m.end(2) :]
        return new_text, True
    return text, False


def _prediction_from_annotate(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return "" if not raw else str(raw[0]).strip()
    return str(raw).strip()


def _load_task_description(task_id: int) -> str:
    name = TASK_DATA_FILES.get(task_id)
    if not name:
        raise ValueError(f"unknown task_id={task_id}")
    path = REPO_ROOT / "data" / name
    with path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)
    defs = task_dict.get("Definition")
    if not defs:
        raise ValueError(f"no Definition in {path}")
    return str(defs[0]).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="task2：大写句转小写后用 spacy v3 提示词重新预测（jsonl 后处理）。")
    root = REPO_ROOT
    p.add_argument(
        "--input_jsonl",
        type=str,
        default=str(root / "examples_main1" / "openseek-2-examples-main1-compare_spacy_v3_zeroshot.jsonl"),
        help="输入 compare jsonl（含 example_id / input / expected_output / model_output）",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default=str(
            root / "examples_main1" / "openseek-2-examples-main1-compare_spacy_v3_zeroshot_lowercase_rerun.jsonl"
        ),
        help="输出 jsonl",
    )
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_wait_seconds", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（<=0 表示全部）")
    p.add_argument("--dry_run", action="store_true", help="只统计将重跑的条数，不写文件、不调 API")
    return p.parse_args()


def main() -> None:
    register_task_prompt(TASK_ID, _prompt_openseek_2_spacy_v3)

    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_description = _load_task_description(TASK_ID)

    rows_in: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_in.append(json.loads(line))

    if args.limit > 0:
        rows_in = rows_in[: args.limit]

    would_rerun = 0
    parse_miss = 0
    for row in rows_in:
        inp = str(row.get("input", ""))
        _, ch = build_input_with_lowercase_sentence(inp)
        if ch:
            would_rerun += 1
        elif inp and "Sentence:" in inp and "Count" in inp:
            m_ok = _RE_SENT_SINGLE.search(inp) or _RE_SENT_DOUBLE.search(inp)
            if not m_ok:
                parse_miss += 1

    print(
        f"[统计] 读取 {len(rows_in)} 条；含大写句将重跑 {would_rerun} 条；"
        f"疑似 task2 但未能解析 Sentence 外壳 {parse_miss} 条"
    )

    if args.dry_run:
        return

    out_lines = 0
    with output_path.open("w", encoding="utf-8") as wf:
        for row in rows_in:
            example_id = str(row.get("example_id", "")).strip()
            inp = str(row.get("input", ""))
            expected = _extract_output(row.get("expected_output", ""))

            prompt_input, lowered = build_input_with_lowercase_sentence(inp)
            prediction = _extract_output(row.get("model_output"))
            is_match = _normalize_text(prediction) == _normalize_text(expected)
            extra: dict[str, Any] = {}

            if lowered:
                prediction = ""
                for attempt in range(1, args.retries + 1):
                    try:
                        prompt = build_prompt(task_description, prompt_input, task_id=TASK_ID)
                        raw = annotate(prompt)
                        prediction = _prediction_from_annotate(raw)
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt >= args.retries:
                            print(f"[推理失败] id={example_id} attempt={attempt}/{args.retries} err={e}")
                        else:
                            print(f"[重试] id={example_id} attempt={attempt}/{args.retries} err={e}")
                            time.sleep(args.retry_wait_seconds)

                is_match = _normalize_text(prediction) == _normalize_text(expected)
                extra["lowercase_sentence_rerun"] = True

            rec = {
                "example_id": example_id,
                "input": inp,
                "expected_output": expected,
                "model_output": prediction,
                "is_match": is_match,
                **extra,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_lines += 1
            wf.flush()

    matched = 0
    total = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if bool(json.loads(line).get("is_match", False)):
                matched += 1

    acc = (matched / total) if total else 0.0
    print(f"[完成] 写出 {out_lines} 条 -> {output_path}")
    print(f"[指标] total={total} matched={matched} accuracy={acc:.2%}")


if __name__ == "__main__":
    main()
