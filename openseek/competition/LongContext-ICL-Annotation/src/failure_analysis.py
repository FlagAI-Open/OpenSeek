#!/usr/bin/env python3
"""从 compare / autoresearch 结果 JSONL 提取失败模式，供 autoresearch_prompt 使用。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FailureExample:
    input: str
    ground_truth: str
    prediction: str
    example_id: str = ""
    thinking: str = ""
    error: str = ""


@dataclass
class FailureCategory:
    name: str
    count: int
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FailureReport:
    accuracy: float
    none_rate: float
    n_correct: int
    n_total: int
    n_none: int
    categories: list[FailureCategory] = field(default_factory=list)


def _row_input(row: dict[str, Any]) -> str:
    return str(row.get("input") or row.get("text2annotate") or "")[:500]


def _row_gold(row: dict[str, Any]) -> str:
    eo = row.get("expected_output") or row.get("ground_truth")
    if isinstance(eo, list):
        return str(eo[0]) if eo else ""
    return str(eo or "").strip()


def _row_pred(row: dict[str, Any]) -> str | None:
    if "prediction" in row:
        p = row.get("prediction")
    else:
        p = row.get("model_output")
    if p is None:
        return None
    s = str(p).strip()
    return s if s else None


def _row_correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    if "is_match" in row:
        return bool(row["is_match"])
    pred = _row_pred(row)
    gold = _row_gold(row)
    if pred is None:
        return False
    return pred.strip().lower() == gold.strip().lower()


def _categorize_error(inp: str, gold: str, pred: str | None) -> str:
    if pred is None or not str(pred).strip():
        return "empty_or_unparseable"
    p = str(pred).strip().lower()
    g = str(gold).strip().lower()
    if p == g:
        return "correct"
    if g in p or p in g:
        return "partial_overlap"
    if len(p.split()) > len(g.split()) * 2 + 3:
        return "too_verbose"
    if len(p.split()) < max(1, len(g.split()) // 2):
        return "too_short"
    if re.search(r"<label>|</label>", p, re.I):
        return "label_leak"
    if re.search(r"^(what|who|where|the )", p) and not re.search(r"^(what|who|where|the )", g):
        return "question_form"
    return "wrong_entity"


def analyze_results(path: str | Path) -> FailureReport:
    """解析 JSONL，汇总准确率、空输出率与错误类别。"""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n_total = len(rows)
    n_correct = 0
    n_none = 0
    buckets: dict[str, list[FailureExample]] = defaultdict(list)

    for row in rows:
        pred = _row_pred(row)
        gold = _row_gold(row)
        inp = _row_input(row)
        ok = _row_correct(row)
        if ok:
            n_correct += 1
            continue
        if pred is None or not str(pred).strip():
            n_none += 1
        cat = _categorize_error(inp, gold, pred)
        buckets[cat].append(
            FailureExample(
                input=inp,
                ground_truth=gold,
                prediction=str(pred or ""),
                example_id=str(row.get("example_id") or row.get("id") or ""),
                thinking=str(row.get("thinking") or "")[:300],
                error=str(row.get("error") or ""),
            )
        )

    accuracy = n_correct / n_total if n_total else 0.0
    none_rate = n_none / n_total if n_total else 0.0

    categories: list[FailureCategory] = []
    for name, items in sorted(buckets.items(), key=lambda x: -len(x[1])):
        categories.append(
            FailureCategory(
                name=name,
                count=len(items),
                examples=[
                    {
                        "input": ex.input,
                        "ground_truth": ex.ground_truth,
                        "prediction": ex.prediction,
                        "example_id": ex.example_id,
                    }
                    for ex in items[:5]
                ],
            )
        )

    return FailureReport(
        accuracy=accuracy,
        none_rate=none_rate,
        n_correct=n_correct,
        n_total=n_total,
        n_none=n_none,
        categories=categories,
    )


def format_report(report: FailureReport) -> str:
    lines = [
        f"Accuracy: {report.accuracy:.4f} ({report.n_correct}/{report.n_total})",
        f"None/unparseable: {report.none_rate:.4f} ({report.n_none})",
        "",
        "Error categories:",
    ]
    for cat in report.categories:
        lines.append(f"  - {cat.name}: {cat.count}")
        for ex in cat.examples[:2]:
            lines.append(
                f"      GT=`{ex['ground_truth'][:60]}` pred=`{ex['prediction'][:60]}`"
            )
    return "\n".join(lines)
