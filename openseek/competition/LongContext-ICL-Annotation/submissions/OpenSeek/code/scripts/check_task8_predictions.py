#!/usr/bin/env python3
"""Static quality checks for task 8 code-generation predictions."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(
    r"\bpass\b|TODO|Your code here|Implement .* here|# Implement|\[Final Source Code\]",
    re.IGNORECASE,
)
CODE_MARKERS = ("def ", "import ", "from ", "return ", "torch.", "@triton.jit", "triton")
PROSE_PREFIXES = (
    "the function",
    "the provided",
    "this function",
    "this code",
    "the task",
    "in this",
    "now,",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ast_ok(prediction: str) -> bool:
    try:
        ast.parse(prediction)
    except SyntaxError:
        return False
    return True


def summarize(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    details = []
    for row in rows:
        prediction = str(row.get("prediction", ""))
        stripped = prediction.strip()
        detail = {
            "test_sample_id": row.get("test_sample_id", ""),
            "length": len(prediction),
            "empty": not stripped,
            "code_marker": any(marker in prediction for marker in CODE_MARKERS),
            "placeholder": bool(PLACEHOLDER_RE.search(prediction)),
            "prose_prefix": stripped.lower().startswith(PROSE_PREFIXES),
            "ast_ok": ast_ok(prediction),
        }
        detail["static_ok"] = (
            not detail["empty"]
            and detail["code_marker"]
            and not detail["placeholder"]
            and not detail["prose_prefix"]
            and detail["ast_ok"]
        )
        details.append(detail)

    def count(key: str) -> int:
        return sum(1 for item in details if item[key])

    return {
        "file": str(path),
        "rows": len(rows),
        "empty": count("empty"),
        "code_marker": count("code_marker"),
        "placeholder": count("placeholder"),
        "prose_prefix": count("prose_prefix"),
        "ast_ok": count("ast_ok"),
        "static_ok": count("static_ok"),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prediction_file",
        nargs="?",
        default="outputs/final_submission_candidate/predictions/openseek-8-v1.jsonl",
        help="Path to openseek-8-v1.jsonl",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    summary = summarize(Path(args.prediction_file))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"file: {summary['file']}")
    print(f"rows: {summary['rows']}")
    print(f"empty: {summary['empty']}")
    print(f"code_marker: {summary['code_marker']}")
    print(f"placeholder: {summary['placeholder']}")
    print(f"prose_prefix: {summary['prose_prefix']}")
    print(f"ast_ok: {summary['ast_ok']}")
    print(f"static_ok: {summary['static_ok']}")


if __name__ == "__main__":
    main()
