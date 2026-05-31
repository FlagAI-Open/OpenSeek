#!/usr/bin/env python3
"""Task3 (collatz_conjecture) 预测后处理：抽取变换后的列表，处理 input+output 复述格式。"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from method_hyb import count_answer

_LIST_LITERAL_RE = re.compile(r"\[[^\[\]]*\]")


def _canonical_list_text(text: str) -> str:
    """将列表字面量规范为 ``str(list)`` 形式，便于字符串比对。"""
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return " ".join(stripped.split())
    if isinstance(value, list):
        return str(value)
    return " ".join(stripped.split())


def normalize_task3_prediction(raw: Any) -> str:
    """
    将模型输出规范为单个 Python 列表字面量字符串。
    覆盖：``[in] [out]`` 复述、``<label>`` 包裹、仅列表无标签等。
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""

    parsed = count_answer(text)
    if parsed is not None and str(parsed).strip():
        text = str(parsed).strip()

    text = re.sub(r"</?label>", "", text, flags=re.IGNORECASE).strip()

    lists = _LIST_LITERAL_RE.findall(text)
    if len(lists) >= 2:
        return _canonical_list_text(lists[-1])
    if len(lists) == 1:
        return _canonical_list_text(lists[0])
    return _canonical_list_text(text)


def _predictions_equal(pred: str, expected: str) -> bool:
    pred_c = _canonical_list_text(pred)
    exp_c = _canonical_list_text(expected)
    if pred_c == exp_c:
        return True
    try:
        return ast.literal_eval(pred_c) == ast.literal_eval(exp_c)
    except (ValueError, SyntaxError):
        return False


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}-postprocessed{input_path.suffix}")


def postprocess_jsonl(
    input_path: Path,
    output_path: Path | None = None,
    *,
    inplace: bool = False,
) -> dict[str, Any]:
    """后处理 JSONL，重写预测并重算 is_match（若有 expected_output）。"""
    input_path = input_path.resolve()
    if inplace:
        target = input_path.with_suffix(input_path.suffix + ".tmp")
    else:
        target = _resolve_output_path(input_path, output_path.resolve() if output_path else None)

    total = 0
    changed = 0

    target.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as rf, target.open(
        "w", encoding="utf-8", newline="\n"
    ) as wf:
        for line_no, raw_line in enumerate(rf, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc

            if "model_output" in row:
                old = str(row.get("model_output", ""))
                new = normalize_task3_prediction(old)
                if new != old:
                    row["model_output"] = new
                    changed += 1
                if "expected_output" in row:
                    row["is_match"] = _predictions_equal(new, str(row["expected_output"]))
            elif "prediction" in row:
                old = str(row.get("prediction", ""))
                new = normalize_task3_prediction(old)
                if new != old:
                    row["prediction"] = new
                    changed += 1
                expected = row.get("expected_output")
                if expected is not None:
                    row["is_match"] = _predictions_equal(new, str(expected))
            else:
                raise ValueError(
                    f"第 {line_no} 行缺少 model_output / prediction 字段，无法后处理"
                )

            total += 1
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    if inplace:
        target.replace(input_path)
        final_path = input_path
    else:
        final_path = target

    matched = 0
    with final_path.open("r", encoding="utf-8") as rf:
        for raw in rf:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if bool(row.get("is_match", False)):
                matched += 1

    accuracy = (matched / total) if total else 0.0
    return {
        "total": total,
        "changed": changed,
        "matched": matched,
        "accuracy": accuracy,
        "file": str(final_path),
    }


def postprocess_jsonl_inplace(input_path: Path) -> dict[str, Any]:
    return postprocess_jsonl(input_path, inplace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task3 预测后处理（collatz_conjecture）")
    parser.add_argument("--input", type=Path, required=True, help="输入 JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL；默认在文件名后追加 -postprocessed",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="原地覆盖输入文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    stats = postprocess_jsonl(
        input_path,
        args.output.resolve() if args.output else None,
        inplace=args.inplace,
    )
    print(f"[task3 后处理] input={input_path}")
    print(f"[task3 后处理] output={stats['file']}")
    print(
        f"[task3 后处理] total={stats['total']} changed={stats['changed']} "
        f"matched={stats['matched']} accuracy={stats['accuracy']:.4%}"
    )


if __name__ == "__main__":
    main()
