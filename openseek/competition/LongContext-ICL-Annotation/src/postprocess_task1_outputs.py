#!/usr/bin/env python3
"""Task1 (closest_integers) 预测后处理：清洗 label 残留并抽取单个整数。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from method_hyb import count_answer

_LABEL_PAIR_RE = re.compile(r"<label>\s*(-?\d+)\s*</label>", re.IGNORECASE | re.DOTALL)
_LABEL_OPEN_RE = re.compile(r"<label>\s*(-?\d+)", re.IGNORECASE)
_LABEL_CLOSE_RE = re.compile(r"(-?\d+)\s*</label>", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def normalize_task1_prediction(raw: Any) -> str:
    """
    将模型原始输出规范为单个整数字符串。
    覆盖：``5</label>``、未闭合 ``<label>2``、``count_answer`` 已解析结果等。
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

    for pattern in (_LABEL_PAIR_RE, _LABEL_OPEN_RE, _LABEL_CLOSE_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    if re.fullmatch(r"-?\d+", text):
        return text

    matches = _INT_RE.findall(text)
    if matches:
        return matches[-1]
    return ""


def _normalize_compare_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


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
    """
    后处理 JSONL（compare 或 outputs 格式），重写预测字段并重算 is_match（若有金标）。
    ``inplace=True`` 时覆盖原文件。
    """
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
                new = normalize_task1_prediction(old)
                if new != old:
                    row["model_output"] = new
                    changed += 1
                if "expected_output" in row:
                    row["is_match"] = _normalize_compare_text(new) == _normalize_compare_text(
                        row["expected_output"]
                    )
            elif "prediction" in row:
                old = str(row.get("prediction", ""))
                new = normalize_task1_prediction(old)
                if new != old:
                    row["prediction"] = new
                    changed += 1
                expected = row.get("expected_output")
                if expected is not None:
                    row["is_match"] = _normalize_compare_text(new) == _normalize_compare_text(
                        expected
                    )
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
    parser = argparse.ArgumentParser(description="Task1 预测后处理（closest_integers）")
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
    print(f"[task1 后处理] input={input_path}")
    print(f"[task1 后处理] output={stats['file']}")
    print(
        f"[task1 后处理] total={stats['total']} changed={stats['changed']} "
        f"matched={stats['matched']} accuracy={stats['accuracy']:.4%}"
    )


if __name__ == "__main__":
    main()
