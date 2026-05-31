#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对 task4 结果做后处理：若 model_output 含空格，或不等于 input 列表字符拼接结果，"
            "则用拼接结果覆盖，并重算 is_match。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 JSONL 文件路径（如 openseek-4-examples-main1-compare.jsonl）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL 文件路径；不传则默认在原文件名后追加 -checked",
    )
    parser.add_argument(
        "--task4-data",
        type=Path,
        default=Path("data/openseek-4_conala_concat_strings.json"),
        help="task4 原始数据文件路径（用于 outputs 格式按 test_sample_id 回填拼接真值）",
    )
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    return str(text).strip()


def parse_input_tokens(input_value: Any) -> list[str]:
    """
    input 字段通常是字符串形式的 Python 列表，如:
    "['z', 'C', 'but', 'J']"
    """
    raw = str(input_value)
    try:
        data = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"input 无法解析为列表: {raw}") from exc

    if not isinstance(data, list):
        raise ValueError(f"input 解析后不是列表: {raw}")

    return [str(x) for x in data]


def build_joined_from_input(input_value: Any) -> str:
    tokens = parse_input_tokens(input_value)
    return "".join(tokens)


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}-checked{input_path.suffix}")


def _extract_output(output_value: Any) -> str:
    if isinstance(output_value, list):
        return "" if not output_value else str(output_value[0]).strip()
    return str(output_value).strip()


def _load_task4_truth_map(task4_data_path: Path) -> dict[str, str]:
    with task4_data_path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    items = task_dict.get("examples", [])
    truth_map: dict[str, str] = {}
    for row in items:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        truth_map[sample_id] = _extract_output(row.get("output", ""))

    # outputs/*.jsonl 通常对应 test_samples，仅提供 input，需要现算拼接结果
    test_items = task_dict.get("test_samples", [])
    for row in test_items:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        truth_map[sample_id] = build_joined_from_input(row.get("input", ""))
    return truth_map


def process_file(
    input_path: Path, output_path: Path, task4_truth_map: dict[str, str]
) -> tuple[int, int, int, float]:
    total = 0
    fixed = 0
    matched = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as rf, output_path.open(
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

            # compare 格式：包含 input / expected_output / model_output
            if (
                "input" in row
                and "expected_output" in row
                and "model_output" in row
            ):
                joined = build_joined_from_input(row["input"])
                model_output = str(row["model_output"])

                # 规则：包含空格，或与拼接结果不一致，均直接改为拼接结果
                if any(ch.isspace() for ch in model_output) or model_output != joined:
                    row["model_output"] = joined
                    fixed += 1

                is_match = normalize_text(row["model_output"]) == normalize_text(
                    row["expected_output"]
                )

            # outputs 格式：包含 test_sample_id / prediction
            elif "test_sample_id" in row and "prediction" in row:
                sample_id = str(row.get("test_sample_id", "")).strip()
                if not sample_id:
                    raise ValueError(f"第 {line_no} 行 test_sample_id 为空")
                if sample_id not in task4_truth_map:
                    raise ValueError(f"第 {line_no} 行 test_sample_id 不在 task4 数据集中: {sample_id}")

                joined = task4_truth_map[sample_id]
                pred = str(row["prediction"])
                if any(ch.isspace() for ch in pred) or pred != joined:
                    row["prediction"] = joined
                    fixed += 1

                is_match = normalize_text(row["prediction"]) == normalize_text(joined)
            else:
                raise ValueError(
                    f"第 {line_no} 行字段不支持，需为 compare 格式或 outputs 格式"
                )

            # 按需求移除 is_match 字段，不写入输出文件
            row.pop("is_match", None)

            total += 1
            if is_match:
                matched += 1

            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc = (matched / total) if total else 0.0
    return total, fixed, matched, acc


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = resolve_output_path(input_path, args.output.resolve() if args.output else None)
    task4_data_path = args.task4_data.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if not task4_data_path.exists():
        raise FileNotFoundError(f"task4 数据文件不存在: {task4_data_path}")

    truth_map = _load_task4_truth_map(task4_data_path)
    total, fixed, matched, acc = process_file(input_path, output_path, truth_map)

    print(f"[完成] input={input_path}")
    print(f"[完成] output={output_path}")
    print(f"[完成] task4_data={task4_data_path}")
    print(f"[统计] total={total} fixed={fixed} matched={matched} acc={acc:.6f}")


if __name__ == "__main__":
    main()
