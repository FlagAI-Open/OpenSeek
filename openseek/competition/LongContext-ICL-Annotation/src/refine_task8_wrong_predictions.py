import argparse
import ast
import json
import os
from pathlib import Path

from infer_task8_test_samples import (
    TASK8_CANONICAL_DESCRIPTION,
    _extract_code_candidate,
    _extract_output,
    _infer_task8_item,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK8_DATA_FILE = REPO_ROOT / "data" / "openseek-8_kernel_generation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅重跑 Task8 结果中明显错误的样本，并使用 ReAct 修复。"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/openseek-8-test_samples-predictions (1).jsonl",
        help="原始预测文件路径（jsonl）。",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/openseek-8-test_samples-predictions-regenerated.jsonl",
        help="修复后输出文件路径（jsonl）。",
    )
    parser.add_argument(
        "--audit_file",
        type=str,
        default="outputs/openseek-8-test_samples-predictions-regenerated.audit.jsonl",
        help="修复审计文件路径（jsonl）。",
    )
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--task8_shot_k", type=int, default=6, help="检索示例数。")
    parser.add_argument("--task8_retrieval_pool_size", type=int, default=200, help="候选池大小。")
    parser.add_argument("--max_react_rounds", type=int, default=2, help="ReAct 最大轮数。")
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否打印模型原始输出预览。",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_bad_prediction(prediction: str) -> tuple[bool, str]:
    cleaned = _extract_code_candidate(prediction)
    if not cleaned.strip():
        return True, "empty_after_clean"
    try:
        tree = ast.parse(cleaned)
    except SyntaxError as e:
        return True, f"syntax_error: {e}"
    has_def = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    if not has_def:
        return True, "no_function_or_class_def"
    # 只在筛选阶段做静态判断，避免执行模型生成代码导致卡死。
    return False, ""


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"

    input_file = _resolve_path(args.input_file)
    output_file = _resolve_path(args.output_file)
    audit_file = _resolve_path(args.audit_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    with TASK8_DATA_FILE.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    test_sample_input_by_id: dict[str, str] = {}
    for sample in task_dict.get("test_samples", []):
        test_sample_input_by_id[str(sample.get("id", "")).strip()] = str(sample.get("input", ""))

    rows = _load_jsonl_rows(input_file)
    print(f"[读取] input={input_file} rows={len(rows)}")

    regenerate_ids: list[str] = []
    audit_rows: list[dict] = []
    for row in rows:
        sample_id = str(row.get("test_sample_id", "")).strip()
        prediction = str(row.get("prediction", ""))
        is_bad, reason = _is_bad_prediction(prediction)
        if is_bad:
            regenerate_ids.append(sample_id)
            audit_rows.append(
                {
                    "test_sample_id": sample_id,
                    "status": "needs_regenerate",
                    "reason": reason,
                }
            )

    print(f"[筛选] need_regenerate={len(regenerate_ids)} / total={len(rows)}")
    regenerate_set = set(regenerate_ids)
    regenerated_map: dict[str, str] = {}

    for idx, sample_id in enumerate(regenerate_ids, start=1):
        input_text = test_sample_input_by_id.get(sample_id, "")
        if not input_text:
            print(f"[跳过] {sample_id} 不在 test_samples 中")
            audit_rows.append(
                {
                    "test_sample_id": sample_id,
                    "status": "skipped",
                    "reason": "missing_input_text",
                }
            )
            continue

        print(f"[重跑] {idx}/{len(regenerate_ids)} id={sample_id}")
        infer_row = _infer_task8_item(
            item={"example_id": sample_id, "input_text": input_text},
            task_description=TASK8_CANONICAL_DESCRIPTION,
            task_id=8,
            cleaned_examples=cleaned_examples,
            task8_shot_k=args.task8_shot_k,
            task8_retrieval_pool_size=args.task8_retrieval_pool_size,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            enable_react=True,
            max_react_rounds=max(1, args.max_react_rounds),
        )
        new_prediction = str(infer_row.get("prediction", ""))
        regenerated_map[sample_id] = new_prediction
        is_bad_after, reason_after = _is_bad_prediction(new_prediction)
        audit_rows.append(
            {
                "test_sample_id": sample_id,
                "status": "regenerated",
                "passed_static_check": not is_bad_after,
                "error_preview": "" if not is_bad_after else reason_after[:300],
                "new_prediction_len": len(new_prediction),
            }
        )

    fixed_count = 0
    with output_file.open("w", encoding="utf-8") as wf:
        for row in rows:
            sample_id = str(row.get("test_sample_id", "")).strip()
            if sample_id in regenerate_set and sample_id in regenerated_map:
                row["prediction"] = regenerated_map[sample_id]
                fixed_count += 1
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    with audit_file.open("w", encoding="utf-8") as af:
        for row in audit_rows:
            af.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[完成] fixed={fixed_count} total={len(rows)}")
    print(f"[输出] {output_file}")
    print(f"[审计] {audit_file}")


if __name__ == "__main__":
    main()
