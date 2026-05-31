import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 examples 对比结果的任务准确率。")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="examples",
        help="对比结果目录（默认 examples）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.input_dir).resolve()
    files = sorted(base_dir.glob("openseek-*-examples-compare.jsonl"))

    if not files:
        print(f"[提示] 未找到对比文件：{base_dir}")
        return

    overall_total = 0
    overall_matched = 0
    per_task: list[dict] = []

    for file_path in files:
        total = 0
        matched = 0
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if bool(row.get("is_match", False)):
                    matched += 1
        accuracy = (matched / total) if total else 0.0
        per_task.append(
            {
                "task_file": file_path.name,
                "total": total,
                "matched": matched,
                "accuracy": accuracy,
            }
        )
        overall_total += total
        overall_matched += matched

    for item in per_task:
        print(
            f"{item['task_file']}: matched={item['matched']}/{item['total']}, "
            f"accuracy={item['accuracy']:.2%}"
        )

    overall_accuracy = (overall_matched / overall_total) if overall_total else 0.0
    print(
        f"overall: matched={overall_matched}/{overall_total}, "
        f"accuracy={overall_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
