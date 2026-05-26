from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from method import postprocess_prediction


TASK_IDS = tuple(range(1, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize and package FlagOS submission files.")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing openseek-*-v*.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write sanitized submission files.",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="",
        help="Optional output zip path. If empty, writes <output-dir>.zip beside the directory.",
    )
    return parser.parse_args()


def resolve_task_file(source_dir: Path, task_id: int) -> Path:
    candidates = sorted(source_dir.glob(f"openseek-{task_id}-v*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Missing task file for task {task_id} under {source_dir}")
    return candidates[-1]


def sanitize_row(task_id: int, row: dict) -> dict:
    clean_row = dict(row)
    if task_id == 8:
        raw_prediction = clean_row.get("prediction")
        raw_text = "" if raw_prediction is None else str(raw_prediction)
        sanitized = postprocess_prediction(raw_text, 8) or ""
        clean_row["prediction"] = sanitized
        # Extra compatibility field for evaluators that expect `code`.
        clean_row["code"] = sanitized
        return clean_row

    if clean_row.get("prediction") is None:
        clean_row["prediction"] = ""
    elif not isinstance(clean_row["prediction"], str):
        clean_row["prediction"] = str(clean_row["prediction"])
    return clean_row


def sanitize_file(task_id: int, src_path: Path, dst_path: Path) -> tuple[int, int]:
    total = 0
    fixed = 0
    with src_path.open("r", encoding="utf-8") as reader, dst_path.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            before = row.get("prediction")
            cleaned = sanitize_row(task_id, row)
            after = cleaned.get("prediction")
            if before != after or (task_id == 8 and cleaned.get("code") != row.get("code")):
                fixed += 1
            writer.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    return total, fixed


def build_zip(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in TASK_IDS:
            file_path = resolve_task_file(output_dir, task_id)
            zf.write(file_path, arcname=file_path.name)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task_id in TASK_IDS:
        src_path = resolve_task_file(source_dir, task_id)
        dst_path = output_dir / src_path.name
        total, fixed = sanitize_file(task_id, src_path, dst_path)
        summaries.append((task_id, total, fixed, dst_path))

    zip_path = Path(args.zip_path).resolve() if args.zip_path else output_dir.with_suffix(".zip")
    build_zip(output_dir, zip_path)

    for task_id, total, fixed, dst_path in summaries:
        print(f"[task {task_id}] rows={total} fixed={fixed} file={dst_path}")
    print(f"[zip] {zip_path}")


if __name__ == "__main__":
    main()
