from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


TASK_IDS = tuple(range(1, 9))


def script_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    root = script_root()
    parser = argparse.ArgumentParser(description="Rebuild the final FlagOS submission zip from jsonl files.")
    parser.add_argument("--source-dir", default=str(root / "final_outputs"))
    parser.add_argument("--output-dir", default=str(root / "build" / "submission"))
    parser.add_argument("--zip-path", default=str(root / "build" / "submission.zip"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid jsonl: {exc}") from exc
            count += 1
    return count


def resolve_task_file(source_dir: Path, task_id: int) -> Path:
    matches = sorted(source_dir.glob(f"openseek-{task_id}-v*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"missing task {task_id} file under {source_dir}")
    return matches[-1]


def build_zip(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in TASK_IDS:
            file_path = resolve_task_file(output_dir, task_id)
            zf.write(file_path, arcname=file_path.name)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    zip_path = Path(args.zip_path).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"source dir not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("openseek-*-v*.jsonl"):
        stale.unlink()

    summaries: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        src = resolve_task_file(source_dir, task_id)
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        summaries.append(
            {
                "task": task_id,
                "file": dst.name,
                "rows": load_jsonl_count(dst),
                "sha256": sha256_file(dst)
            }
        )

    build_zip(output_dir, zip_path)
    for item in summaries:
        print(
            f"[task {item['task']}] {item['file']} rows={item['rows']} sha256={item['sha256']}"
        )
    print(f"[zip] {zip_path} sha256={sha256_file(zip_path)}")


if __name__ == "__main__":
    main()
