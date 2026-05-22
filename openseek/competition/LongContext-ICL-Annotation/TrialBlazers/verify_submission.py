from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


TASK_IDS = tuple(range(1, 9))
DEFAULT_ROWS = {task_id: 500 for task_id in range(1, 8)} | {8: 166}


def script_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    root = script_root()
    parser = argparse.ArgumentParser(description="Verify final submission files and zip structure.")
    parser.add_argument("--manifest", default=str(root / "manifest.json"))
    parser.add_argument("--submission-dir", default=str(root / "final_outputs"))
    parser.add_argument("--zip-path", default="", help="Defaults to <submission-dir>.zip when present.")
    parser.add_argument("--skip-sha", action="store_true", help="Only check structure and row counts.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def expected_by_task(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["task_id"]): item for item in manifest["tasks"]}


def default_expected_by_task() -> dict[int, dict[str, Any]]:
    return {
        task_id: {
            "task_id": task_id,
            "file": f"openseek-{task_id}-v1.jsonl",
            "rows": DEFAULT_ROWS[task_id],
        }
        for task_id in TASK_IDS
    }


def resolve_zip_path(raw_zip_path: str, submission_dir: Path) -> Path | None:
    if raw_zip_path:
        return Path(raw_zip_path).resolve()
    sibling = submission_dir.with_suffix(".zip")
    if sibling.exists():
        return sibling.resolve()
    return None


def resolve_task_file(submission_dir: Path, task_id: int) -> Path:
    matches = sorted(submission_dir.glob(f"openseek-{task_id}-v*.jsonl"))
    if not matches:
        raise AssertionError(f"missing task {task_id} jsonl under {submission_dir}")
    if len(matches) > 1:
        raise AssertionError(f"multiple files for task {task_id}: {[p.name for p in matches]}")
    return matches[0]


def validate_jsonl(path: Path, task_id: int) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_no} is not valid json: {exc}") from exc
            if "prediction" not in row:
                raise AssertionError(f"{path}:{line_no} missing prediction field")
            if task_id == 8:
                code = row.get("code")
                pred = row.get("prediction")
                if code is not None and pred is not None and str(code) != str(pred):
                    raise AssertionError(f"{path}:{line_no} task8 code field differs from prediction")
            rows += 1
    return rows


def validate_zip(zip_path: Path, expected_files: list[str]) -> None:
    if not zip_path.exists():
        raise AssertionError(f"zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    expected = sorted(expected_files)
    if names != expected:
        raise AssertionError(f"zip entries mismatch. expected={expected}, actual={names}")
    nested = [name for name in names if "/" in name or name.endswith("/")]
    if nested:
        raise AssertionError(f"zip contains nested paths: {nested}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    submission_dir = Path(args.submission_dir).resolve()
    zip_path = resolve_zip_path(args.zip_path, submission_dir)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        expected = expected_by_task(manifest)
        check_sha = not args.skip_sha
    else:
        manifest = {}
        expected = default_expected_by_task()
        check_sha = False
        print(f"[info] manifest not found, using default row-count checks: {manifest_path}")

    expected_files: list[str] = []
    for task_id in TASK_IDS:
        item = expected[task_id]
        path = resolve_task_file(submission_dir, task_id)
        expected_files.append(path.name)
        if path.name != item["file"]:
            raise AssertionError(f"task {task_id} filename mismatch: {path.name} != {item['file']}")
        rows = validate_jsonl(path, task_id)
        if rows != int(item["rows"]):
            raise AssertionError(f"task {task_id} rows mismatch: {rows} != {item['rows']}")
        file_sha = sha256_file(path)
        if check_sha and file_sha != item["sha256"]:
            raise AssertionError(f"task {task_id} sha256 mismatch: {file_sha} != {item['sha256']}")
        print(f"[ok] task {task_id}: {path.name} rows={rows} sha256={file_sha}")

    if zip_path is not None:
        validate_zip(zip_path, expected_files)
        zip_sha = sha256_file(zip_path)
        print(f"[ok] zip entries: {zip_path}")
        print(f"[info] zip sha256={zip_sha}")
        final_zip_sha = manifest.get("final_submission_zip", {}).get("sha256")
        final_zip_path = manifest.get("final_submission_zip", {}).get("path")
        if check_sha and final_zip_path and zip_path.name == Path(final_zip_path).name:
            if final_zip_sha and zip_sha != final_zip_sha:
                raise AssertionError(f"final zip sha256 mismatch: {zip_sha} != {final_zip_sha}")
    else:
        print("[info] zip path not provided and sibling zip not found; skipped zip entry check")
    print("[verify] all checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[verify] failed: {exc}", file=sys.stderr)
        raise
