from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from validate_submission_zip import validate_zip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply final platform-validated overrides to a submission zip.")
    parser.add_argument("--input", required=True, help="Input submission zip")
    parser.add_argument("--output", required=True, help="Output submission zip")
    parser.add_argument("--overrides", default="configs/final_overrides.json", help="Override manifest JSON")
    return parser


def _load_overrides(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for item in data.get("overrides", []):
        task_file = str(item["task_file"])
        sample_id = str(item["test_sample_id"])
        prediction = str(item["prediction"])
        if sample_id in grouped[task_file]:
            raise ValueError(f"Duplicate override for {task_file}:{sample_id}")
        grouped[task_file][sample_id] = prediction
    return dict(grouped)


def _read_jsonl(payload: bytes, name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != {"test_sample_id", "prediction"}:
            raise ValueError(f"{name}:{line_number} must contain exactly test_sample_id and prediction")
        records.append(record)
    return records


def main() -> int:
    args = build_parser().parse_args()
    input_zip = Path(args.input)
    output_zip = Path(args.output)
    overrides = _load_overrides(Path(args.overrides))

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    applied = 0
    with zipfile.ZipFile(input_zip) as source, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name in sorted(source.namelist()):
            if name.endswith("/"):
                continue
            if name not in overrides:
                target.writestr(name, source.read(name))
                continue
            rows = _read_jsonl(source.read(name), name)
            task_overrides = overrides[name]
            seen = set()
            for row in rows:
                sample_id = str(row["test_sample_id"])
                seen.add(sample_id)
                if sample_id in task_overrides:
                    row["prediction"] = task_overrides[sample_id]
                    applied += 1
            missing = sorted(set(task_overrides) - seen)
            if missing:
                raise ValueError(f"{name} is missing override ids: {missing}")
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            target.writestr(name, payload.encode("utf-8"))

    _, errors = validate_zip(output_zip)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Wrote {output_zip} with {applied} final overrides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
