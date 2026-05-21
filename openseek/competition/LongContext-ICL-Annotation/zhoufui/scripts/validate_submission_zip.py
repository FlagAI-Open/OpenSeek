from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "openseek-1-v1.jsonl": 500,
    "openseek-2-v1.jsonl": 500,
    "openseek-3-v1.jsonl": 500,
    "openseek-4-v1.jsonl": 500,
    "openseek-5-v1.jsonl": 500,
    "openseek-6-v1.jsonl": 500,
    "openseek-7-v1.jsonl": 500,
    "openseek-8-v1.jsonl": 166,
}

REQUIRED_FIELDS = {"test_sample_id", "prediction"}


def _format_error(name: str, message: str) -> str:
    return f"{name}: {message}"


def _parse_jsonl(name: str, payload: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if payload.startswith(b"\xef\xbb\xbf"):
        errors.append(_format_error(name, "file has UTF-8 BOM; platform packages should be UTF-8 without BOM"))
        payload = payload[3:]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [], [_format_error(name, f"not valid UTF-8: {exc}")]

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            errors.append(_format_error(name, f"line {line_number} is empty"))
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(_format_error(name, f"line {line_number} is not valid JSON: {exc}"))
            continue
        if not isinstance(record, dict):
            errors.append(_format_error(name, f"line {line_number} is {type(record).__name__}, expected object"))
            continue
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            errors.append(_format_error(name, f"line {line_number} missing fields: {', '.join(missing)}"))
        extra = sorted(set(record) - REQUIRED_FIELDS)
        if extra:
            errors.append(_format_error(name, f"line {line_number} has extra fields: {', '.join(extra)}"))
        if not isinstance(record.get("test_sample_id"), str) or not record.get("test_sample_id"):
            errors.append(_format_error(name, f"line {line_number} has invalid test_sample_id"))
        if not isinstance(record.get("prediction"), str):
            errors.append(_format_error(name, f"line {line_number} prediction must be a string"))
        records.append(record)
    return records, errors


def validate_zip(path: Path) -> tuple[dict[str, int], list[str]]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    if not path.exists():
        return counts, [f"{path}: file does not exist"]
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return counts, [f"{path}: not a valid zip file: {exc}"]

    with archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        root_jsonl = sorted(name for name in names if name.endswith(".jsonl") and "/" not in name and "\\" not in name)
        nested = sorted(name for name in names if name.endswith(".jsonl") and name not in root_jsonl)
        unexpected = sorted(name for name in names if name not in EXPECTED_COUNTS)

        missing = sorted(set(EXPECTED_COUNTS) - set(root_jsonl))
        if missing:
            errors.append(f"missing root JSONL files: {', '.join(missing)}")
        if nested:
            errors.append(f"JSONL files must be at zip root, found nested: {', '.join(nested[:8])}")
        if unexpected:
            errors.append(f"unexpected files at zip root or archive: {', '.join(unexpected[:8])}")

        for name in sorted(set(root_jsonl) & set(EXPECTED_COUNTS)):
            records, file_errors = _parse_jsonl(name, archive.read(name))
            counts[name] = len(records)
            errors.extend(file_errors)
            expected = EXPECTED_COUNTS[name]
            if len(records) != expected:
                errors.append(_format_error(name, f"has {len(records)} records, expected {expected}"))
            seen_ids: set[str] = set()
            duplicates: set[str] = set()
            for record in records:
                sample_id = record.get("test_sample_id")
                if sample_id in seen_ids:
                    duplicates.add(str(sample_id))
                elif isinstance(sample_id, str):
                    seen_ids.add(sample_id)
            if duplicates:
                preview = ", ".join(sorted(duplicates)[:5])
                errors.append(_format_error(name, f"duplicate test_sample_id values: {preview}"))
    return counts, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an OpenSeek Track 3 submission zip package")
    parser.add_argument("zip_path", help="Path to the submission zip")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counts, errors = validate_zip(Path(args.zip_path))
    for name in sorted(EXPECTED_COUNTS):
        count = counts.get(name, 0)
        print(f"{name}: {count}/{EXPECTED_COUNTS[name]}")
    if errors:
        print("\nINVALID submission package:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nOK: submission package structure, JSONL records, UTF-8 encoding, and row counts look valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
