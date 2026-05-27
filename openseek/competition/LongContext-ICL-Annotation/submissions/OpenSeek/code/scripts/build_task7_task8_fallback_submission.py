#!/usr/bin/env python3
"""Build the final task-7 + task-8 fallback submission candidate.

The base candidate already uses the FlagScale full run for tasks 1-7 and
deterministic PyTorch wrappers for task 8. This script applies a narrow task 7
cleanup for visibly polluted generation outputs by replacing them with answers
from the previous validated candidate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PREDICTIONS = (
    ROOT
    / "outputs/flagscale_task8_fallback_candidate_final/flagscale_task8_fallback_candidate/predictions"
)
TASK7_FALLBACK = ROOT / "outputs/final_submission_candidate/predictions/openseek-7-v1.jsonl"
OUT_DIR = ROOT / "outputs/flagscale_task7_task8_fallback_candidate_final/flagscale_task7_task8_fallback_candidate"

TASK7_REPAIR_IDS = {
    "openseek-7-2c42773c25b64b7d8dccf7cd34082c20",
    "openseek-7-85a77001724b43c8ac81c878b696848e",
    "openseek-7-86cd91cfd1ab4b50a8947846d20435cc",
    "openseek-7-62cd015a8b884abebed9d00cce40e6bd",
    "openseek-7-c4bb76159c934ceab9b8b34f92265773",
    "openseek-7-cb8fa1a62ec44bb2bf46407ab442a3ce",
    "openseek-7-ace3b952632242c886fc6640f59f778b",
    "openseek-7-765797f7199e4d1688670c986ca85798",
    "openseek-7-92b1856702a64b1eb25c06b84c4929ed",
    "openseek-7-01e4fbfca16345b898064b73d9c18659",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _short_answer_ok(text: str) -> bool:
    stripped = text.strip()
    blocked = ("Candidate", "Answer:", "Explanation", "```", "Clue:")
    return bool(stripped) and "\n" not in stripped and len(stripped.split()) <= 8 and not any(
        marker in stripped for marker in blocked
    )


def build(base_predictions: Path, task7_fallback: Path, out_dir: Path) -> dict[str, Any]:
    pred_dir = out_dir / "predictions"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pred_dir.mkdir(parents=True)

    for source in sorted(base_predictions.glob("openseek-*-v1.jsonl")):
        shutil.copy2(source, pred_dir / source.name)

    fallback_rows = {row["test_sample_id"]: row for row in _read_jsonl(task7_fallback)}
    task7_path = pred_dir / "openseek-7-v1.jsonl"
    repaired_rows: list[dict[str, Any]] = []
    replacements: list[tuple[str, str, str]] = []

    for row in _read_jsonl(task7_path):
        sample_id = row["test_sample_id"]
        if sample_id in TASK7_REPAIR_IDS:
            old_prediction = str(row.get("prediction", ""))
            replacement = dict(fallback_rows[sample_id])
            replacement["meta"] = dict(replacement.get("meta") or {})
            replacement["meta"]["repair_source"] = "final_submission_candidate_task7_fallback"
            replacement["meta"]["replaced_prediction"] = old_prediction
            row = replacement
            replacements.append((sample_id, old_prediction, str(row.get("prediction", ""))))
        repaired_rows.append(row)

    _write_jsonl(task7_path, repaired_rows)

    source_candidate = base_predictions.parent
    for name in ("submission_validation.json", "source_submission_validation.json", "task8_runtime_report.json"):
        source = source_candidate / name
        if source.exists():
            shutil.copy2(source, out_dir / name)

    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(pred_dir.glob("openseek-*-v1.jsonl")):
            info = zipfile.ZipInfo(file.name, date_time=(2026, 5, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, file.read_bytes())
    shutil.copy2(out_dir / "submission.zip", out_dir / "submission-task7-task8-fallback.zip")

    task7_ok = sum(_short_answer_ok(str(row.get("prediction", ""))) for row in repaired_rows)
    return {
        "out_dir": str(out_dir),
        "submission": str(out_dir / "submission.zip"),
        "replacements": len(replacements),
        "task7_short_answer_ok": task7_ok,
        "task7_total": len(repaired_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", type=Path, default=BASE_PREDICTIONS)
    parser.add_argument("--task7-fallback", type=Path, default=TASK7_FALLBACK)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    result = build(args.base_predictions, args.task7_fallback, args.out_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
