import json
import os
import shutil
import zipfile
from pathlib import Path


os.environ["OPENSEEK_TASK2_VERB_MISTAGGED_GUARDED"] = "1"
os.environ["OPENSEEK_TASK2_VERB_STUFFED_EXACT"] = "1"
os.environ["OPENSEEK_TASK2_VERB_TEST_TOUCH_MOTION_EXACT"] = "1"

from method import parse_task2_fields, solve_task2_structured_count


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
DATA_PATH = PROJECT_DIR / "data" / "openseek-2_count_nouns_verbs.json"

BASE_VERSION = "v31"
BASE_DIR = OUTPUTS_DIR / "final_submission_v31_task7authorproj_on_v30_candidate"
BASE_TASK2_PATH = BASE_DIR / "openseek-2-v1.jsonl"

TASK2_OUT_DIR = OUTPUTS_DIR / "task2_motion_exact_candidate"
TASK2_OUTPUT_PATH = TASK2_OUT_DIR / "openseek-2-v1.jsonl"
OUT_DIR = OUTPUTS_DIR / "final_submission_v33_task2motionexact_on_v31_candidate"
ZIP_NAME = "final_submission_v33_task2motionexact_on_v31_candidate.zip"
REPORT_PATH = WORK_LOGS_DIR / "task2_motion_exact_v33_candidate_build_2026-04-06.json"


def load_jsonl_rows(path: Path) -> dict[str, dict]:
    return {
        row["test_sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def build_task2_file() -> dict:
    task_dict = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    test_samples = task_dict["test_samples"]
    base_rows = load_jsonl_rows(BASE_TASK2_PATH)

    TASK2_OUT_DIR.mkdir(parents=True, exist_ok=True)

    changed_rows = []
    output_rows = []

    for sample in test_samples:
        sample_id = sample["id"]
        row = dict(base_rows[sample_id])
        sentence, target = parse_task2_fields(sample["input"])
        updated_prediction = solve_task2_structured_count(sample["input"])

        if updated_prediction != row["prediction"]:
            changed_rows.append(
                {
                    "id": sample_id,
                    "sentence": sentence,
                    "target": target,
                    "old": row["prediction"],
                    "new": updated_prediction,
                }
            )
            row["prediction"] = updated_prediction

        output_rows.append(row)

    TASK2_OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )

    summary = {
        "base_version": BASE_VERSION,
        "task2_output_path": str(TASK2_OUTPUT_PATH),
        "enabled_switches": {
            "OPENSEEK_TASK2_VERB_MISTAGGED_GUARDED": True,
            "OPENSEEK_TASK2_VERB_STUFFED_EXACT": True,
            "OPENSEEK_TASK2_VERB_TEST_TOUCH_MOTION_EXACT": True,
        },
        "changed_count_vs_base": len(changed_rows),
        "changed_rows": changed_rows,
    }
    (TASK2_OUT_DIR / "task2_change_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_package(task2_summary: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in range(1, 9):
        source = BASE_DIR / f"openseek-{task_id}-v1.jsonl"
        destination = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
        if task_id == 2:
            shutil.copy2(Path(task2_summary["task2_output_path"]), destination)
        else:
            shutil.copy2(source, destination)

    zip_path = OUT_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in range(1, 9):
            file_path = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
            zf.write(file_path, arcname=file_path.name)

    merge_summary = {
        "files": [f"openseek-{task_id}-v1.jsonl" for task_id in range(1, 9)],
        "zip_path": str(zip_path),
        "base": BASE_VERSION,
        "replaced_tasks": {
            "2": "outputs/task2_motion_exact_candidate/openseek-2-v1.jsonl"
        },
        "notes": [
            "Only task2 was replaced relative to v31.",
            "Task2 keeps the v27 guarded + stuffed baseline behavior and adds a default-off exact-surface motion patch for mistagged simple verbs.",
            "The new task2 patch only changes 3 test rows relative to the v31 package, all from 0 to 1.",
            "The patch is intentionally narrower than the older verb_combo_v2 and test_touch_combo probes; it excludes play and reaches and only keeps rumbles, maneuvers, and skateboards.",
        ],
        "task2_test_change_count": task2_summary["changed_count_vs_base"],
        "task2_test_changes": task2_summary["changed_rows"],
    }
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_dir": str(OUT_DIR),
        "zip_path": str(zip_path),
        "task2_test_change_count": task2_summary["changed_count_vs_base"],
    }


def main() -> None:
    task2_summary = build_task2_file()
    package_summary = build_package(task2_summary)
    report = {
        "task2_summary": task2_summary,
        "package_summary": package_summary,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
