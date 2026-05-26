import json
import shutil
import zipfile
from pathlib import Path

from task8_semantics import audit_task8_code


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
DATA_PATH = PROJECT_DIR / "data" / "openseek-8_kernel_generation.json"

BASE_VERSION = "v27"
BASE_DIR = OUTPUTS_DIR / "final_submission_v27_task2stuffed_on_v25_candidate"
BASE_TASK8_PATH = BASE_DIR / "openseek-8-v1.jsonl"

PATCH_SOURCE_VERSION = "v14"
PATCH_SOURCE_DIR = OUTPUTS_DIR / "final_submission_v14_task8lexical_clean2_candidate"
PATCH_SOURCE_TASK8_PATH = PATCH_SOURCE_DIR / "openseek-8-v1.jsonl"
PATCH_SUMMARY_PATH = PATCH_SOURCE_DIR / "task8_patch_summary.json"

TASK8_OUT_DIR = OUTPUTS_DIR / "task8_v30_clean2_candidate"
TASK8_OUT_PATH = TASK8_OUT_DIR / "openseek-8-v1.jsonl"

OUT_DIR = OUTPUTS_DIR / "final_submission_v30_task8clean2_on_v27_candidate"
ZIP_NAME = "final_submission_v30_task8clean2_on_v27_candidate.zip"

REPORT_STEM = "task8_v30_candidate_build_2026-04-01"


def load_jsonl_rows(path: Path) -> dict[str, dict]:
    return {
        row["test_sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_task8_test_map() -> dict[str, dict]:
    task8 = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {sample["id"]: sample for sample in task8["test_samples"]}


def build_task8_file() -> dict:
    base_rows = load_jsonl_rows(BASE_TASK8_PATH)
    patch_rows = load_jsonl_rows(PATCH_SOURCE_TASK8_PATH)
    test_map = load_task8_test_map()
    patch_summary = json.loads(PATCH_SUMMARY_PATH.read_text(encoding="utf-8"))
    selected_ids = patch_summary["selected_replacements"]

    TASK8_OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_rows = []
    changed_rows = []

    for sample_id, base_row in base_rows.items():
        row = dict(base_row)
        if sample_id in selected_ids:
            patched_row = patch_rows[sample_id]
            if patched_row["prediction"] != base_row["prediction"]:
                candidate_prediction = patched_row["prediction"]
                audit = audit_task8_code(test_map[sample_id]["input"], candidate_prediction)
                changed_rows.append(
                    {
                        "id": sample_id,
                        "old_length": len(base_row["prediction"]),
                        "new_length": len(candidate_prediction),
                        "wrapper_name": audit["wrapper_name"],
                        "expected_wrapper": audit["expected_wrapper"],
                        "blocker_count": audit["blocker_count"],
                        "warning_count": audit["warning_count"],
                        "family_match": audit["family"]["match"],
                        "candidate_file_source": str(PATCH_SOURCE_TASK8_PATH),
                    }
                )
                row = dict(patched_row)
        output_rows.append(row)

    TASK8_OUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )

    summary = {
        "base_version": BASE_VERSION,
        "patch_source_version": PATCH_SOURCE_VERSION,
        "selected_replacements": selected_ids,
        "task8_output_path": str(TASK8_OUT_PATH),
        "changed_count_vs_base": len(changed_rows),
        "changed_rows": changed_rows,
    }
    (TASK8_OUT_DIR / "task8_change_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_package(task8_summary: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in range(1, 9):
        source = BASE_DIR / f"openseek-{task_id}-v1.jsonl"
        destination = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
        if task_id == 8:
            shutil.copy2(Path(task8_summary["task8_output_path"]), destination)
        else:
            shutil.copy2(source, destination)

    zip_path = OUT_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in range(1, 9):
            file_path = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
            zf.write(file_path, arcname=file_path.name)

    task8_patch_summary = {
        "base": BASE_VERSION,
        "patched_task8_source": "outputs/final_submission_v14_task8lexical_clean2_candidate/openseek-8-v1.jsonl",
        "selected_replacements": task8_summary["selected_replacements"],
        "selection_rule": "Carry forward the 2-row audited-clean lexical subset from v14 onto the current v27 base.",
    }
    (OUT_DIR / "task8_patch_summary.json").write_text(
        json.dumps(task8_patch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merge_summary = {
        "files": [f"openseek-{task_id}-v1.jsonl" for task_id in range(1, 9)],
        "zip_path": str(zip_path),
        "base": BASE_VERSION,
        "replaced_tasks": {
            "8": "outputs/task8_v30_clean2_candidate/openseek-8-v1.jsonl"
        },
        "notes": [
            "Only task8 was replaced relative to v27.",
            "Task8 carries forward the ultra-conservative 2-row lexical clean subset from v14.",
            "Relative to the current v27 submission file, task8 test predictions changed on 2 rows.",
            "Both changed rows had 0 blockers and 0 warnings under task8_candidate_audit.py.",
            "This candidate is intentionally maintenance-scale rather than a broad task8 retrieval rewrite.",
        ],
        "task8_test_change_count": task8_summary["changed_count_vs_base"],
        "task8_test_changes": task8_summary["changed_rows"],
    }
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_dir": str(OUT_DIR),
        "zip_path": str(zip_path),
        "task8_test_change_count": task8_summary["changed_count_vs_base"],
    }


def write_markdown_report(report: dict) -> None:
    task8_summary = report["task8_summary"]
    package_summary = report["package_summary"]
    lines = [
        "# Task8 v30 Candidate Build",
        "",
        f"- Base: `{BASE_VERSION}`",
        f"- Patch source: `{PATCH_SOURCE_VERSION}` ultra-clean task8 subset",
        f"- Changed task8 test rows vs base: `{task8_summary['changed_count_vs_base']}`",
        f"- Package zip: `{package_summary['zip_path']}`",
        "",
        "## Changed rows",
        "",
    ]
    for row in task8_summary["changed_rows"]:
        lines.extend(
            [
                f"- `{row['id']}`",
                f"  - wrapper: `{row['wrapper_name']}` vs expected `{row['expected_wrapper']}`",
                f"  - blockers/warnings: `{row['blocker_count']}/{row['warning_count']}`",
                f"  - family_match: `{row['family_match']}`",
                f"  - length: `{row['old_length']} -> {row['new_length']}`",
            ]
        )

    md_path = WORK_LOGS_DIR / f"{REPORT_STEM}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    task8_summary = build_task8_file()
    package_summary = build_package(task8_summary)
    report = {
        "task8_summary": task8_summary,
        "package_summary": package_summary,
    }
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = WORK_LOGS_DIR / f"{REPORT_STEM}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
