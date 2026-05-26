import json
import shutil
import zipfile
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

BASE_VERSION = "v30"
BASE_DIR = OUTPUTS_DIR / "final_submission_v30_task8clean2_on_v27_candidate"
BASE_TASK7_PATH = BASE_DIR / "openseek-7-v1.jsonl"

TASK7_SOURCE_VERSION = "authorproj_live_r2"
TASK7_SOURCE_DIR = OUTPUTS_DIR / "task7_author_projection_live_candidate"
TASK7_SOURCE_PATH = TASK7_SOURCE_DIR / "openseek-7-v1.jsonl"

OUT_DIR = OUTPUTS_DIR / "final_submission_v31_task7authorproj_on_v30_candidate"
ZIP_NAME = "final_submission_v31_task7authorproj_on_v30_candidate.zip"
REPORT_STEM = "task7_v31_authorproj_candidate_build_2026-04-06"


def load_jsonl_rows(path: Path) -> dict[str, dict]:
    return {
        row["test_sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def build_task7_summary() -> dict:
    base_rows = load_jsonl_rows(BASE_TASK7_PATH)
    candidate_rows = load_jsonl_rows(TASK7_SOURCE_PATH)

    changed_rows = []
    unchanged_count = 0
    added_ids = []
    missing_ids = []

    for sample_id, candidate_row in candidate_rows.items():
        base_row = base_rows.get(sample_id)
        if base_row is None:
            added_ids.append(sample_id)
            continue
        if candidate_row["prediction"] != base_row["prediction"]:
            changed_rows.append(
                {
                    "id": sample_id,
                    "old_prediction": base_row["prediction"],
                    "new_prediction": candidate_row["prediction"],
                    "old_length": len(base_row["prediction"]),
                    "new_length": len(candidate_row["prediction"]),
                }
            )
        else:
            unchanged_count += 1

    for sample_id in base_rows:
        if sample_id not in candidate_rows:
            missing_ids.append(sample_id)

    summary = {
        "base_version": BASE_VERSION,
        "task7_source_version": TASK7_SOURCE_VERSION,
        "base_task7_path": str(BASE_TASK7_PATH),
        "task7_source_path": str(TASK7_SOURCE_PATH),
        "base_row_count": len(base_rows),
        "candidate_row_count": len(candidate_rows),
        "changed_count_vs_base": len(changed_rows),
        "unchanged_count_vs_base": unchanged_count,
        "added_ids": added_ids,
        "missing_ids": missing_ids,
        "changed_rows": changed_rows,
    }
    return summary


def build_package(task7_summary: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in range(1, 9):
        source = BASE_DIR / f"openseek-{task_id}-v1.jsonl"
        destination = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
        if task_id == 7:
            shutil.copy2(TASK7_SOURCE_PATH, destination)
        else:
            shutil.copy2(source, destination)

    zip_path = OUT_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in range(1, 9):
            file_path = OUT_DIR / f"openseek-{task_id}-v1.jsonl"
            zf.write(file_path, arcname=file_path.name)

    task7_patch_summary = {
        "base": BASE_VERSION,
        "patched_task7_source": "outputs/task7_author_projection_live_candidate/openseek-7-v1.jsonl",
        "selection_rule": "Carry forward the live author-projection Task7 output built on the v26 typed-narrow rerank base with author-only direct-fact projection enabled.",
        "changed_count_vs_base": task7_summary["changed_count_vs_base"],
    }
    (OUT_DIR / "task7_patch_summary.json").write_text(
        json.dumps(task7_patch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merge_summary = {
        "files": [f"openseek-{task_id}-v1.jsonl" for task_id in range(1, 9)],
        "zip_path": str(zip_path),
        "base": BASE_VERSION,
        "replaced_tasks": {
            "7": "outputs/task7_author_projection_live_candidate/openseek-7-v1.jsonl"
        },
        "notes": [
            "Only task7 was replaced relative to v30.",
            "Task7 carries forward the typed-narrow append_unique rerank base and adds author-only direct-fact projection for quoted_work_author_relation rows.",
            "The selected append_unique reserved secondary judge slots setting is 2 after the targeted aligned author-bucket sweep tied across 2/3/4.",
            "Formal aligned targeted evaluation recovered a judge gain from 0.3 to 0.4 and oracle hit rate from 0.4 to 0.6 on the author bucket after the routing fix.",
            "This package is intended as a narrow Task7 candidate layered on top of the current v30 stable package format.",
        ],
        "task7_test_change_count": task7_summary["changed_count_vs_base"],
        "task7_test_changes": task7_summary["changed_rows"],
    }
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_dir": str(OUT_DIR),
        "zip_path": str(zip_path),
        "task7_test_change_count": task7_summary["changed_count_vs_base"],
    }


def write_markdown_report(report: dict) -> None:
    task7_summary = report["task7_summary"]
    package_summary = report["package_summary"]
    lines = [
        "# Task7 v31 Author Projection Candidate Build",
        "",
        f"- Base: `{BASE_VERSION}`",
        f"- Task7 source: `{TASK7_SOURCE_VERSION}`",
        f"- Changed task7 test rows vs base: `{task7_summary['changed_count_vs_base']}`",
        f"- Package zip: `{package_summary['zip_path']}`",
        "",
        "## Candidate rationale",
        "",
        "- Keep the existing v30 package untouched except for task7.",
        "- Reuse the typed-narrow Task7 rerank base and layer author-only direct-fact projection on top.",
        "- Use the conservative reserved secondary slots setting `2` because the targeted aligned author-bucket sweep tied across `2/3/4`.",
        "",
        "## Changed rows",
        "",
    ]
    for row in task7_summary["changed_rows"]:
        lines.extend(
            [
                f"- `{row['id']}`",
                f"  - length: `{row['old_length']} -> {row['new_length']}`",
                f"  - old: `{row['old_prediction']}`",
                f"  - new: `{row['new_prediction']}`",
            ]
        )

    md_path = WORK_LOGS_DIR / f"{REPORT_STEM}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    task7_summary = build_task7_summary()
    package_summary = build_package(task7_summary)
    report = {
        "task7_summary": task7_summary,
        "package_summary": package_summary,
    }
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = WORK_LOGS_DIR / f"{REPORT_STEM}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
