import json
import shutil
import zipfile
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

BASE_VERSION = "v34"
BASE_DIR = OUTPUTS_DIR / "final_submission_v34_task7authorproj_stability_on_v30_candidate"
BASE_TASK7_PATH = BASE_DIR / "openseek-7-v1.jsonl"
V30_TASK7_PATH = OUTPUTS_DIR / "final_submission_v30_task8clean2_on_v27_candidate" / "openseek-7-v1.jsonl"
AUDIT_PATH = WORK_LOGS_DIR / "task7_v34_pairwise_revert_audit_full_2026-04-09.json"

TASK7_SOURCE_VERSION = "authorproj_revert69_from_pairwise_audit"
TASK7_SOURCE_DIR = OUTPUTS_DIR / "task7_v35_authorproj_revert69_candidate"
TASK7_SOURCE_PATH = TASK7_SOURCE_DIR / "openseek-7-v1.jsonl"

OUT_DIR = OUTPUTS_DIR / "final_submission_v35_task7authorproj_revert69_on_v34_candidate"
ZIP_NAME = "final_submission_v35_task7authorproj_revert69_on_v34_candidate.zip"
REPORT_STEM = "task7_v35_authorproj_revert69_candidate_build_2026-04-09"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_rows(path: Path) -> dict[str, dict]:
    return {
        row["test_sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def write_jsonl_rows(path: Path, rows: dict[str, dict]) -> None:
    ordered_rows = sorted(rows.values(), key=lambda row: row["test_sample_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_task7_candidate() -> dict:
    audit = load_json(AUDIT_PATH)
    base_rows = load_jsonl_rows(BASE_TASK7_PATH)
    v30_rows = load_jsonl_rows(V30_TASK7_PATH)

    revert_rows = [row for row in audit["rows"] if row["pairwise"]["recommendation"] == "revert_to_v30"]
    revert_ids = {row["id"] for row in revert_rows}

    candidate_rows = {}
    changed_from_v34 = []
    changed_from_v30 = []

    for sample_id, base_row in base_rows.items():
        candidate_row = dict(base_row)
        if sample_id in revert_ids:
            candidate_row["prediction"] = v30_rows[sample_id]["prediction"]
        candidate_rows[sample_id] = candidate_row

        if candidate_row["prediction"] != base_row["prediction"]:
            changed_from_v34.append(
                {
                    "id": sample_id,
                    "old_prediction": base_row["prediction"],
                    "new_prediction": candidate_row["prediction"],
                }
            )
        if candidate_row["prediction"] != v30_rows[sample_id]["prediction"]:
            changed_from_v30.append(
                {
                    "id": sample_id,
                    "old_prediction": v30_rows[sample_id]["prediction"],
                    "new_prediction": candidate_row["prediction"],
                }
            )

    write_jsonl_rows(TASK7_SOURCE_PATH, candidate_rows)

    return {
        "revert_count_from_audit": len(revert_rows),
        "revert_ids": sorted(revert_ids),
        "task7_source_path": str(TASK7_SOURCE_PATH),
        "changed_count_vs_v34": len(changed_from_v34),
        "changed_rows_vs_v34": changed_from_v34,
        "changed_count_vs_v30": len(changed_from_v30),
        "changed_rows_vs_v30": changed_from_v30,
    }


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
        "patched_task7_source": "outputs/task7_v35_authorproj_revert69_candidate/openseek-7-v1.jsonl",
        "selection_rule": "Start from v34 task7 and revert only rows whose v30 answer won 2/2 pairwise against the v34 answer in the semantic_jump audit.",
        "revert_count_from_audit": task7_summary["revert_count_from_audit"],
        "changed_count_vs_v34": task7_summary["changed_count_vs_v34"],
        "changed_count_vs_v30": task7_summary["changed_count_vs_v30"],
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
            "7": "outputs/task7_v35_authorproj_revert69_candidate/openseek-7-v1.jsonl"
        },
        "notes": [
            "Only task7 was replaced relative to v34.",
            "The candidate keeps the v34 stability-gate line but reverts semantic_jump rows whose v30 answer won 2/2 pairwise against the v34 answer.",
            "This package is intended as a narrow post-v34 cleanup candidate rather than a new method-line expansion.",
        ],
        "task7_test_change_count_vs_v34": task7_summary["changed_count_vs_v34"],
        "task7_test_changes_vs_v34": task7_summary["changed_rows_vs_v34"],
        "task7_test_change_count_vs_v30": task7_summary["changed_count_vs_v30"],
        "task7_test_changes_vs_v30": task7_summary["changed_rows_vs_v30"],
    }
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_dir": str(OUT_DIR),
        "zip_path": str(zip_path),
    }


def write_markdown_report(report: dict) -> None:
    task7_summary = report["task7_summary"]
    package_summary = report["package_summary"]
    lines = [
        "# Task7 v35 Author Projection Revert69 Candidate Build",
        "",
        f"- Base: `{BASE_VERSION}`",
        f"- Task7 source: `{TASK7_SOURCE_VERSION}`",
        f"- Pairwise revert count: `{task7_summary['revert_count_from_audit']}`",
        f"- Changed task7 rows vs `{BASE_VERSION}`: `{task7_summary['changed_count_vs_v34']}`",
        f"- Changed task7 rows vs `v30`: `{task7_summary['changed_count_vs_v30']}`",
        f"- Package zip: `{package_summary['zip_path']}`",
        "",
        "## Candidate rationale",
        "",
        "- Keep v34 as the mainline and only revert rows where the old v30 answer beat the v34 answer 2/2 in pairwise judge comparisons.",
        "- Do not touch the known target rescue rows.",
        "- Use this as a narrow cleanup candidate for the next online comparison.",
        "",
    ]
    md_path = WORK_LOGS_DIR / f"{REPORT_STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    task7_summary = build_task7_candidate()
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
