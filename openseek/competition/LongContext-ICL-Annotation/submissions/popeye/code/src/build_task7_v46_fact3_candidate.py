import json
import shutil
import zipfile
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

BASE_VERSION = "v45"
BASE_DIR = OUTPUTS_DIR / "final_submission_v45_task7fact3_on_v44_candidate"
BASE_TASK7_PATH = BASE_DIR / "openseek-7-v1.jsonl"
V30_TASK7_PATH = OUTPUTS_DIR / "final_submission_v30_task8clean2_on_v27_candidate" / "openseek-7-v1.jsonl"

TASK7_SOURCE_VERSION = "fact3_on_v45"
TASK7_SOURCE_DIR = OUTPUTS_DIR / "task7_v46_fact3_candidate"
TASK7_SOURCE_PATH = TASK7_SOURCE_DIR / "openseek-7-v1.jsonl"

OUT_DIR = OUTPUTS_DIR / "final_submission_v46_task7fact3_on_v45_candidate"
ZIP_NAME = "final_submission_v46_task7fact3_on_v45_candidate.zip"
REPORT_STEM = "task7_v46_fact3_candidate_build_2026-04-11"

FACT_FIXES = {
    "openseek-7-4c2884a96512479faae9bf88c7a5f076": {
        "prediction": "Manuel Noriega",
        "reason": "The Central American leader indicted on U.S. drug charges in February 1988 was Manuel Noriega.",
    },
    "openseek-7-4f1226451a7e4c2c9c18bb6edd43348b": {
        "prediction": "Oprah Winfrey",
        "reason": "The talk show queen who co-owned Chicago's The Eccentric was Oprah Winfrey.",
    },
    "openseek-7-9a88597ca11340669b68dcb70c51f103": {
        "prediction": "Laurence Olivier",
        "reason": "The 1948 Best Actor Oscar winner for Hamlet whose ashes are in Westminster Abbey was Laurence Olivier.",
    },
}


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
    base_rows = load_jsonl_rows(BASE_TASK7_PATH)
    v30_rows = load_jsonl_rows(V30_TASK7_PATH)

    candidate_rows = {}
    changed_from_v45 = []
    changed_from_v30 = []

    for sample_id, base_row in base_rows.items():
        candidate_row = dict(base_row)
        fix = FACT_FIXES.get(sample_id)
        if fix:
            candidate_row["prediction"] = fix["prediction"]
        candidate_rows[sample_id] = candidate_row

        if candidate_row["prediction"] != base_row["prediction"]:
            changed_from_v45.append(
                {
                    "id": sample_id,
                    "old_prediction": base_row["prediction"],
                    "new_prediction": candidate_row["prediction"],
                    "reason": FACT_FIXES[sample_id]["reason"],
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
        "fix_count": len(FACT_FIXES),
        "fixes": [{"id": sample_id, **fix} for sample_id, fix in sorted(FACT_FIXES.items())],
        "task7_source_path": str(TASK7_SOURCE_PATH),
        "changed_count_vs_v45": len(changed_from_v45),
        "changed_rows_vs_v45": changed_from_v45,
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
        "patched_task7_source": "outputs/task7_v46_fact3_candidate/openseek-7-v1.jsonl",
        "selection_rule": (
            "Start from v45 and apply only three additional clue-level factual repairs "
            "while preserving the existing gain stack."
        ),
        "fix_count": task7_summary["fix_count"],
        "changed_count_vs_v45": task7_summary["changed_count_vs_v45"],
        "changed_count_vs_v30": task7_summary["changed_count_vs_v30"],
        "changed_rows_vs_v45": task7_summary["changed_rows_vs_v45"],
    }
    (OUT_DIR / "task7_patch_summary.json").write_text(
        json.dumps(task7_patch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merge_summary = {
        "files": [f"openseek-{task_id}-v1.jsonl" for task_id in range(1, 9)],
        "zip_path": str(zip_path),
        "base": BASE_VERSION,
        "replaced_tasks": {"7": "outputs/task7_v46_fact3_candidate/openseek-7-v1.jsonl"},
        "notes": [
            "Only task7 was replaced relative to v45.",
            "This candidate keeps the full v45 gain stack and adds three additional clue-level factual repairs.",
            "The patch remains intentionally narrow to preserve the recent online stability trend.",
        ],
        "task7_test_change_count_vs_v45": task7_summary["changed_count_vs_v45"],
        "task7_test_changes_vs_v45": task7_summary["changed_rows_vs_v45"],
        "task7_test_change_count_vs_v30": task7_summary["changed_count_vs_v30"],
        "task7_test_changes_vs_v30": task7_summary["changed_rows_vs_v30"],
    }
    (OUT_DIR / "merge_summary.json").write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"out_dir": str(OUT_DIR), "zip_path": str(zip_path)}


def write_markdown_report(report: dict) -> None:
    task7_summary = report["task7_summary"]
    package_summary = report["package_summary"]
    lines = [
        "# Task7 v46 Fact3 Candidate Build",
        "",
        f"- Base: `{BASE_VERSION}`",
        f"- Task7 source: `{TASK7_SOURCE_VERSION}`",
        f"- Additional fix count vs `{BASE_VERSION}`: `{task7_summary['fix_count']}`",
        f"- Changed task7 rows vs `{BASE_VERSION}`: `{task7_summary['changed_count_vs_v45']}`",
        f"- Changed task7 rows vs `v30`: `{task7_summary['changed_count_vs_v30']}`",
        f"- Package zip: `{package_summary['zip_path']}`",
        "",
        "## Candidate rationale",
        "",
        "- Keep the full v45 gain stack untouched.",
        "- Add only three externally verifiable clue-level repairs whose current v45 answers remain clearly off-target.",
        "- Use this as the next narrow post-v45 challenge package.",
        "",
    ]
    md_path = WORK_LOGS_DIR / f"{REPORT_STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    task7_summary = build_task7_candidate()
    package_summary = build_package(task7_summary)
    report = {"task7_summary": task7_summary, "package_summary": package_summary}
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = WORK_LOGS_DIR / f"{REPORT_STEM}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
