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

TASK7_SOURCE_VERSION = "authorproj_stability_gate"
TASK7_SOURCE_DIR = OUTPUTS_DIR / "task7_author_projection_stability_live_candidate"
TASK7_SOURCE_PATH = TASK7_SOURCE_DIR / "openseek-7-v1.jsonl"

OUT_DIR = OUTPUTS_DIR / "final_submission_v34_task7authorproj_stability_on_v30_candidate"
ZIP_NAME = "final_submission_v34_task7authorproj_stability_on_v30_candidate.zip"
REPORT_STEM = "task7_v34_authorproj_stability_candidate_build_2026-04-09"
PROMOTION_READOUT_STEM = "task7_v34_authorproj_stability_candidate_promotion_readout_2026-04-09"
VALIDATION_SUMMARY_PATH = WORK_LOGS_DIR / "task7_author_projection_stability_gate_validation_summary_2026-04-09.json"
CALIBRATION_SUMMARY_PATH = (
    WORK_LOGS_DIR / "task7_author_projection_spillover_probe_refresh_default_seed2026_stability_gate_36rows_2026-04-09.json"
)


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

    return {
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
        "patched_task7_source": "outputs/task7_author_projection_stability_live_candidate/openseek-7-v1.jsonl",
        "selection_rule": "Apply author_primary_anchor_top1 only when the author bucket stays repeat-stable and the projected winner also wins 2/2 pairwise against the baseline winner.",
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
            "7": "outputs/task7_author_projection_stability_live_candidate/openseek-7-v1.jsonl"
        },
        "notes": [
            "Only task7 was replaced relative to v30.",
            "Task7 keeps the narrow author-projection path and adds an author-bucket-only stability gate before projected answers can overtake the baseline winner.",
            "The gate requires repeat stability on the frozen anchor1 judge list plus 2/2 pairwise wins against the baseline winner.",
            "This package is intended as the first score-seeking stability-gated Task7 candidate on top of the v30 base.",
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


def build_promotion_readout(report: dict) -> dict:
    validation_summary = json.loads(VALIDATION_SUMMARY_PATH.read_text(encoding="utf-8"))
    calibration_summary = json.loads(CALIBRATION_SUMMARY_PATH.read_text(encoding="utf-8"))

    trace3_rows = {
        row["test_sample_id"]: row["per_seed"]
        for row in validation_summary.get("trace3_rows", [])
    }
    rescue_row = trace3_rows.get("openseek-7-88c5e426f6b84e959de64d77e0862297", {})

    return {
        "generated_on": "2026-04-09",
        "candidate_version": "v34",
        "base_version": BASE_VERSION,
        "task7_source_version": TASK7_SOURCE_VERSION,
        "package_zip_path": report["package_summary"]["zip_path"],
        "changed_row_count_vs_v30": report["task7_summary"]["changed_count_vs_base"],
        "trace3_three_seed_all_on_match": validation_summary["summary"]["trace3_three_seed_all_on_match"],
        "known_target_rows": validation_summary.get("trace3_rows", []),
        "gate_rescue_row_88c5": {
            "test_sample_id": "openseek-7-88c5e426f6b84e959de64d77e0862297",
            "seed_2027": rescue_row.get("2027"),
            "seed_2028": rescue_row.get("2028"),
        },
        "calibration_36rows": {
            "row_count": calibration_summary["summary"]["row_count"],
            "stability_gate_changed_in_replay_count": calibration_summary["summary"]["changed_in_replay_count"],
            "baseline_changed_in_replay_count": validation_summary["summary"]["baseline_36row_changed_in_replay_count"],
            "anchor1_changed_in_replay_count": validation_summary["summary"]["anchor1_36row_changed_in_replay_count"],
        },
        "conclusion": "candidate ready for external scoring comparison",
        "scope_note": "Local-only promotion readout. This artifact does not claim the package is officially above 73.05.",
    }


def write_promotion_readout(report: dict) -> None:
    readout = build_promotion_readout(report)
    json_path = WORK_LOGS_DIR / f"{PROMOTION_READOUT_STEM}.json"
    md_path = WORK_LOGS_DIR / f"{PROMOTION_READOUT_STEM}.md"
    json_path.write_text(json.dumps(readout, ensure_ascii=False, indent=2), encoding="utf-8")

    seed_2027 = readout["gate_rescue_row_88c5"]["seed_2027"]
    seed_2028 = readout["gate_rescue_row_88c5"]["seed_2028"]
    calibration = readout["calibration_36rows"]
    lines = [
        "# Task7 v34 Stability Candidate Promotion Readout",
        "",
        f"- Candidate version: `v34` on `{BASE_VERSION}`",
        f"- Package zip: `{readout['package_zip_path']}`",
        f"- Changed Task7 rows vs `{BASE_VERSION}`: `{readout['changed_row_count_vs_v30']}`",
        f"- Trace3 three-seed success: `{readout['trace3_three_seed_all_on_match']}`",
        f"- 36-row calibration changed-in-replay: stability gate `{calibration['stability_gate_changed_in_replay_count']}`, baseline `{calibration['baseline_changed_in_replay_count']}`, anchor1 `{calibration['anchor1_changed_in_replay_count']}`",
        "",
        "## 88c5 gate rescue",
        "",
        f"- Seed `2027`: baseline `{seed_2027['off_prediction']}` -> final `{seed_2027['on_prediction']}` via `{seed_2027['final_decision_source']}`",
        f"- Seed `2028`: baseline `{seed_2028['off_prediction']}` -> final `{seed_2028['on_prediction']}` via `{seed_2028['final_decision_source']}`",
        "",
        "## Conclusion",
        "",
        f"- {readout['conclusion']}",
        f"- {readout['scope_note']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(report: dict) -> None:
    task7_summary = report["task7_summary"]
    package_summary = report["package_summary"]
    lines = [
        "# Task7 v34 Author Projection Stability Candidate Build",
        "",
        f"- Base: `{BASE_VERSION}`",
        f"- Task7 source: `{TASK7_SOURCE_VERSION}`",
        f"- Changed task7 test rows vs base: `{task7_summary['changed_count_vs_base']}`",
        f"- Package zip: `{package_summary['zip_path']}`",
        "",
        "## Candidate rationale",
        "",
        "- Keep the existing v30 package untouched except for task7.",
        "- Preserve the narrow author-projection gain path but require repeat stability and pairwise confirmation before projected winners overtake the baseline winner.",
        "- Build this as a score-seeking Task7-only replacement candidate on top of v30.",
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
    write_promotion_readout(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
