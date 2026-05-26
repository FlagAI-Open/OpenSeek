import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

PACKAGE_DIR = OUTPUTS_DIR / "final_submission_v41_task7singapore1_on_v40_candidate"
TASK7_PATH = PACKAGE_DIR / "openseek-7-v1.jsonl"
MERGE_SUMMARY_PATH = PACKAGE_DIR / "merge_summary.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v41_candidate_smoke_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v41_candidate_smoke_2026-04-10.md"

EXPECTED_ROWS = {
    "openseek-7-3deb283e13ec4b5393b1f1d61b426a4e": "Joseph Conrad",
    "openseek-7-88c5e426f6b84e959de64d77e0862297": "John Dos Passos",
    "openseek-7-02e748b0da7f4492864df5ab8b804556": "jack london",
    "openseek-7-220c54dc2fa44178a8b1e881e9ee6fb9": "selenium",
    "openseek-7-63f7eeaee4f4454ea81d62622f2d773a": "petrograd",
    "openseek-7-7b7c13a6f01b4f96997a82f3725258d6": "Victoria",
    "openseek-7-34bd86c0bffd49ce86d2e8267b023d92": "Louisa",
    "openseek-7-8da173db613a4fa895adc022789f5631": "turban",
    "openseek-7-bc0b7dbb83eb4abcaa447bfc6e13e29c": "sodden",
    "openseek-7-01e4fbfca16345b898064b73d9c18659": "Singapore",
}


def load_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report() -> dict:
    rows = load_jsonl_rows(TASK7_PATH)
    merge_summary = json.loads(MERGE_SUMMARY_PATH.read_text(encoding="utf-8"))
    ids = [row["test_sample_id"] for row in rows]
    predictions = {row["test_sample_id"]: row["prediction"] for row in rows}
    return {
        "generated_on": "2026-04-10",
        "package_dir": str(PACKAGE_DIR),
        "zip_path": str(PACKAGE_DIR / "final_submission_v41_task7singapore1_on_v40_candidate.zip"),
        "line_count": len(rows),
        "unique_id_count": len(set(ids)),
        "duplicate_ids": sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1}),
        "spot_checks": {
            sample_id: {
                "expected": expected,
                "actual": predictions.get(sample_id),
                "matched": predictions.get(sample_id) == expected,
            }
            for sample_id, expected in EXPECTED_ROWS.items()
        },
        "merge_summary": {
            "base": merge_summary.get("base"),
            "replaced_tasks": merge_summary.get("replaced_tasks"),
            "task7_test_change_count_vs_v40": merge_summary.get("task7_test_change_count_vs_v40"),
        },
        "ready_for_external_scoring": (
            len(rows) == 500
            and len(set(ids)) == 500
            and all(predictions.get(sample_id) == expected for sample_id, expected in EXPECTED_ROWS.items())
            and merge_summary.get("base") == "v40"
            and merge_summary.get("replaced_tasks")
            == {"7": "outputs/task7_v41_singapore1_candidate/openseek-7-v1.jsonl"}
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v41 Candidate Smoke",
        "",
        f"- Package dir: `{report['package_dir']}`",
        f"- Zip path: `{report['zip_path']}`",
        f"- Task7 line count: `{report['line_count']}`",
        f"- Task7 unique id count: `{report['unique_id_count']}`",
        f"- Ready for external scoring: `{report['ready_for_external_scoring']}`",
        "",
        "## Spot checks",
        "",
    ]
    for sample_id, check in report["spot_checks"].items():
        lines.append(
            f"- `{sample_id}`: expected `{check['expected']}` | actual `{check['actual']}` | matched `{check['matched']}`"
        )
    lines.extend(
        [
            "",
            "## Package checks",
            "",
            f"- merge base: `{report['merge_summary']['base']}`",
            f"- replaced tasks: `{report['merge_summary']['replaced_tasks']}`",
            f"- changed rows vs v40: `{report['merge_summary']['task7_test_change_count_vs_v40']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
