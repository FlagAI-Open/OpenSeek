import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

PATCH_NOTES_PATH = WORK_LOGS_DIR / "task7_v39_verified_patch_notes_2026-04-10.json"
BUILD_REPORT_PATH = WORK_LOGS_DIR / "task7_v39_typofix3_candidate_build_2026-04-10.json"
SMOKE_REPORT_PATH = WORK_LOGS_DIR / "task7_v39_candidate_smoke_2026-04-10.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v39_candidate_promotion_readout_2026-04-10.json"
OUTPUT_MD = WORK_LOGS_DIR / "task7_v39_candidate_promotion_readout_2026-04-10.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    patch_notes = load_json(PATCH_NOTES_PATH)
    build_report = load_json(BUILD_REPORT_PATH)
    smoke_report = load_json(SMOKE_REPORT_PATH)
    return {
        "generated_on": "2026-04-10",
        "base_version": "v37",
        "candidate_version": "v39",
        "package_zip_path": build_report["package_summary"]["zip_path"],
        "fix_count": build_report["task7_summary"]["fix_count"],
        "changed_rows_vs_v37": build_report["task7_summary"]["changed_count_vs_v37"],
        "changed_rows_vs_v30": build_report["task7_summary"]["changed_count_vs_v30"],
        "verified_patch_ids": [row["id"] for row in patch_notes["verified_rows"]],
        "watchlist_ids": [row["id"] for row in patch_notes["watchlist_rows"]],
        "smoke_ready_for_external_scoring": smoke_report["ready_for_external_scoring"],
        "local_conclusion": (
            "v39 is ready for external scoring as a local-only ultra-narrow typo/canonical "
            "cleanup challenge package on top of v37."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 v39 Candidate Promotion Readout",
        "",
        f"- Base version: `{report['base_version']}`",
        f"- Candidate version: `{report['candidate_version']}`",
        f"- Package zip: `{report['package_zip_path']}`",
        f"- Fix count: `{report['fix_count']}`",
        f"- Changed rows vs `v37`: `{report['changed_rows_vs_v37']}`",
        f"- Changed rows vs `v30`: `{report['changed_rows_vs_v30']}`",
        f"- Verified patch ids: `{report['verified_patch_ids']}`",
        f"- Watchlist ids: `{report['watchlist_ids']}`",
        f"- Smoke ready for external scoring: `{report['smoke_ready_for_external_scoring']}`",
        "",
        "## Conclusion",
        "",
        f"- {report['local_conclusion']}",
        "- This readout is local-only and does not claim any official score until an external evaluation returns.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
