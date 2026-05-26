import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
PHASE_SUMMARIES_DIR = PROJECT_DIR / "docs" / "phase-summaries"

BUILD_REPORT_PATH = WORK_LOGS_DIR / "task7_v36_manualfix6_candidate_build_2026-04-09.json"
LEDGER_PATH = WORK_LOGS_DIR / "official_score_ledger_2026-04-09.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v36_online_readout_2026-04-09.json"
OUTPUT_MD = PHASE_SUMMARIES_DIR / "2026-04-09-v36-task7-online-readout.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    build_report = load_json(BUILD_REPORT_PATH)
    ledger = load_json(LEDGER_PATH)
    v36_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v36")
    v35_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v35")
    v27_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v27")

    return {
        "generated_on": "2026-04-09",
        "version": "v36",
        "official_score": v36_entry["official_score"],
        "base_version": v36_entry["base_version"],
        "base_score": v35_entry["official_score"],
        "delta_vs_base": v36_entry["official_delta_vs_base"],
        "delta_vs_v27": round(v36_entry["official_score"] - v27_entry["official_score"], 2),
        "current_stable_official_baseline": ledger["stable_official_baseline"],
        "package_zip_path": build_report["package_summary"]["zip_path"],
        "manual_fix_count": build_report["task7_summary"]["fix_count"],
        "changed_rows_vs_v35": build_report["task7_summary"]["changed_count_vs_v35"],
        "changed_rows_vs_v30": build_report["task7_summary"]["changed_count_vs_v30"],
        "conclusion": "v36 is now the best official package and the current stable official baseline.",
        "leaderboard_note": "This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.",
        "next_priority": "Branch from v36 and search for another ultra-narrow Task7 factual patch, starting from the unresolved shortlist rather than reopening broad judge changes.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "---",
        'project: "LongContext-ICL-Annotation"',
        'created_at: "2026-04-10 00:05:00 CST"',
        'phase: "v36-online-readout"',
        'status: "promoted"',
        "---",
        "",
        "# v36 task7 online readout",
        "",
        "## 1. Current Snapshot",
        f"- Current official result: Treat `v36 = {report['official_score']}` as the returned official online score on `2026-04-09`.",
        f"- Base comparison: `v36` was built on `v35 = {report['base_score']}` and gained `{report['delta_vs_base']}` online.",
        f"- Previous stable comparison: `v36` beat `v27 = {round(report['official_score'] - report['delta_vs_v27'], 2)}` by `{report['delta_vs_v27']}`.",
        f"- Stable official baseline: `{report['current_stable_official_baseline']}`.",
        f"- Main conclusion: {report['conclusion']}",
        "",
        "## 2. Promotion Evidence",
        f"- Package zip: `{report['package_zip_path']}`",
        f"- Manual factual fix count: `{report['manual_fix_count']}`",
        f"- Changed Task7 rows vs `v35`: `{report['changed_rows_vs_v35']}`",
        f"- Changed Task7 rows vs `v30`: `{report['changed_rows_vs_v30']}`",
        "",
        "## 3. Operational Implications",
        "- The best official baseline is now the post-v35 manual-fix line, not only the revert69 cleanup line.",
        "- Future Task7 candidates should branch from `v36` and preserve the known target rescues plus the six clue-level fixes unless a replacement is directly validated.",
        f"- {report['leaderboard_note']}",
        "",
        "## 4. Recommended Next Step",
        f"- {report['next_priority']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    PHASE_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
