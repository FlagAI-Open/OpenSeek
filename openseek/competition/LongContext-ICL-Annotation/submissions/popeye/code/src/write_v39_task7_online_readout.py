import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
PHASE_SUMMARIES_DIR = PROJECT_DIR / "docs" / "phase-summaries"

BUILD_REPORT_PATH = WORK_LOGS_DIR / "task7_v39_typofix3_candidate_build_2026-04-10.json"
LEDGER_PATH = WORK_LOGS_DIR / "official_score_ledger_2026-04-09.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v39_online_readout_2026-04-10.json"
OUTPUT_MD = PHASE_SUMMARIES_DIR / "2026-04-10-v39-task7-online-readout.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    build_report = load_json(BUILD_REPORT_PATH)
    ledger = load_json(LEDGER_PATH)
    v39_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v39")
    v37_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v37")
    v27_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v27")

    return {
        "generated_on": "2026-04-10",
        "version": "v39",
        "official_score": v39_entry["official_score"],
        "base_version": v39_entry["base_version"],
        "base_score": v37_entry["official_score"],
        "delta_vs_base": v39_entry["official_delta_vs_base"],
        "delta_vs_v27": round(v39_entry["official_score"] - v27_entry["official_score"], 2),
        "current_stable_official_baseline": ledger["stable_official_baseline"],
        "package_zip_path": build_report["package_summary"]["zip_path"],
        "manual_fix_count": build_report["task7_summary"]["fix_count"],
        "changed_rows_vs_v37": build_report["task7_summary"]["changed_count_vs_v37"],
        "changed_rows_vs_v30": build_report["task7_summary"]["changed_count_vs_v30"],
        "conclusion": "v39 matched the new best official score at 73.78 and becomes the stable official baseline.",
        "leaderboard_note": "This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.",
        "next_priority": "Stack the tied 73.78 gains by combining the v38 and v39 Task7 micro-patches into one next challenge package.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "---",
        'project: "LongContext-ICL-Annotation"',
        'created_at: "2026-04-10 02:15:00 CST"',
        'phase: "v39-online-readout"',
        'status: "promoted"',
        "---",
        "",
        "# v39 task7 online readout",
        "",
        "## 1. Current Snapshot",
        f"- Current official result: Treat `v39 = {report['official_score']}` as the returned official online score on `2026-04-10`.",
        f"- Base comparison: `v39` was built on `v37 = {report['base_score']}` and gained `{report['delta_vs_base']}` online.",
        f"- Previous stable comparison: `v39` beat `v27 = {round(report['official_score'] - report['delta_vs_v27'], 2)}` by `{report['delta_vs_v27']}`.",
        f"- Stable official baseline: `{report['current_stable_official_baseline']}`.",
        f"- Main conclusion: {report['conclusion']}",
        "",
        "## 2. Promotion Evidence",
        f"- Package zip: `{report['package_zip_path']}`",
        f"- Manual typo/canonical fix count: `{report['manual_fix_count']}`",
        f"- Changed Task7 rows vs `v37`: `{report['changed_rows_vs_v37']}`",
        f"- Changed Task7 rows vs `v30`: `{report['changed_rows_vs_v30']}`",
        "",
        "## 3. Operational Implications",
        "- The typo/canonical cleanup line is now officially calibrated online, not just locally plausible.",
        "- Because it ties the best score with a narrower patch surface, it becomes the default stable branch point.",
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
