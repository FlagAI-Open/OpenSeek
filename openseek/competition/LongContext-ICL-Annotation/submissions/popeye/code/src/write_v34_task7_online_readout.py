import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"
PHASE_SUMMARIES_DIR = PROJECT_DIR / "docs" / "phase-summaries"

PROMOTION_READOUT_PATH = WORK_LOGS_DIR / "task7_v34_authorproj_stability_candidate_promotion_readout_2026-04-09.json"
LEDGER_PATH = WORK_LOGS_DIR / "official_score_ledger_2026-04-09.json"
OUTPUT_JSON = WORK_LOGS_DIR / "task7_v34_online_readout_2026-04-09.json"
OUTPUT_MD = PHASE_SUMMARIES_DIR / "2026-04-09-v34-task7-online-readout.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    promotion = load_json(PROMOTION_READOUT_PATH)
    ledger = load_json(LEDGER_PATH)
    v34_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v34")
    v30_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v30")
    v27_entry = next(entry for entry in ledger["entries"] if entry["version"] == "v27")

    return {
        "generated_on": "2026-04-09",
        "version": "v34",
        "official_score": v34_entry["official_score"],
        "base_version": v34_entry["base_version"],
        "base_score": v30_entry["official_score"],
        "delta_vs_base": v34_entry["official_delta_vs_base"],
        "delta_vs_v27": round(v34_entry["official_score"] - v27_entry["official_score"], 2),
        "current_stable_official_baseline": ledger["stable_official_baseline"],
        "package_zip_path": promotion["package_zip_path"],
        "task7_changed_rows_vs_v30": promotion["changed_row_count_vs_v30"],
        "trace3_three_seed_all_on_match": promotion["trace3_three_seed_all_on_match"],
        "gate_rescue_row_88c5": promotion["gate_rescue_row_88c5"],
        "calibration_36rows": promotion["calibration_36rows"],
        "conclusion": "v34 is now the best official package and the current stable official baseline.",
        "leaderboard_note": "This readout records the official online score only. Leaderboard placement can still move as other teams submit.",
        "next_priority": "Use v34 as the new base and search for a low-risk narrow patch worth at least +0.12 to challenge the current #6 score band.",
    }


def render_markdown(report: dict) -> str:
    seed_2027 = report["gate_rescue_row_88c5"]["seed_2027"]
    seed_2028 = report["gate_rescue_row_88c5"]["seed_2028"]
    calibration = report["calibration_36rows"]
    lines = [
        "---",
        'project: "LongContext-ICL-Annotation"',
        'created_at: "2026-04-09 20:50:00 CST"',
        'phase: "v34-online-readout"',
        'status: "promoted"',
        "---",
        "",
        "# v34 task7 online readout",
        "",
        "## 1. Current Snapshot",
        f"- Current official result: Treat `v34 = {report['official_score']}` as the returned official online score on `2026-04-09`.",
        f"- Base comparison: `v34` was built on `v30 = {report['base_score']}` and gained `{report['delta_vs_base']}` online.",
        f"- Previous best comparison: `v34` beat `v27 = {round(report['official_score'] - report['delta_vs_v27'], 2)}` by `{report['delta_vs_v27']}`.",
        f"- Stable official baseline: `{report['current_stable_official_baseline']}`.",
        f"- Main conclusion: {report['conclusion']}",
        "",
        "## 2. Promotion Evidence",
        f"- Package zip: `{report['package_zip_path']}`",
        f"- Task7 changed rows vs `v30`: `{report['task7_changed_rows_vs_v30']}`",
        f"- Trace3 three-seed success: `{report['trace3_three_seed_all_on_match']}`",
        f"- 36-row calibration: stability gate `{calibration['stability_gate_changed_in_replay_count']}`, baseline `{calibration['baseline_changed_in_replay_count']}`, anchor1 `{calibration['anchor1_changed_in_replay_count']}`",
        f"- `88c5` seed `2027`: `{seed_2027['off_prediction']}` -> `{seed_2027['on_prediction']}` via `{seed_2027['final_decision_source']}`",
        f"- `88c5` seed `2028`: `{seed_2028['off_prediction']}` -> `{seed_2028['on_prediction']}` via `{seed_2028['final_decision_source']}`",
        "",
        "## 3. Operational Implications",
        "- The `author projection -> stability gate -> task7-only package` line is no longer research-only; it has now converted online.",
        "- Future narrow candidates should branch from `v34`, not `v27` or `v30`.",
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
