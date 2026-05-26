import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

ASSET_PATHS = {
    "triggered_bundle": WORK_LOGS_DIR / "task7_v26_triggered_diagnostics_2026-03-30.json",
    "mismatch_audit": WORK_LOGS_DIR / "task7_v26_online_mismatch_diagnosis_2026-03-30.json",
    "order_bias_probe": WORK_LOGS_DIR / "task7_v26_order_bias_probe_2026-03-30.json",
    "pairwise_probe": WORK_LOGS_DIR / "task7_v26_pairwise_probe_2026-03-30.json",
    "targeted_topk_diagnosis": WORK_LOGS_DIR / "task7_targeted_topk_diagnosis_2026-03-31.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    triggered = load_json(ASSET_PATHS["triggered_bundle"])
    mismatch = load_json(ASSET_PATHS["mismatch_audit"])
    order_bias = load_json(ASSET_PATHS["order_bias_probe"])
    pairwise = load_json(ASSET_PATHS["pairwise_probe"])
    targeted = load_json(ASSET_PATHS["targeted_topk_diagnosis"])

    retained_rows = []
    for row in targeted["rows"]:
        retained_rows.append(
            {
                "id": row["id"],
                "gold": row["gold"],
                "gold_visible_in_candidates": row["gold_visible_in_candidates"],
                "secondary_only_candidates": row["secondary_only_candidates"],
                "baseline_visible": row["baseline_visible"],
            }
        )

    gold_missing_count = sum(not row["gold_visible_in_candidates"] for row in retained_rows)

    report = {
        "generated_on": "2026-03-31",
        "asset_manifest": {
            name: str(path)
            for name, path in ASSET_PATHS.items()
        },
        "freeze_scope": {
            "task7_submission_state": "frozen_as_diagnosis_only",
            "allow_new_promotable_packages": False,
            "allow_new_typed_narrow_variants": False,
            "allow_new_pairwise_triggered_variants": False,
            "allow_new_judge_amplification_variants": False,
        },
        "coverage_check": {
            "triggered_row_count": triggered["triggered_row_count"],
            "changed_test_row_count": mismatch["coverage"]["changed_row_count"],
            "all_changed_rows_accounted_for": mismatch["coverage"]["all_changed_rows_accounted_for"],
            "retained_target_count": len(retained_rows),
            "gold_missing_on_retained_targets": gold_missing_count,
        },
        "diagnostic_findings": {
            "mismatch_conclusion": mismatch["conclusion"],
            "triggered_gold_presence_bucket_counts": triggered["gold_presence_bucket_counts"],
            "triggered_selection_risk_marker_counts": triggered["selection_risk_marker_counts"],
            "order_bias": {
                "row_count": order_bias["row_count"],
                "top_choice_flip_count": order_bias["top_choice_flip_count"],
                "disagreement_rate": order_bias["disagreement_rate"],
                "flips_involving_secondary_only_count": order_bias["flips_involving_secondary_only_count"],
            },
            "pairwise": {
                "evaluated_row_count": pairwise["evaluated_row_count"],
                "total_comparison_count": pairwise["total_comparison_count"],
                "secondary_strong_win_count": pairwise["secondary_strong_win_count"],
                "secondary_split_win_count": pairwise["secondary_split_win_count"],
                "secondary_strong_win_rate": pairwise["secondary_strong_win_rate"],
                "avg_secondary_preference_rate": pairwise["avg_secondary_preference_rate"],
            },
            "retained_targets": retained_rows,
        },
        "current_research_question": (
            "How do we raise gold-in-candidate-set on retained targets, instead of spending more cycles on rerank thresholds, ordering, or takeover rules?"
        ),
        "reopen_conditions": [
            "Retained targets show a clear increase in gold_visible_in_candidates.",
            "Conversion improves in a way that is not explained only by order sensitivity or weak pairwise positives.",
            "The new mechanism is stronger than another narrow rerank threshold/package tweak.",
        ],
        "final_status": {
            "task7_line": "diagnosis_first",
            "bottleneck": "candidate_generation_visibility",
            "do_not_return_to": [
                "typed_narrow_package_promotion",
                "pairwise_triggered_package_promotion",
                "threshold_micro_tuning_as_primary_line",
            ],
        },
    }
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Post-v27 Freeze",
        "",
        "- Task7 is frozen as a diagnosis-only line.",
        "- No new typed narrow, pairwise-triggered, or judge-amplification package should be promoted from this state.",
        "",
        "## Coverage",
        "",
        f"- Triggered rows captured: `{report['coverage_check']['triggered_row_count']}`",
        f"- Changed test rows accounted for: `{report['coverage_check']['changed_test_row_count']}`",
        f"- Retained targets reviewed: `{report['coverage_check']['retained_target_count']}`",
        f"- Retained targets with gold missing from the candidate set: `{report['coverage_check']['gold_missing_on_retained_targets']}`",
        "",
        "## Key Findings",
        "",
        f"- Mismatch conclusion: `{report['diagnostic_findings']['mismatch_conclusion']['recommended_direction']}`",
        f"- Order-bias disagreement rate: `{report['diagnostic_findings']['order_bias']['disagreement_rate']}`",
        f"- Pairwise secondary strong-win rate: `{report['diagnostic_findings']['pairwise']['secondary_strong_win_rate']}`",
        f"- Current research question: {report['current_research_question']}",
        "",
        "## Retained Targets",
        "",
    ]

    for row in report["diagnostic_findings"]["retained_targets"]:
        lines.append(
            f"- `{row['id']}` gold=`{row['gold']}` visible_in_candidates=`{row['gold_visible_in_candidates']}`"
        )

    lines.extend(["", "## Reopen Conditions", ""])
    for item in report["reopen_conditions"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    json_path = WORK_LOGS_DIR / "task7_post_v27_freeze_2026-03-31.json"
    md_path = WORK_LOGS_DIR / "task7_post_v27_freeze_2026-03-31.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
