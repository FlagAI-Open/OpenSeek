import argparse
import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_REFRESH_SUMMARY_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_summary_2026-04-08.json"
DEFAULT_TARGET_REPRO_PATH = WORK_LOGS_DIR / "task7_author_projection_target_repro_ledger_2026-04-09.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_clean_eval_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_clean_eval_2026-04-09.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_summary_path", type=str, default=str(DEFAULT_REFRESH_SUMMARY_PATH))
    parser.add_argument("--target_repro_path", type=str, default=str(DEFAULT_TARGET_REPRO_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def classify_clean_eval(summary: dict, classified_rows: list[dict], target_repro: dict) -> tuple[str, str]:
    stable_positive_count = summary["stable_positive_count"]
    stable_trigger_no_gain_count = summary["stable_trigger_no_gain_count"]
    full_run_target_bucket_changed_count = summary["full_run_target_bucket_changed_count"]
    full_run_non_target_changed_count = summary["full_run_non_target_changed_count"]
    target_repro_summary = target_repro["summary"]

    traced_row_count = len(classified_rows)
    stable_signal_count = stable_positive_count + stable_trigger_no_gain_count

    if (
        traced_row_count > 0
        and stable_positive_count >= 2
        and stable_signal_count == traced_row_count
        and full_run_target_bucket_changed_count > 0
        and full_run_non_target_changed_count <= full_run_target_bucket_changed_count
        and target_repro_summary["isolation_reproduced_count"] == target_repro_summary["target_row_count"]
    ):
        return (
            "package_ready_with_clean_targeted_evidence",
            "Targeted author-bucket evidence is stable, reproduced across target rows, and the package-level spillover is narrow enough to justify packaging.",
        )

    if target_repro_summary["isolation_reproduced_count"] == target_repro_summary["target_row_count"]:
        return (
            "targeted_signal_present_but_package_evidence_insufficient",
            "Targeted author-bucket gains are stable and now reproduce across the target rows, but package-level evidence is still insufficient because the full-run diff is dominated by non-target drift.",
        )

    return (
        "targeted_signal_present_but_package_evidence_insufficient",
        "Targeted author-bucket gains are real, but package-level evidence is still insufficient because the full-run diff is dominated by non-target drift and the target bucket ledger still does not reproduce in isolation.",
    )


def build_report(refresh_summary: dict, target_repro: dict, *, refresh_summary_path: str, target_repro_path: str) -> dict:
    summary = refresh_summary["summary"]
    classified_rows = refresh_summary["classified_rows"]
    target_repro_summary = target_repro["summary"]

    traced_row_count = len(classified_rows)
    stable_positive_count = summary["stable_positive_count"]
    stable_trigger_no_gain_count = summary["stable_trigger_no_gain_count"]
    no_stable_author_effect_count = summary["no_stable_author_effect_count"]
    stable_signal_count = stable_positive_count + stable_trigger_no_gain_count

    metrics = {
        "traced_row_count": traced_row_count,
        "stable_positive_count": stable_positive_count,
        "stable_trigger_no_gain_count": stable_trigger_no_gain_count,
        "no_stable_author_effect_count": no_stable_author_effect_count,
        "stable_signal_count": stable_signal_count,
        "stable_signal_rate": safe_divide(stable_signal_count, traced_row_count),
        "stable_positive_rate_within_trace": safe_divide(stable_positive_count, traced_row_count),
        "full_run_changed_count": summary["full_run_changed_count"],
        "full_run_target_bucket_changed_count": summary["full_run_target_bucket_changed_count"],
        "full_run_non_target_changed_count": summary["full_run_non_target_changed_count"],
        "full_run_target_precision": safe_divide(
            summary["full_run_target_bucket_changed_count"], summary["full_run_changed_count"]
        ),
        "full_run_non_target_share": safe_divide(
            summary["full_run_non_target_changed_count"], summary["full_run_changed_count"]
        ),
        "isolation_changed_count": summary["isolation_changed_count"],
        "isolation_target_bucket_changed_count": summary["isolation_target_bucket_changed_count"],
        "isolation_non_target_changed_count": summary["isolation_non_target_changed_count"],
        "target_bucket_reproducibility_rate": target_repro_summary["isolation_reproduction_rate"],
        "target_row_count": target_repro_summary["target_row_count"],
        "target_trace_supported_count": target_repro_summary["trace_supported_count"],
        "target_trace_isolation_disagreement_count": target_repro_summary["trace_isolation_disagreement_count"],
    }

    row_ledger = []
    for row in classified_rows:
        row_ledger.append(
            {
                "test_sample_id": row["test_sample_id"],
                "category": row["category"],
                "clue": row["clue"],
                "classification": row["classification"],
                "targeted_trigger_present": bool(row["projected_candidates"]),
                "prediction_changed": row["off_prediction"] != row["on_prediction"],
                "stable_gain": row["classification"] == "stable_positive_correction",
                "off_prediction": row["off_prediction"],
                "on_prediction": row["on_prediction"],
                "projected_candidates": row["projected_candidates"],
                "on_judge_candidates": row["on_judge_candidates"],
                "on_judge_raw": row["on_judge_raw"],
                "on_judge_index": row["on_judge_index"],
            }
        )

    recommendation, rationale = classify_clean_eval(summary, classified_rows, target_repro)

    return {
        "generated_on": "2026-04-09",
        "inputs": {
            "refresh_summary_path": refresh_summary_path,
            "target_repro_path": target_repro_path,
        },
        "evaluation_scope": {
            "primary_signal": "stable_author_row_trace",
            "target_repro_signal": "target_bucket_reproducibility_ledger",
            "package_signal": "targeted_author_bucket_evidence",
            "contamination_context": "full_run_and_isolation_diff_audits",
        },
        "metrics": metrics,
        "row_ledger": row_ledger,
        "target_repro_snapshot": target_repro_summary,
        "clean_eval_recommendation": recommendation,
        "clean_eval_rationale": rationale,
        "next_gate_intent": (
            "promote_to_package_gate"
            if recommendation == "package_ready_with_clean_targeted_evidence"
            else "hold_research_only"
        ),
    }


def format_ratio(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    target_repro_summary = report["target_repro_snapshot"]
    lines = [
        "# Task7 Author Projection Clean Evaluation",
        "",
        f"- Generated on: `{report['generated_on']}`",
        "- Goal: promote Task7 author-projection using targeted evidence instead of noisy full-run changed-row counts",
        f"- Recommendation: `{report['clean_eval_recommendation']}`",
        f"- Next gate intent: `{report['next_gate_intent']}`",
        "",
        "## Targeted evidence summary",
        "",
        f"- Traced author rows: `{metrics['traced_row_count']}`",
        f"- Stable positives: `{metrics['stable_positive_count']}`",
        f"- Stable trigger-no-gain rows: `{metrics['stable_trigger_no_gain_count']}`",
        f"- No stable author effect rows: `{metrics['no_stable_author_effect_count']}`",
        f"- Stable signal rate: `{format_ratio(metrics['stable_signal_rate'])}`",
        f"- Stable positive rate within trace: `{format_ratio(metrics['stable_positive_rate_within_trace'])}`",
        "",
        "## Target reproducibility summary",
        "",
        f"- Target rows in full-run diff: `{target_repro_summary['target_row_count']}`",
        f"- Target rows supported by stable trace: `{target_repro_summary['trace_supported_count']}`",
        f"- Target rows reproduced in isolation: `{target_repro_summary['isolation_reproduced_count']}`",
        f"- Target-bucket reproducibility rate: `{format_ratio(target_repro_summary['isolation_reproduction_rate'])}`",
        f"- Trace/isolation disagreement count: `{target_repro_summary['trace_isolation_disagreement_count']}`",
        "",
        "## Package contamination summary",
        "",
        f"- Full-run changed rows: `{metrics['full_run_changed_count']}`",
        f"- Full-run target-bucket changed rows: `{metrics['full_run_target_bucket_changed_count']}`",
        f"- Full-run non-target changed rows: `{metrics['full_run_non_target_changed_count']}`",
        f"- Full-run target precision: `{format_ratio(metrics['full_run_target_precision'])}`",
        f"- Full-run non-target share: `{format_ratio(metrics['full_run_non_target_share'])}`",
        f"- Isolation changed rows: `{metrics['isolation_changed_count']}`",
        f"- Isolation target-bucket changed rows: `{metrics['isolation_target_bucket_changed_count']}`",
        "",
        "## Row ledger",
        "",
    ]

    for row in report["row_ledger"]:
        lines.extend(
            [
                f"### {row['test_sample_id']}",
                "",
                f"- Category: `{row['category']}`",
                f"- Classification: `{row['classification']}`",
                f"- Targeted trigger present: `{row['targeted_trigger_present']}`",
                f"- Prediction changed: `{row['prediction_changed']}`",
                f"- Stable gain: `{row['stable_gain']}`",
                f"- Off prediction: `{row['off_prediction']}`",
                f"- On prediction: `{row['on_prediction']}`",
                f"- Projected candidates: `{', '.join(row['projected_candidates']) if row['projected_candidates'] else '(empty)'}`",
                f"- On judge candidates: `{', '.join(row['on_judge_candidates']) if row['on_judge_candidates'] else '(empty)'}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Rationale",
            "",
            f"- {report['clean_eval_rationale']}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    refresh_summary = load_json(args.refresh_summary_path)
    target_repro = load_json(args.target_repro_path)
    report = build_report(
        refresh_summary,
        target_repro,
        refresh_summary_path=args.refresh_summary_path,
        target_repro_path=args.target_repro_path,
    )
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
