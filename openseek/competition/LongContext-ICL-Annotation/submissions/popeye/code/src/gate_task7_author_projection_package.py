import argparse
import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_REFRESH_SUMMARY_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_summary_2026-04-08.json"
DEFAULT_CLEAN_EVAL_PATH = WORK_LOGS_DIR / "task7_author_projection_clean_eval_2026-04-09.json"
DEFAULT_LEDGER_PATH = WORK_LOGS_DIR / "official_score_ledger_2026-04-01.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_package_gate_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_package_gate_2026-04-09.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_summary_path", type=str, default=str(DEFAULT_REFRESH_SUMMARY_PATH))
    parser.add_argument("--clean_eval_path", type=str, default=str(DEFAULT_CLEAN_EVAL_PATH))
    parser.add_argument("--ledger_path", type=str, default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_current_best_entry(ledger: dict) -> dict:
    return max(ledger["entries"], key=lambda entry: entry["official_score"])


def build_gate_report(
    refresh_summary: dict,
    clean_eval: dict,
    ledger: dict,
    *,
    refresh_summary_path: str,
    clean_eval_path: str,
    ledger_path: str,
) -> dict:
    best_entry = find_current_best_entry(ledger)
    summary = refresh_summary["summary"]
    clean_metrics = clean_eval["metrics"]

    gate_checks = {
        "stable_positive_hits_present": summary["stable_positive_count"] >= 2,
        "stable_trigger_signal_present": summary["stable_trigger_no_gain_count"] >= 1,
        "targeted_stable_signal_complete": clean_metrics["stable_signal_count"] == clean_metrics["traced_row_count"],
        "targeted_stable_signal_rate_high_enough": clean_metrics["stable_signal_rate"] >= 1.0,
        "targeted_positive_rate_high_enough": clean_metrics["stable_positive_rate_within_trace"] >= (2 / 3),
        "non_target_spillover_bounded": clean_metrics["full_run_non_target_share"] <= 0.25,
        "target_bucket_reproducibility_present": clean_metrics["target_bucket_reproducibility_rate"] > 0.0,
        "clean_eval_recommends_packaging": clean_eval["clean_eval_recommendation"] == "package_ready_with_clean_targeted_evidence",
        "has_above_baseline_package_evidence": clean_eval["clean_eval_recommendation"] == "package_ready_with_above_baseline_evidence",
    }

    failed_checks = [name for name, passed in gate_checks.items() if not passed]
    package_ready = all(
        gate_checks[name]
        for name in (
            "stable_positive_hits_present",
            "stable_trigger_signal_present",
            "targeted_stable_signal_complete",
            "targeted_stable_signal_rate_high_enough",
            "targeted_positive_rate_high_enough",
            "non_target_spillover_bounded",
            "target_bucket_reproducibility_present",
            "clean_eval_recommends_packaging",
        )
    )
    submit_ready = package_ready and gate_checks["has_above_baseline_package_evidence"]

    if submit_ready:
        gate_decision = "submit_candidate"
        rationale = "Current targeted evaluation clears both the package gate and the above-baseline submit gate."
    elif package_ready:
        gate_decision = "package_candidate"
        rationale = "Current targeted evaluation clears the package gate, but it still lacks explicit above-baseline package evidence for submit."
    else:
        repro_ready = gate_checks["target_bucket_reproducibility_present"]
        gate_decision = "hold_research_only"
        if repro_ready:
            rationale = (
                "Current targeted evaluation still looks research-only: stable local author-row gains exist and target-bucket reproducibility is now present, "
                "but package-level promotion is still blocked by heavy non-target spillover and the lack of above-baseline package evidence."
            )
        else:
            rationale = (
                "Current targeted evaluation still looks research-only: stable local author-row gains exist, "
                "but package-level promotion is blocked by spillover and missing target-bucket reproducibility."
            )

    return {
        "generated_on": "2026-04-09",
        "inputs": {
            "refresh_summary_path": refresh_summary_path,
            "clean_eval_path": clean_eval_path,
            "ledger_path": ledger_path,
        },
        "current_best_official": {
            "version": best_entry["version"],
            "score": best_entry["official_score"],
            "base_version": ledger["stable_official_baseline"],
        },
        "refresh_snapshot": {
            "frontier_status": summary["frontier_status"],
            "packaging_recommendation": summary["packaging_recommendation"],
            "full_run_changed_count": summary["full_run_changed_count"],
            "full_run_target_bucket_changed_count": summary["full_run_target_bucket_changed_count"],
            "full_run_non_target_changed_count": summary["full_run_non_target_changed_count"],
            "isolation_changed_count": summary["isolation_changed_count"],
            "isolation_target_bucket_changed_count": summary["isolation_target_bucket_changed_count"],
            "stable_positive_count": summary["stable_positive_count"],
            "stable_trigger_no_gain_count": summary["stable_trigger_no_gain_count"],
        },
        "clean_eval_snapshot": {
            "recommendation": clean_eval["clean_eval_recommendation"],
            "next_gate_intent": clean_eval["next_gate_intent"],
            "stable_signal_rate": clean_metrics["stable_signal_rate"],
            "stable_positive_rate_within_trace": clean_metrics["stable_positive_rate_within_trace"],
            "full_run_non_target_share": clean_metrics["full_run_non_target_share"],
            "target_bucket_reproducibility_rate": clean_metrics["target_bucket_reproducibility_rate"],
            "target_row_count": clean_metrics["target_row_count"],
            "target_trace_supported_count": clean_metrics["target_trace_supported_count"],
            "target_trace_isolation_disagreement_count": clean_metrics["target_trace_isolation_disagreement_count"],
        },
        "gate_checks": gate_checks,
        "failed_checks": failed_checks,
        "gate_decision": gate_decision,
        "package_ready": package_ready,
        "submit_ready": submit_ready,
        "rationale": rationale,
        "next_action": (
            "Do not package yet. Keep using the clean evaluation summary as the primary promotion signal, and only revisit packaging after spillover is bounded and above-baseline package evidence appears."
            if not package_ready
            else "Build a narrow Task7 candidate package and review it against the current official best."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Author Projection Package Gate",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Current best official version: `{report['current_best_official']['version']}`",
        f"- Current best official score: `{report['current_best_official']['score']}`",
        f"- Gate decision: `{report['gate_decision']}`",
        f"- Package ready: `{report['package_ready']}`",
        f"- Submit ready: `{report['submit_ready']}`",
        "",
        "## Refresh snapshot",
        "",
        f"- Frontier status: `{report['refresh_snapshot']['frontier_status']}`",
        f"- Packaging recommendation: `{report['refresh_snapshot']['packaging_recommendation']}`",
        f"- Full-run changed rows: `{report['refresh_snapshot']['full_run_changed_count']}`",
        f"- Full-run target-bucket changed rows: `{report['refresh_snapshot']['full_run_target_bucket_changed_count']}`",
        f"- Full-run non-target changed rows: `{report['refresh_snapshot']['full_run_non_target_changed_count']}`",
        f"- Isolation changed rows: `{report['refresh_snapshot']['isolation_changed_count']}`",
        f"- Isolation target-bucket changed rows: `{report['refresh_snapshot']['isolation_target_bucket_changed_count']}`",
        f"- Stable positive hits: `{report['refresh_snapshot']['stable_positive_count']}`",
        f"- Stable trigger-no-gain hits: `{report['refresh_snapshot']['stable_trigger_no_gain_count']}`",
        "",
        "## Clean evaluation snapshot",
        "",
        f"- Recommendation: `{report['clean_eval_snapshot']['recommendation']}`",
        f"- Next gate intent: `{report['clean_eval_snapshot']['next_gate_intent']}`",
        f"- Stable signal rate: `{report['clean_eval_snapshot']['stable_signal_rate']}`",
        f"- Stable positive rate within trace: `{report['clean_eval_snapshot']['stable_positive_rate_within_trace']}`",
        f"- Full-run non-target share: `{report['clean_eval_snapshot']['full_run_non_target_share']}`",
        f"- Target-bucket reproducibility rate: `{report['clean_eval_snapshot']['target_bucket_reproducibility_rate']}`",
        f"- Target rows in ledger: `{report['clean_eval_snapshot']['target_row_count']}`",
        f"- Target rows with trace support: `{report['clean_eval_snapshot']['target_trace_supported_count']}`",
        f"- Trace/isolation disagreements: `{report['clean_eval_snapshot']['target_trace_isolation_disagreement_count']}`",
        "",
        "## Gate checks",
        "",
    ]

    for name, passed in report["gate_checks"].items():
        lines.append(f"- `{name}`: `{passed}`")

    lines.extend(
        [
            "",
            "## Decision rationale",
            "",
            f"- {report['rationale']}",
            "",
            "## Next action",
            "",
            f"- {report['next_action']}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    refresh_summary = load_json(args.refresh_summary_path)
    clean_eval = load_json(args.clean_eval_path)
    ledger = load_json(args.ledger_path)
    report = build_gate_report(
        refresh_summary,
        clean_eval,
        ledger,
        refresh_summary_path=args.refresh_summary_path,
        clean_eval_path=args.clean_eval_path,
        ledger_path=args.ledger_path,
    )
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
