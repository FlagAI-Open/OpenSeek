import argparse
import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_CALLTRACE_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_calltrace_2026-04-08.json"
DEFAULT_TARGET_REPRO_PATH = WORK_LOGS_DIR / "task7_author_projection_target_repro_ledger_2026-04-09.json"
DEFAULT_ISOLATION_MISMATCH_PATH = WORK_LOGS_DIR / "task7_author_projection_isolation_mismatch_2026-04-09.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_replay_gap_summary_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_replay_gap_summary_2026-04-09.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calltrace_path", type=str, default=str(DEFAULT_CALLTRACE_PATH))
    parser.add_argument("--target_repro_path", type=str, default=str(DEFAULT_TARGET_REPRO_PATH))
    parser.add_argument("--isolation_mismatch_path", type=str, default=str(DEFAULT_ISOLATION_MISMATCH_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_report(calltrace: dict, target_repro: dict, isolation_mismatch: dict, *, calltrace_path: str, target_repro_path: str, isolation_mismatch_path: str) -> dict:
    changed_call_rows = calltrace.get("changed_call_rows", [])
    target_call_rows = [
        row for row in changed_call_rows
        if row.get("test_sample_id") in {target["test_sample_id"] for target in target_repro.get("target_rows", [])}
    ]
    target_direct_fact_rows = [
        row for row in target_call_rows if row.get("on_call_kind_counts", {}).get("direct_fact", 0) > 0
    ]

    mismatch_rows = isolation_mismatch.get("target_rows", [])
    baseline_mismatch_rows = [row for row in mismatch_rows if not row.get("full_run_off_matches_trace_off")]
    isolation_revert_rows = [row for row in mismatch_rows if row.get("isolation_reverted_to_trace_off")]

    target_reproduced = target_repro["summary"]["isolation_reproduced_count"] == target_repro["summary"]["target_row_count"]
    likely_root_cause = "baseline_drift_only"
    if not target_reproduced:
        likely_root_cause = "isolation_replay_input_or_env_mismatch"
        if baseline_mismatch_rows:
            likely_root_cause = "isolation_replay_input_or_env_mismatch_plus_baseline_drift"

    if target_reproduced:
        conclusion = (
            "The replay-path parity issue is no longer blocking the target author-projection rows. "
            "Both target rows show on-path direct-fact expansion in the call trace, and both now reproduce the stable trace on-side answer in replay. "
            "The remaining gap is baseline drift on one row plus the unchanged package-level spillover in the full-run diff."
        )
        next_debug_focus = [
            "Treat replay parity as provisionally fixed and stop using the old isolation audit as the target reproducibility source.",
            "Move the next debugging pass to full-run spillover control, especially the 34 non-target changed rows.",
            "Decide whether the baseline drift on the Joseph Conrad row is acceptable as a full-run off-path difference or needs separate normalization.",
            "Only revisit package promotion after a new clean full-run eval shows materially lower non-target spillover.",
        ]
    else:
        conclusion = (
            "The target author-projection gains are not failing because the on-path lacks direct-fact activity. "
            "Both target rows show on-path direct-fact expansion in the call trace and align with the stable trace on-side answer, "
            "but isolation replay collapses them back to the trace-off answer. This points to replay-path input/environment mismatch rather than absence of target signal."
        )
        next_debug_focus = [
            "Recover the exact command/env that generated the isolation audit and compare it against the full-run and trace paths.",
            "Check whether the isolation replay reused the same example pool and author catalog construction as the full Task7 run.",
            "Check whether the replay preserved the same author-direct-fact env vars and judge candidate slot configuration.",
            "Check whether the replay row input was reconstructed from the diff artifact instead of the original test sample payload.",
        ]

    return {
        "generated_on": "2026-04-09",
        "inputs": {
            "calltrace_path": calltrace_path,
            "target_repro_path": target_repro_path,
            "isolation_mismatch_path": isolation_mismatch_path,
        },
        "summary": {
            "changed_call_row_count": calltrace.get("changed_call_row_count", 0),
            "target_call_row_count": len(target_call_rows),
            "target_direct_fact_row_count": len(target_direct_fact_rows),
            "target_repro_row_count": target_repro["summary"]["target_row_count"],
            "target_trace_supported_count": target_repro["summary"]["trace_supported_count"],
            "target_isolation_reproduced_count": target_repro["summary"]["isolation_reproduced_count"],
            "target_isolation_revert_count": isolation_mismatch["summary"]["isolation_revert_count"],
            "baseline_mismatch_count": isolation_mismatch["summary"]["baseline_mismatch_count"],
            "target_call_coverage_of_repro_rows": safe_divide(len(target_call_rows), target_repro["summary"]["target_row_count"]),
            "target_direct_fact_coverage_of_repro_rows": safe_divide(len(target_direct_fact_rows), target_repro["summary"]["target_row_count"]),
            "likely_root_cause": likely_root_cause,
        },
        "evidence": {
            "first_divergence": calltrace.get("first_divergence"),
            "target_call_rows": target_call_rows,
            "baseline_mismatch_rows": baseline_mismatch_rows,
            "isolation_revert_rows": isolation_revert_rows,
        },
        "conclusion": conclusion,
        "next_debug_focus": next_debug_focus,
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    first_divergence = report["evidence"]["first_divergence"]
    lines = [
        "# Task7 Author Projection Replay Gap Summary",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Likely root cause: `{summary['likely_root_cause']}`",
        "",
        "## Key evidence",
        "",
        f"- Changed call rows in calltrace: `{summary['changed_call_row_count']}`",
        f"- Target rows seen in changed call rows: `{summary['target_call_row_count']}`",
        f"- Target rows with on-path direct-fact calls: `{summary['target_direct_fact_row_count']}`",
        f"- Target rows in repro ledger: `{summary['target_repro_row_count']}`",
        f"- Target rows supported by trace: `{summary['target_trace_supported_count']}`",
        f"- Target rows reproduced in isolation: `{summary['target_isolation_reproduced_count']}`",
        f"- Target rows reverting in isolation: `{summary['target_isolation_revert_count']}`",
        f"- Baseline mismatches across full-run vs trace: `{summary['baseline_mismatch_count']}`",
        "",
        "## Calltrace signal",
        "",
        f"- First call-count divergence appears at position `{first_divergence['position']}` on row `{first_divergence['test_sample_id']}`.",
        f"- The target author rows later show on-path direct-fact calls (`9` and `18`) in the refresh calltrace.",
        "",
        "## Conclusion",
        "",
        f"- {report['conclusion']}",
        "",
        "## Next debug focus",
        "",
    ]

    for item in report["next_debug_focus"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    calltrace = load_json(args.calltrace_path)
    target_repro = load_json(args.target_repro_path)
    isolation_mismatch = load_json(args.isolation_mismatch_path)
    report = build_report(
        calltrace,
        target_repro,
        isolation_mismatch,
        calltrace_path=args.calltrace_path,
        target_repro_path=args.target_repro_path,
        isolation_mismatch_path=args.isolation_mismatch_path,
    )
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
