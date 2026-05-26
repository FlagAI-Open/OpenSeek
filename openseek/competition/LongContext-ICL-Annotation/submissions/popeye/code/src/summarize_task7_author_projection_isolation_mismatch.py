import argparse
import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_REFRESH_SUMMARY_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_summary_2026-04-08.json"
DEFAULT_DIFF_AUDIT_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_diff_audit_2026-04-08.json"
DEFAULT_REPLAY_PATH = WORK_LOGS_DIR / "task7_author_projection_instrumented_isolation_replay_2026-04-09.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_isolation_mismatch_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_isolation_mismatch_2026-04-09.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_summary_path", type=str, default=str(DEFAULT_REFRESH_SUMMARY_PATH))
    parser.add_argument("--diff_audit_path", type=str, default=str(DEFAULT_DIFF_AUDIT_PATH))
    parser.add_argument("--replay_path", type=str, default=str(DEFAULT_REPLAY_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_replay_rows_by_id(replay_report: dict) -> dict[str, dict]:
    rows_by_id = {}
    for row in replay_report.get("rows", []):
        rows_by_id[row["test_sample_id"]] = {
            "off_prediction": row.get("off", {}).get("prediction"),
            "on_prediction": row.get("on", {}).get("prediction"),
            "changed_in_replay": bool(row.get("match_summary", {}).get("changed_in_replay")),
        }
    return rows_by_id


def classify_mismatch(
    *,
    full_run_off: str | None,
    full_run_on: str | None,
    trace_off: str | None,
    trace_on: str | None,
    isolation_on: str | None,
    changed_in_isolation: bool,
) -> str:
    full_run_off_matches_trace_off = full_run_off == trace_off
    full_run_on_matches_trace_on = full_run_on == trace_on
    isolation_reverted_to_trace_off = isolation_on == trace_off

    if not full_run_off_matches_trace_off and full_run_on_matches_trace_on and isolation_reverted_to_trace_off:
        return "baseline_mismatch_plus_isolation_revert"
    if full_run_off_matches_trace_off and full_run_on_matches_trace_on and isolation_reverted_to_trace_off and not changed_in_isolation:
        return "aligned_full_run_but_isolation_revert"
    if full_run_on_matches_trace_on and changed_in_isolation:
        return "trace_aligned_and_isolation_reproduced"
    if not full_run_on_matches_trace_on:
        return "full_run_trace_on_mismatch"
    return "other_target_mismatch"


def build_report(
    refresh_summary: dict,
    diff_audit: dict,
    replay_report: dict,
    *,
    refresh_summary_path: str,
    diff_audit_path: str,
    replay_path: str,
) -> dict:
    classified_by_id = {row["test_sample_id"]: row for row in refresh_summary["classified_rows"]}
    replay_by_id = build_replay_rows_by_id(replay_report)
    target_rows = [row for row in diff_audit["rows"] if row.get("is_target_bucket")]

    mismatch_rows = []
    for row in target_rows:
        sample_id = row["id"]
        trace_row = classified_by_id.get(sample_id)
        replay_row = replay_by_id.get(sample_id)

        full_run_off = row.get("off_prediction")
        full_run_on = row.get("on_prediction")
        trace_off = trace_row.get("off_prediction") if trace_row else None
        trace_on = trace_row.get("on_prediction") if trace_row else None
        isolation_off = replay_row.get("off_prediction") if replay_row else None
        isolation_on = replay_row.get("on_prediction") if replay_row else None
        changed_in_isolation = bool(replay_row and replay_row.get("changed_in_replay"))

        mismatch_rows.append(
            {
                "test_sample_id": sample_id,
                "category": row["category"],
                "clue": row["clue"],
                "bucket": row.get("bucket") or "none",
                "trace_classification": trace_row.get("classification") if trace_row else None,
                "full_run_off_prediction": full_run_off,
                "full_run_on_prediction": full_run_on,
                "trace_off_prediction": trace_off,
                "trace_on_prediction": trace_on,
                "isolation_off_prediction": isolation_off,
                "isolation_on_prediction": isolation_on,
                "changed_in_isolation": changed_in_isolation,
                "full_run_off_matches_trace_off": full_run_off == trace_off,
                "full_run_on_matches_trace_on": full_run_on == trace_on,
                "isolation_off_matches_trace_off": isolation_off == trace_off,
                "isolation_on_matches_trace_on": isolation_on == trace_on,
                "isolation_reverted_to_trace_off": isolation_on == trace_off,
                "mismatch_status": classify_mismatch(
                    full_run_off=full_run_off,
                    full_run_on=full_run_on,
                    trace_off=trace_off,
                    trace_on=trace_on,
                    isolation_on=isolation_on,
                    changed_in_isolation=changed_in_isolation,
                ),
            }
        )

    target_row_count = len(mismatch_rows)
    baseline_mismatch_count = sum(1 for row in mismatch_rows if not row["full_run_off_matches_trace_off"])
    full_run_trace_on_aligned_count = sum(1 for row in mismatch_rows if row["full_run_on_matches_trace_on"])
    isolation_revert_count = sum(1 for row in mismatch_rows if row["isolation_reverted_to_trace_off"])
    isolation_trace_on_match_count = sum(1 for row in mismatch_rows if row["isolation_on_matches_trace_on"])

    if target_row_count > 0 and isolation_revert_count == target_row_count:
        recommendation = "audit_isolation_replay_path"
        rationale = (
            "All current target rows collapse back to the trace-off answer in isolation, "
            "so the next debugging step should focus on isolation replay inputs and environment parity."
        )
    else:
        recommendation = "review_mixed_target_mismatch"
        rationale = "Target-row mismatch patterns are mixed and need case-by-case review."

    return {
        "generated_on": "2026-04-09",
        "inputs": {
            "refresh_summary_path": refresh_summary_path,
            "diff_audit_path": diff_audit_path,
            "replay_path": replay_path,
        },
        "summary": {
            "target_row_count": target_row_count,
            "baseline_mismatch_count": baseline_mismatch_count,
            "full_run_trace_on_aligned_count": full_run_trace_on_aligned_count,
            "isolation_revert_count": isolation_revert_count,
            "isolation_trace_on_match_count": isolation_trace_on_match_count,
            "baseline_mismatch_rate": safe_divide(baseline_mismatch_count, target_row_count),
            "full_run_trace_on_alignment_rate": safe_divide(full_run_trace_on_aligned_count, target_row_count),
            "isolation_revert_rate": safe_divide(isolation_revert_count, target_row_count),
            "isolation_trace_on_match_rate": safe_divide(isolation_trace_on_match_count, target_row_count),
            "recommendation": recommendation,
        },
        "rationale": rationale,
        "target_rows": mismatch_rows,
    }


def format_ratio(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Task7 Author Projection Isolation Mismatch Audit",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Recommendation: `{summary['recommendation']}`",
        "",
        "## Summary",
        "",
        f"- Target rows reviewed: `{summary['target_row_count']}`",
        f"- Full-run off vs trace off mismatches: `{summary['baseline_mismatch_count']}`",
        f"- Full-run on vs trace on alignments: `{summary['full_run_trace_on_aligned_count']}`",
        f"- Isolation reverts to trace-off: `{summary['isolation_revert_count']}`",
        f"- Isolation matches trace-on: `{summary['isolation_trace_on_match_count']}`",
        f"- Baseline mismatch rate: `{format_ratio(summary['baseline_mismatch_rate'])}`",
        f"- Full-run/trace on alignment rate: `{format_ratio(summary['full_run_trace_on_alignment_rate'])}`",
        f"- Isolation revert rate: `{format_ratio(summary['isolation_revert_rate'])}`",
        f"- Isolation trace-on match rate: `{format_ratio(summary['isolation_trace_on_match_rate'])}`",
        "",
        "## Rationale",
        "",
        f"- {report['rationale']}",
        "",
        "## Target row mismatch ledger",
        "",
    ]

    for row in report["target_rows"]:
        lines.extend(
            [
                f"### {row['test_sample_id']}",
                "",
                f"- Category: `{row['category']}`",
                f"- Bucket: `{row['bucket']}`",
                f"- Trace classification: `{row['trace_classification']}`",
                f"- Full-run off -> on: `{row['full_run_off_prediction']}` -> `{row['full_run_on_prediction']}`",
                f"- Trace off -> on: `{row['trace_off_prediction']}` -> `{row['trace_on_prediction']}`",
                f"- Isolation off -> on: `{row['isolation_off_prediction']}` -> `{row['isolation_on_prediction']}`",
                f"- Full-run off matches trace off: `{row['full_run_off_matches_trace_off']}`",
                f"- Full-run on matches trace on: `{row['full_run_on_matches_trace_on']}`",
                f"- Isolation off matches trace off: `{row['isolation_off_matches_trace_off']}`",
                f"- Isolation on matches trace on: `{row['isolation_on_matches_trace_on']}`",
                f"- Isolation reverted to trace off: `{row['isolation_reverted_to_trace_off']}`",
                f"- Changed in isolation: `{row['changed_in_isolation']}`",
                f"- Mismatch status: `{row['mismatch_status']}`",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    refresh_summary = load_json(args.refresh_summary_path)
    diff_audit = load_json(args.diff_audit_path)
    replay_report = load_json(args.replay_path)
    report = build_report(
        refresh_summary,
        diff_audit,
        replay_report,
        refresh_summary_path=args.refresh_summary_path,
        diff_audit_path=args.diff_audit_path,
        replay_path=args.replay_path,
    )
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
