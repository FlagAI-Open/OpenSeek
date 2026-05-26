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
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_target_repro_ledger_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_target_repro_ledger_2026-04-09.md"


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


def classify_repro_status(*, trace_classification: str | None, changed_in_isolation: bool) -> str:
    if trace_classification == "stable_positive_correction" and changed_in_isolation:
        return "stable_positive_and_isolation_reproduced"
    if trace_classification == "stable_positive_correction" and not changed_in_isolation:
        return "stable_positive_but_not_isolation_reproduced"
    if trace_classification == "stable_trigger_no_gain" and changed_in_isolation:
        return "stable_trigger_and_isolation_reproduced"
    if trace_classification == "stable_trigger_no_gain" and not changed_in_isolation:
        return "stable_trigger_but_not_isolation_reproduced"
    if changed_in_isolation:
        return "isolation_reproduced_without_trace_support"
    return "no_reproducible_target_signal"


def build_report(
    refresh_summary: dict,
    diff_audit: dict,
    replay_report: dict,
    *,
    refresh_summary_path: str,
    diff_audit_path: str,
    replay_path: str,
) -> dict:
    classified_by_id = {
        row["test_sample_id"]: row for row in refresh_summary["classified_rows"]
    }
    replay_by_id = build_replay_rows_by_id(replay_report)
    target_rows = [row for row in diff_audit["rows"] if row.get("is_target_bucket")]

    ledger_rows = []
    for row in target_rows:
        sample_id = row["id"]
        trace_row = classified_by_id.get(sample_id)
        replay_row = replay_by_id.get(sample_id)
        changed_in_isolation = bool(replay_row and replay_row.get("changed_in_replay"))
        trace_classification = trace_row["classification"] if trace_row else None

        ledger_rows.append(
            {
                "test_sample_id": sample_id,
                "category": row["category"],
                "clue": row["clue"],
                "bucket": row.get("bucket") or "none",
                "full_run_off_prediction": row["off_prediction"],
                "full_run_on_prediction": row["on_prediction"],
                "full_run_target_changed": True,
                "trace_classification": trace_classification,
                "trace_off_prediction": trace_row["off_prediction"] if trace_row else None,
                "trace_on_prediction": trace_row["on_prediction"] if trace_row else None,
                "trace_projected_candidates": trace_row["projected_candidates"] if trace_row else [],
                "trace_has_stable_signal": bool(trace_row and trace_classification in {"stable_positive_correction", "stable_trigger_no_gain"}),
                "trace_has_stable_positive": bool(trace_row and trace_classification == "stable_positive_correction"),
                "isolation_changed_in_isolation": changed_in_isolation,
                "isolation_off_prediction": replay_row.get("off_prediction") if replay_row else None,
                "isolation_on_prediction": replay_row.get("on_prediction") if replay_row else None,
                "repro_status": classify_repro_status(
                    trace_classification=trace_classification,
                    changed_in_isolation=changed_in_isolation,
                ),
            }
        )

    target_row_count = len(ledger_rows)
    trace_supported_count = sum(1 for row in ledger_rows if row["trace_has_stable_signal"])
    trace_positive_count = sum(1 for row in ledger_rows if row["trace_has_stable_positive"])
    isolation_reproduced_count = sum(1 for row in ledger_rows if row["isolation_changed_in_isolation"])
    disagreement_count = sum(
        1
        for row in ledger_rows
        if row["trace_has_stable_signal"] and not row["isolation_changed_in_isolation"]
    )

    if target_row_count > 0 and trace_supported_count == target_row_count and isolation_reproduced_count == 0:
        recommendation = "trace_supported_but_not_isolation_reproduced"
        rationale = (
            "Every current target-bucket full-run delta is supported by stable trace evidence, "
            "but none of them reproduce in the isolation audit yet."
        )
    elif target_row_count > 0 and trace_supported_count == target_row_count and isolation_reproduced_count == target_row_count:
        recommendation = "target_reproducibility_ready"
        rationale = "Target-bucket deltas are supported by trace evidence and they reproduce in isolation."
    else:
        recommendation = "target_reproducibility_mixed"
        rationale = "Target-bucket evidence is mixed across trace support and isolation reproduction."

    return {
        "generated_on": "2026-04-09",
        "inputs": {
            "refresh_summary_path": refresh_summary_path,
            "diff_audit_path": diff_audit_path,
            "replay_path": replay_path,
        },
        "summary": {
            "target_row_count": target_row_count,
            "trace_supported_count": trace_supported_count,
            "trace_positive_count": trace_positive_count,
            "isolation_reproduced_count": isolation_reproduced_count,
            "trace_support_rate": safe_divide(trace_supported_count, target_row_count),
            "trace_positive_rate": safe_divide(trace_positive_count, target_row_count),
            "isolation_reproduction_rate": safe_divide(isolation_reproduced_count, target_row_count),
            "trace_isolation_disagreement_count": disagreement_count,
            "recommendation": recommendation,
        },
        "rationale": rationale,
        "target_rows": ledger_rows,
    }


def format_ratio(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Task7 Author Projection Target Reproducibility Ledger",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Recommendation: `{summary['recommendation']}`",
        "",
        "## Summary",
        "",
        f"- Target rows in full-run diff: `{summary['target_row_count']}`",
        f"- Target rows supported by stable trace: `{summary['trace_supported_count']}`",
        f"- Target rows with stable positive trace: `{summary['trace_positive_count']}`",
        f"- Target rows reproduced in isolation: `{summary['isolation_reproduced_count']}`",
        f"- Trace support rate: `{format_ratio(summary['trace_support_rate'])}`",
        f"- Trace positive rate: `{format_ratio(summary['trace_positive_rate'])}`",
        f"- Isolation reproduction rate: `{format_ratio(summary['isolation_reproduction_rate'])}`",
        f"- Trace/isolation disagreement count: `{summary['trace_isolation_disagreement_count']}`",
        "",
        "## Rationale",
        "",
        f"- {report['rationale']}",
        "",
        "## Target row ledger",
        "",
    ]

    for row in report["target_rows"]:
        lines.extend(
            [
                f"### {row['test_sample_id']}",
                "",
                f"- Category: `{row['category']}`",
                f"- Bucket: `{row['bucket']}`",
                f"- Full-run off -> on: `{row['full_run_off_prediction']}` -> `{row['full_run_on_prediction']}`",
                f"- Trace classification: `{row['trace_classification']}`",
                f"- Trace off -> on: `{row['trace_off_prediction']}` -> `{row['trace_on_prediction']}`",
                f"- Trace projected candidates: `{', '.join(row['trace_projected_candidates']) if row['trace_projected_candidates'] else '(empty)'}`",
                f"- Isolation changed: `{row['isolation_changed_in_isolation']}`",
                f"- Isolation off -> on: `{row['isolation_off_prediction']}` -> `{row['isolation_on_prediction']}`",
                f"- Repro status: `{row['repro_status']}`",
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
