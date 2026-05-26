import argparse
import json
from collections import Counter
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_TRACE_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_author_row_trace_2026-04-08.json"
DEFAULT_DIFF_AUDIT_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_diff_audit_2026-04-08.json"
DEFAULT_ISOLATION_AUDIT_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_isolation_audit_2026-04-08.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_refresh_summary_2026-04-08.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_refresh_summary_2026-04-08.md"
DEFAULT_STABILITY_NOTE_MD = WORK_LOGS_DIR / "task7_author_projection_refresh_stability_note_2026-04-08.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", type=str, default=str(DEFAULT_TRACE_PATH))
    parser.add_argument("--diff_audit_path", type=str, default=str(DEFAULT_DIFF_AUDIT_PATH))
    parser.add_argument("--isolation_audit_path", type=str, default=str(DEFAULT_ISOLATION_AUDIT_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--stability_note_path", type=str, default=str(DEFAULT_STABILITY_NOTE_MD))
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classify_trace_row(row: dict) -> dict:
    off = row.get("off", {})
    on = row.get("on", {})
    off_prediction = off.get("prediction")
    on_prediction = on.get("prediction")
    projected_candidates = on.get("author_projected_candidates") or []
    judge_candidates = on.get("judge_candidates")
    judge_raw = on.get("judge_raw")
    judge_index = on.get("judge_index")

    if projected_candidates and off_prediction != on_prediction and on_prediction in projected_candidates:
        classification = "stable_positive_correction"
    elif projected_candidates and off_prediction == on_prediction:
        classification = "stable_trigger_no_gain"
    else:
        classification = "no_stable_author_effect"

    return {
        "test_sample_id": row["test_sample_id"],
        "category": row["category"],
        "clue": row["clue"],
        "classification": classification,
        "off_prediction": off_prediction,
        "on_prediction": on_prediction,
        "projected_candidates": projected_candidates,
        "on_judge_candidates": judge_candidates,
        "on_judge_raw": judge_raw,
        "on_judge_index": judge_index,
        "off_judge_raw": off.get("judge_raw"),
        "off_judge_index": off.get("judge_index"),
    }


def build_report(
    trace_report: dict,
    diff_audit: dict,
    isolation_audit: dict,
    *,
    trace_path: str,
    diff_audit_path: str,
    isolation_audit_path: str,
) -> dict:
    classified_rows = [classify_trace_row(row) for row in trace_report.get("rows", [])]
    class_counts = Counter(row["classification"] for row in classified_rows)

    summary = {
        "full_run_changed_count": diff_audit["changed_count"],
        "full_run_target_bucket_changed_count": diff_audit["target_bucket_changed_count"],
        "full_run_non_target_changed_count": diff_audit["non_target_changed_count"],
        "isolation_changed_count": isolation_audit["changed_in_isolation_count"],
        "isolation_target_bucket_changed_count": isolation_audit["target_bucket_changed_count"],
        "isolation_non_target_changed_count": isolation_audit["non_target_changed_count"],
        "stable_positive_count": class_counts["stable_positive_correction"],
        "stable_trigger_no_gain_count": class_counts["stable_trigger_no_gain"],
        "no_stable_author_effect_count": class_counts["no_stable_author_effect"],
        "primary_evidence": "stable_author_row_trace",
        "packaging_recommendation": "not_package_ready_from_full_run_diff",
        "frontier_status": "method_valid_but_full_run_diff_noisy",
    }

    return {
        "generated_on": trace_report.get("generated_on") or diff_audit.get("generated_on") or isolation_audit.get("generated_on"),
        "inputs": {
            "trace_path": trace_path,
            "diff_audit_path": diff_audit_path,
            "isolation_audit_path": isolation_audit_path,
        },
        "summary": summary,
        "classified_rows": classified_rows,
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    rows = report["classified_rows"]
    stable_positive_rows = [row for row in rows if row["classification"] == "stable_positive_correction"]
    stable_trigger_rows = [row for row in rows if row["classification"] == "stable_trigger_no_gain"]
    no_effect_rows = [row for row in rows if row["classification"] == "no_stable_author_effect"]

    lines = [
        "# Task7 Author Projection Refresh Summary",
        "",
        f"- Generated on: `{report['generated_on']}`",
        "- Goal: separate stable author-bucket evidence from noisy full-run off/on diff counts",
        "",
        "## Key conclusion",
        "",
        "- The Task7 author-projection frontier remains method-valid.",
        "- The raw full-run `off -> on` diff is not a safe proxy for feature effect.",
        "- The defensible signal is the stable author-row trace, not aggregate changed-row count.",
        "",
        "## Full-run contamination summary",
        "",
        f"- Full-run changed rows: `{summary['full_run_changed_count']}`",
        f"- Full-run target-bucket changed rows: `{summary['full_run_target_bucket_changed_count']}`",
        f"- Full-run non-target changed rows: `{summary['full_run_non_target_changed_count']}`",
        f"- Isolation changed rows: `{summary['isolation_changed_count']}`",
        f"- Isolation target-bucket changed rows: `{summary['isolation_target_bucket_changed_count']}`",
        f"- Isolation non-target changed rows: `{summary['isolation_non_target_changed_count']}`",
        "",
        "## Stable author-bucket hits",
        "",
        f"- Stable positive corrections: `{summary['stable_positive_count']}`",
        f"- Stable trigger without gain: `{summary['stable_trigger_no_gain_count']}`",
        f"- No stable author effect: `{summary['no_stable_author_effect_count']}`",
        "",
    ]

    for index, row in enumerate(stable_positive_rows, start=1):
        lines.extend(
            [
                f"### Stable positive hit {index}",
                "",
                f"- Row: `{row['test_sample_id']}`",
                f"- Category: `{row['category']}`",
                f"- Clue: `{row['clue']}`",
                f"- Off prediction: `{row['off_prediction']}`",
                f"- On author-projected candidates: `{', '.join(row['projected_candidates']) if row['projected_candidates'] else '(empty)'}`",
                f"- On judge candidates: `{', '.join(row['on_judge_candidates']) if row['on_judge_candidates'] else '(empty)'}`",
                f"- On judge winner: `{row['on_prediction']}`",
                f"- Judge raw/index: `{row['on_judge_raw']}` / `{row['on_judge_index']}`",
                f"- Classification: `{row['classification']}`",
                "",
            ]
        )

    for row in stable_trigger_rows:
        lines.extend(
            [
                "### Stable trigger without gain",
                "",
                f"- Row: `{row['test_sample_id']}`",
                f"- Category: `{row['category']}`",
                f"- Clue: `{row['clue']}`",
                f"- Off prediction: `{row['off_prediction']}`",
                f"- On author-projected candidates: `{', '.join(row['projected_candidates']) if row['projected_candidates'] else '(empty)'}`",
                f"- On judge candidates: `{', '.join(row['on_judge_candidates']) if row['on_judge_candidates'] else '(empty)'}`",
                f"- On judge winner: `{row['on_prediction']}`",
                f"- Judge raw/index: `{row['on_judge_raw']}` / `{row['on_judge_index']}`",
                f"- Classification: `{row['classification']}`",
                "",
            ]
        )

    if no_effect_rows:
        lines.extend([
            "## Traced rows without stable author effect",
            "",
        ])
        for row in no_effect_rows:
            lines.extend(
                [
                    f"- `{row['test_sample_id']}`: off `{row['off_prediction']}` -> on `{row['on_prediction']}`, projected `{', '.join(row['projected_candidates']) if row['projected_candidates'] else '(empty)'}`",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Packaging implication",
            "",
            f"- Recommendation: `{summary['packaging_recommendation']}`",
            "- Keep the author-projection line active.",
            "- Do not use the raw full-run diff as candidate-package evidence.",
            "- Revisit packaging only after a cleaner evaluation method separates true author-bucket gains from request-order drift.",
            "",
        ]
    )

    return "\n".join(lines)


def render_stability_note(report: dict) -> str:
    summary = report["summary"]
    rows = report["classified_rows"]
    stable_positive_rows = [row for row in rows if row["classification"] == "stable_positive_correction"]
    stable_trigger_rows = [row for row in rows if row["classification"] == "stable_trigger_no_gain"]

    lines = [
        "# Task7 Author Projection Refresh Stability Note",
        "",
        f"- Generated on: `{report['generated_on']}`",
        "- Scope: current-mainline Task7 author projection refresh",
        "- Goal: separate stable author-bucket gains from polluted full-run off/on diffs before any packaging decision",
        "",
        "## Key conclusion",
        "",
        "- The Task7 author-projection frontier remains valid.",
        "- The raw full-run `off -> on` diff is **not** a safe proxy for feature effect.",
        "- The defensible evidence is now the stable per-row author-bucket trace, not the aggregate changed-row count.",
        "",
        "## Full-run refresh summary",
        "",
        f"- Fresh current-mainline full Task7 rebuilds (`off` vs `on`) changed `{summary['full_run_changed_count']}` / `500` rows.",
        f"- Only `{summary['full_run_target_bucket_changed_count']}` of those `{summary['full_run_changed_count']}` changes were in the intended `quoted_work_author_relation` bucket.",
        f"- `{summary['full_run_non_target_changed_count']}` / `{summary['full_run_changed_count']}` were non-target rows.",
        "",
        "Artifacts:",
        "- `outputs/task7_author_projection_refresh_seed2026_off/openseek-7-v1.jsonl`",
        "- `outputs/task7_author_projection_refresh_seed2026_on/openseek-7-v1.jsonl`",
        "- `outputs/work_logs/task7_author_projection_refresh_seed2026_diff_audit_2026-04-08.json`",
        "- `outputs/work_logs/task7_author_projection_refresh_seed2026_diff_audit_2026-04-08.md`",
        "",
        "## Isolation replay summary",
        "",
        f"- Replayed all `{summary['full_run_changed_count']}` changed rows individually with the same Task7 env.",
        f"- Only `{summary['isolation_changed_count']}` rows still changed in isolation.",
        f"- All `{summary['isolation_non_target_changed_count']}` isolated changes were non-target rows.",
        f"- The `{summary['full_run_target_bucket_changed_count']}` target-bucket rows from the full-run diff did not reproduce as stable isolated deltas.",
        "",
        "Artifact:",
        "- `outputs/work_logs/task7_author_projection_refresh_seed2026_isolation_audit_2026-04-08.json`",
        "",
        "Interpretation:",
        "- The full-run off/on diff is contaminated by request-order / seed-sequence effects.",
        f"- The aggregate `{summary['full_run_changed_count']}-row` diff should not be used as packaging evidence.",
        "",
        "## Stable author-bucket traces",
        "",
        "The reliable signal comes from direct row-level traces of known author-bucket examples/test-preview rows.",
        "",
        "Artifact:",
        "- `outputs/work_logs/task7_author_projection_refresh_seed2026_author_row_trace_2026-04-08.json`",
        "",
    ]

    for index, row in enumerate(stable_positive_rows, start=1):
        lines.extend(
            [
                f"### Stable positive hit {index}",
                "",
                f"- Row: `{row['test_sample_id']}`",
                f"- Category: `{row['category']}`",
                f"- Clue: `{row['clue']}`",
                f"- Off prediction: `{row['off_prediction']}`",
                f"- On author-projected candidates: `{', '.join(row['projected_candidates']) if row['projected_candidates'] else '(empty)'}`",
                f"- On judge winner: `{row['on_prediction']}`",
                "- Outcome: stable positive correction",
                "",
            ]
        )

    for row in stable_trigger_rows:
        lines.extend(
            [
                "### Stable trigger without gain",
                "",
                f"- Row: `{row['test_sample_id']}`",
                f"- Category: `{row['category']}`",
                f"- Clue: `{row['clue']}`",
                f"- Off prediction: `{row['off_prediction']}`",
                f"- On author-projected candidates: `{', '.join(row['projected_candidates']) if row['projected_candidates'] else '(empty)'}`",
                f"- On judge winner: `{row['on_prediction']}`",
                "- Outcome: stable author-projection trigger but no net gain",
                "",
            ]
        )

    lines.extend(
        [
            "## Packaging implication",
            "",
            "- Do **not** treat the current-mainline full-run diff as a candidate-package justification.",
            "- Do **not** refresh a Task7-only candidate package from current mainline based on this refresh alone.",
            "- Treat the frontier as:",
            "  - method-valid,",
            "  - locally judge-positive on author rows,",
            "  - globally evaluation-noisy under fixed-seed full-run comparison.",
            "",
            "## Recommended next step",
            "",
            "- Keep the author-projection line active.",
            "- Use targeted author-bucket validation / trace evidence as the primary decision signal.",
            "- Only revisit packaging after establishing a cleaner evaluation method that separates true author-bucket gains from request-order drift.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    trace_report = load_json(args.trace_path)
    diff_audit = load_json(args.diff_audit_path)
    isolation_audit = load_json(args.isolation_audit_path)
    report = build_report(
        trace_report,
        diff_audit,
        isolation_audit,
        trace_path=args.trace_path,
        diff_audit_path=args.diff_audit_path,
        isolation_audit_path=args.isolation_audit_path,
    )
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report) + "\n", encoding="utf-8")
    Path(args.stability_note_path).write_text(render_stability_note(report) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
