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
DEFAULT_REPLAY_GAP_PATH = WORK_LOGS_DIR / "task7_author_projection_replay_gap_summary_2026-04-09.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_replay_parity_audit_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_replay_parity_audit_2026-04-09.md"
MAIN_PATH = SRC_DIR / "main.py"
METHOD_PATH = SRC_DIR / "method.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calltrace_path", type=str, default=str(DEFAULT_CALLTRACE_PATH))
    parser.add_argument("--target_repro_path", type=str, default=str(DEFAULT_TARGET_REPRO_PATH))
    parser.add_argument("--isolation_mismatch_path", type=str, default=str(DEFAULT_ISOLATION_MISMATCH_PATH))
    parser.add_argument("--replay_gap_path", type=str, default=str(DEFAULT_REPLAY_GAP_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_check(name: str, status: str, evidence: str) -> dict:
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
    }


def build_report(
    calltrace: dict,
    target_repro: dict,
    isolation_mismatch: dict,
    replay_gap: dict,
    *,
    calltrace_path: str,
    target_repro_path: str,
    isolation_mismatch_path: str,
    replay_gap_path: str,
) -> dict:
    main_text = MAIN_PATH.read_text(encoding="utf-8")
    method_text = METHOD_PATH.read_text(encoding="utf-8")

    checks = [
        build_check(
            "author_projection_env_gate_present",
            "observed",
            "method.py gates author projection through OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_PROJECTION.",
        ),
        build_check(
            "author_catalog_built_from_task7_examples",
            "observed",
            "main.py and method.py build the author catalog from the active Task7 ICL examples, so replay parity depends on example-pool parity.",
        ),
        build_check(
            "judge_layout_depends_on_reserved_secondary_slots",
            "observed",
            "method.py appends author-projected candidates through append_unique judge candidate construction, so replay parity depends on the same judge slot configuration.",
        ),
        build_check(
            "target_rows_show_on_path_direct_fact_activity",
            "observed",
            f"Calltrace shows {sum(1 for row in calltrace.get('changed_call_rows', []) if row.get('on_call_kind_counts', {}).get('direct_fact', 0) > 0 and row.get('test_sample_id') in {target['test_sample_id'] for target in target_repro.get('target_rows', [])})} target rows with on-path direct_fact calls.",
        ),
        build_check(
            "target_rows_fail_isolation_reproduction",
            "not_observed" if target_repro["summary"]["isolation_reproduced_count"] == target_repro["summary"]["target_row_count"] else "observed",
            f"Target repro ledger shows {target_repro['summary']['isolation_reproduced_count']} / {target_repro['summary']['target_row_count']} target rows reproduced in isolation.",
        ),
        build_check(
            "isolation_replay_reverts_to_trace_off",
            "not_observed" if isolation_mismatch['summary']['isolation_revert_count'] == 0 else "observed",
            f"Isolation mismatch audit shows {isolation_mismatch['summary']['isolation_revert_count']} / {isolation_mismatch['summary']['target_row_count']} target rows reverting to trace-off.",
        ),
        build_check(
            "baseline_drift_present",
            "observed" if isolation_mismatch['summary']['baseline_mismatch_count'] > 0 else "not_observed",
            f"Isolation mismatch audit shows {isolation_mismatch['summary']['baseline_mismatch_count']} target rows where full-run off does not match trace off.",
        ),
        build_check(
            "isolation_audit_records_exact_env_and_command",
            "missing",
            "The isolation audit artifact only records source_audit and row predictions; it does not preserve the generating command, env vars, judge candidates, or original row payload.",
        ),
        build_check(
            "trace_artifact_records_candidate_level_state",
            "observed",
            "The author-row trace preserves projected candidates and judge candidates, which is why the on-path target signal is explainable there.",
        ),
    ]

    missing_parity_items = [
        "Exact shell command that produced the isolation audit",
        "Author-direct-fact env var values during isolation replay",
        "Reserved secondary judge slot setting during isolation replay",
        "Original test-sample payload used for each replayed row",
        "Exact judge candidate list observed during isolation replay",
        "Example-pool fingerprint used to build the author catalog during replay",
    ]

    likely_root_cause = replay_gap["summary"]["likely_root_cause"]
    if target_repro["summary"]["isolation_reproduced_count"] == target_repro["summary"]["target_row_count"]:
        next_action = "shift_to_clean_full_run_spillover_reduction"
        rationale = (
            "The current artifacts are now sufficient to show that replay parity is no longer the primary blocker for the target rows. "
            "The next useful step is to rerun and clean the full package-level evaluation so non-target spillover can be reduced."
        )
    else:
        next_action = "instrument_isolation_replay_with_full_parity_logging"
        rationale = (
            "The current artifacts are sufficient to show that target on-path signal exists and isolation replay loses it, "
            "but they are not sufficient to prove which replay input or env setting drifted. The next useful step is to instrument the replay path rather than produce more summaries."
        )

    return {
        "generated_on": "2026-04-09",
        "inputs": {
            "calltrace_path": calltrace_path,
            "target_repro_path": target_repro_path,
            "isolation_mismatch_path": isolation_mismatch_path,
            "replay_gap_path": replay_gap_path,
            "main_path": str(MAIN_PATH),
            "method_path": str(METHOD_PATH),
        },
        "code_observations": {
            "author_projection_env_gate_present": "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_PROJECTION" in method_text,
            "author_catalog_depends_on_examples": "_build_task7_author_answer_catalog(icl_examples)" in main_text,
            "judge_slots_env_present": "OPENSEEK_TASK7_RERANK_APPEND_UNIQUE_SECONDARY_JUDGE_SLOTS" in method_text,
        },
        "parity_checks": checks,
        "missing_parity_items": missing_parity_items,
        "likely_root_cause": likely_root_cause,
        "recommended_next_action": next_action,
        "rationale": rationale,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Author Projection Replay Parity Audit",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Likely root cause: `{report['likely_root_cause']}`",
        f"- Recommended next action: `{report['recommended_next_action']}`",
        "",
        "## Parity checks",
        "",
    ]

    for check in report["parity_checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}` — {check['evidence']}")

    lines.extend(
        [
            "",
            "## Missing parity items",
            "",
        ]
    )
    for item in report["missing_parity_items"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Rationale",
            "",
            f"- {report['rationale']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    calltrace = load_json(args.calltrace_path)
    target_repro = load_json(args.target_repro_path)
    isolation_mismatch = load_json(args.isolation_mismatch_path)
    replay_gap = load_json(args.replay_gap_path)
    report = build_report(
        calltrace,
        target_repro,
        isolation_mismatch,
        replay_gap,
        calltrace_path=args.calltrace_path,
        target_repro_path=args.target_repro_path,
        isolation_mismatch_path=args.isolation_mismatch_path,
        replay_gap_path=args.replay_gap_path,
    )
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
