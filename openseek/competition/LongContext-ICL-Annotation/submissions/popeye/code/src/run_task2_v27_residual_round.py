import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

BASELINE_VERSION = "v27"
BASELINE_SCORE = 73.05
BASELINE_VARIANT = "verb_participle_stuffed_exact"
BASELINE_TASK2_FILE = (
    OUTPUTS_DIR
    / "final_submission_v27_task2stuffed_on_v25_candidate"
    / "openseek-2-v1.jsonl"
)

CANDIDATE_SPECS = [
    {
        "variant": "verb_be_vbg_chain",
        "family": "auxiliary_plus_participle_chain",
        "probe_path": WORK_LOGS_DIR / "task2_rule_probe_2026-03-31_bevbg_vs_v27.json",
        "delta_path": WORK_LOGS_DIR / "task2_probe_delta_audit_2026-03-31_bevbg_vs_v27.json",
        "why_scanned": "Checks whether compressing be + VBG chains can produce another exact-case residual after v27.",
    },
    {
        "variant": "count_rule_helpers",
        "family": "broad_helper_expansion",
        "probe_path": WORK_LOGS_DIR / "task2_rule_probe_2026-03-31_helpers_vs_v27.json",
        "delta_path": WORK_LOGS_DIR / "task2_probe_delta_audit_2026-03-31_helpers_vs_v27.json",
        "why_scanned": "Checks the broad helper branch one final time against the v27 baseline before freezing the line.",
    },
]

MAX_TINY_TEST_DIFF = 12


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_candidate(spec: dict) -> dict:
    probe = load_json(spec["probe_path"])
    delta = load_json(spec["delta_path"])

    baseline_report = probe["variant_reports"]["baseline"]
    variant_report = probe["variant_reports"][spec["variant"]]
    baseline_full = baseline_report["full_examples"]
    variant_full = variant_report["full_examples"]

    baseline_holdout_scores = [run["avg_score"] for run in baseline_report["holdout_runs"]]
    variant_holdout_scores = [run["avg_score"] for run in variant_report["holdout_runs"]]
    holdout_delta_mean = mean(variant_holdout_scores) - mean(baseline_holdout_scores)

    test_changes = variant_report.get("test_changes_vs_compare") or {}
    test_change_count = test_changes.get("changed_count", 0)

    zero_regression = delta["full_examples"]["regression_count"] == 0
    full_audit_positive = (variant_full["avg_score"] - baseline_full["avg_score"]) > 0 and holdout_delta_mean >= 0
    nonzero_test_diff = test_change_count > 0
    tiny_test_diff = test_change_count <= MAX_TINY_TEST_DIFF
    risk_not_above_v27 = zero_regression and tiny_test_diff

    reject_reasons = []
    if not zero_regression:
        reject_reasons.append("has_regressions")
    if not full_audit_positive:
        reject_reasons.append("no_positive_full_audit")
    if not nonzero_test_diff:
        reject_reasons.append("no_test_change")
    if not tiny_test_diff:
        reject_reasons.append("test_diff_too_large")
    if spec["family"] != "exact_case":
        reject_reasons.append("not_same_risk_as_stuffed_exact")

    return {
        "variant": spec["variant"],
        "family": spec["family"],
        "why_scanned": spec["why_scanned"],
        "baseline_variant": probe["baseline_variant"],
        "full_examples_avg": variant_full["avg_score"],
        "full_examples_delta_vs_v27": round(variant_full["avg_score"] - baseline_full["avg_score"], 6),
        "full_examples_mismatch_delta_vs_v27": (
            variant_full["mismatch_count"] - baseline_full["mismatch_count"]
        ),
        "holdout_delta_mean_vs_v27": round(holdout_delta_mean, 6),
        "fix_count": delta["full_examples"]["fix_count"],
        "regression_count": delta["full_examples"]["regression_count"],
        "net_gain": delta["full_examples"]["fix_count"] - delta["full_examples"]["regression_count"],
        "test_change_count_vs_v27": test_change_count,
        "test_change_directions": test_changes.get("direction_counter", {}),
        "sample_test_changes": test_changes.get("changed_examples", [])[:8],
        "gate_results": {
            "zero_regression": zero_regression,
            "full_audit_positive": full_audit_positive,
            "nonzero_test_diff": nonzero_test_diff,
            "tiny_test_diff": tiny_test_diff,
            "risk_not_above_v27": risk_not_above_v27,
        },
        "reject_reasons": reject_reasons,
        "decision": "promotable" if not reject_reasons else "reject",
    }


def build_report() -> dict:
    candidates = [summarize_candidate(spec) for spec in CANDIDATE_SPECS]
    promotable = [row for row in candidates if row["decision"] == "promotable"]

    conclusion = {
        "baseline_version": BASELINE_VERSION,
        "baseline_score": BASELINE_SCORE,
        "baseline_task2_file": str(BASELINE_TASK2_FILE),
        "baseline_variant": BASELINE_VARIANT,
        "promotable_candidate_count": len(promotable),
        "residual_round_result": (
            "found_same_risk_candidate"
            if promotable
            else "task2_residual_closed_stop_high_frequency_scan"
        ),
        "next_action": (
            "stay_on_task2_mainline"
            if promotable
            else "keep_v27_as_formal_base_and_shift_primary_resources_to_method_spike"
        ),
        "stop_reasons_triggered": [
            "no_new_zero_regression_candidate",
            "remaining_variants_slid_into_broader_proxy_or_helper_patterns",
        ]
        if not promotable
        else [],
    }

    return {
        "generated_on": "2026-03-31",
        "baseline": {
            "version": BASELINE_VERSION,
            "official_score": BASELINE_SCORE,
            "task2_file": str(BASELINE_TASK2_FILE),
            "baseline_variant": BASELINE_VARIANT,
        },
        "candidate_specs": [
            {
                "variant": spec["variant"],
                "family": spec["family"],
                "probe_path": str(spec["probe_path"]),
                "delta_path": str(spec["delta_path"]),
            }
            for spec in CANDIDATE_SPECS
        ],
        "gate_policy": {
            "require_zero_regression": True,
            "require_positive_full_audit": True,
            "require_nonzero_test_diff": True,
            "require_tiny_test_diff": True,
            "max_tiny_test_diff": MAX_TINY_TEST_DIFF,
            "require_same_risk_as_v27": True,
        },
        "candidates": candidates,
        "conclusion": conclusion,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task2 v27 Residual Round",
        "",
        f"- Baseline: `{report['baseline']['version']} = {report['baseline']['official_score']}`",
        f"- Baseline variant: `{report['baseline']['baseline_variant']}`",
        f"- Gate policy: `0 regression`, positive full audit, non-zero tiny test diff, risk not above `v27`.",
        "",
        "## Candidate Summary",
        "",
        "| Variant | Family | Full delta vs v27 | Holdout mean delta | Fixes | Regressions | Test changes | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in report["candidates"]:
        lines.append(
            f"| `{row['variant']}` | `{row['family']}` | `{row['full_examples_delta_vs_v27']}` | "
            f"`{row['holdout_delta_mean_vs_v27']}` | `{row['fix_count']}` | `{row['regression_count']}` | "
            f"`{row['test_change_count_vs_v27']}` | `{row['decision']}` |"
        )

    lines.extend(["", "## Gate Findings", ""])
    for row in report["candidates"]:
        lines.append(f"- `{row['variant']}`: {row['why_scanned']}")
        lines.append(
            "  Gate results: "
            + ", ".join(f"{name}={value}" for name, value in row["gate_results"].items())
        )
        if row["reject_reasons"]:
            lines.append(f"  Reject reasons: {', '.join(row['reject_reasons'])}")

    conclusion = report["conclusion"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Residual round result: `{conclusion['residual_round_result']}`",
            f"- Next action: `{conclusion['next_action']}`",
        ]
    )
    if conclusion["stop_reasons_triggered"]:
        lines.append(
            f"- Stop reasons triggered: `{', '.join(conclusion['stop_reasons_triggered'])}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    json_path = WORK_LOGS_DIR / "task2_v27_residual_round_2026-03-31.json"
    md_path = WORK_LOGS_DIR / "task2_v27_residual_round_2026-03-31.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
