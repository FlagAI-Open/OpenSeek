import json
from collections import Counter
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"


AUDIT_BASELINE_PATH = WORK_LOGS_DIR / "task2_structured_audit_2026-03-29_v2.json"
AUDIT_MAINLINE_PATH = WORK_LOGS_DIR / "task2_structured_audit_2026-03-29_v5.json"
PROBE_PATH = WORK_LOGS_DIR / "task2_rule_probe_2026-03-29_v3_extended.json"
DELTA_AUDIT_PATH = WORK_LOGS_DIR / "task2_probe_delta_audit_2026-03-29_verb_rel_or_have.json"
LEDGER_PATH = WORK_LOGS_DIR / "official_score_ledger_2026-03-29.json"
CALIBRATION_PATH = WORK_LOGS_DIR / "offline_online_calibration_2026-03-29.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def summarize_bucket_deltas(baseline_audit: dict, mainline_audit: dict) -> list[dict]:
    baseline_counter = Counter(baseline_audit["full_examples"]["error_type_counter"])
    mainline_counter = Counter(mainline_audit["full_examples"]["error_type_counter"])
    rows = []
    for bucket in sorted(set(baseline_counter) | set(mainline_counter)):
        before = baseline_counter.get(bucket, 0)
        after = mainline_counter.get(bucket, 0)
        delta = after - before
        rows.append(
            {
                "error_type": bucket,
                "baseline_count": before,
                "mainline_count": after,
                "delta_count": delta,
                "delta_ratio_vs_baseline": round(safe_div(delta, before), 6),
            }
        )
    rows.sort(key=lambda row: (row["delta_count"], row["baseline_count"]))
    return rows


def summarize_probe_variants(probe_report: dict) -> list[dict]:
    baseline_runs = {
        run["seed"]: run["avg_score"]
        for run in probe_report["variant_reports"]["baseline"]["holdout_runs"]
    }
    rows = []
    for variant, report in probe_report["variant_reports"].items():
        full_section = report["full_examples"]
        holdout_runs = report["holdout_runs"]
        variant_runs = {run["seed"]: run["avg_score"] for run in holdout_runs}
        wins = sum(variant_runs[seed] > baseline_runs[seed] for seed in baseline_runs)
        ties = sum(variant_runs[seed] == baseline_runs[seed] for seed in baseline_runs)
        losses = sum(variant_runs[seed] < baseline_runs[seed] for seed in baseline_runs)
        mean_delta = round(
            sum(variant_runs[seed] - baseline_runs[seed] for seed in baseline_runs) / len(baseline_runs),
            6,
        )
        rows.append(
            {
                "variant": variant,
                "full_examples_avg": full_section["avg_score"],
                "full_examples_delta_vs_baseline": round(
                    full_section["avg_score"] - probe_report["variant_reports"]["baseline"]["full_examples"]["avg_score"],
                    6,
                ),
                "holdout_mean": round(sum(run["avg_score"] for run in holdout_runs) / len(holdout_runs), 6),
                "holdout_delta_mean_vs_baseline": mean_delta,
                "holdout_wins": wins,
                "holdout_ties": ties,
                "holdout_losses": losses,
                "triggered_adjustments": full_section.get("triggered_adjustments", 0),
                "fixed_vs_baseline": full_section.get("fixed_vs_baseline", {}),
                "regressed_vs_baseline": full_section.get("regressed_vs_baseline", {}),
                "local_gate": classify_local_gate(
                    variant=variant,
                    full_delta=full_section["avg_score"] - probe_report["variant_reports"]["baseline"]["full_examples"]["avg_score"],
                    losses=losses,
                    wins=wins,
                ),
            }
        )
    rows.sort(key=lambda row: (row["full_examples_avg"], row["holdout_mean"]), reverse=True)
    return rows


def classify_local_gate(variant: str, full_delta: float, losses: int, wins: int) -> str:
    if variant == "baseline":
        return "reference"
    if full_delta > 0.005 and losses == 0 and wins >= 3:
        return "promotable_now"
    if full_delta > 0 and losses <= 1:
        return "exploratory_positive"
    return "reject_for_now"


def summarize_official_task2_line(ledger: dict, calibration: dict) -> dict:
    entries = [entry for entry in ledger["entries"] if entry["version"] in {"v18", "v22"}]
    line_row = next(row for row in calibration["line_reports"] if row["line_name"] == "task2_structured")
    return {
        "stable_official_baseline": ledger["stable_official_baseline"],
        "task2_line_versions": [entry["version"] for entry in entries],
        "task2_online_deltas": [
            {
                "version": entry["version"],
                "base_version": entry["base_version"],
                "official_score": entry["official_score"],
                "official_delta_vs_base": entry["official_delta_vs_base"],
            }
            for entry in entries
        ],
        "calibration_label": line_row["calibration_label"],
        "promotion_gate": line_row["promotion_gate"],
        "rationale": line_row["rationale"],
    }


def summarize_delta_audit(delta_audit: dict) -> dict:
    full_section = delta_audit["full_examples"]
    changed = full_section["changed_count"]
    fixes = full_section["fix_count"]
    regressions = full_section["regression_count"]
    partial = changed - fixes - regressions
    return {
        "variant": delta_audit["variant"],
        "changed_count": changed,
        "fix_count": fixes,
        "regression_count": regressions,
        "partial_count": partial,
        "fix_precision_over_changed": round(safe_div(fixes, changed), 6),
        "net_gain": fixes - regressions,
        "fixed_baseline_error_counter": full_section["fixed_baseline_error_counter"],
        "regression_error_counter": full_section["regression_error_counter"],
        "sample_fix_examples": full_section["fix_examples"][:8],
        "sample_regression_examples": full_section["regression_examples"][:5],
    }


def build_recommendations(
    bucket_deltas: list[dict],
    variant_rows: list[dict],
    delta_summary: dict,
    official_summary: dict,
) -> list[dict]:
    by_bucket = {row["error_type"]: row for row in bucket_deltas}
    best_variant = next(row for row in variant_rows if row["variant"] == "verb_rel_or_have")
    combo_variant = next(row for row in variant_rows if row["variant"] == "combo_conservative")
    noun_variant = next(row for row in variant_rows if row["variant"] == "noun_initial_multi")

    return [
        {
            "priority": "high",
            "track": "extend_verified_verb_rules",
            "recommendation": "Continue task2 verb-side refinements that are close to verb_rel_or_have: small change sets, few-row impact, and no broad noun rule rewrites.",
            "evidence": {
                "official_task2_line": official_summary["task2_online_deltas"],
                "delta_audit_net_gain": delta_summary["net_gain"],
                "delta_audit_fix_precision_over_changed": delta_summary["fix_precision_over_changed"],
                "tokenization_gap_baseline_to_mainline": {
                    "before": by_bucket["tokenization_gap"]["baseline_count"],
                    "after": by_bucket["tokenization_gap"]["mainline_count"],
                },
                "variant_gate": best_variant["local_gate"],
            },
        },
        {
            "priority": "medium",
            "track": "noun_side_exploration_only",
            "recommendation": "Keep noun_initial_multi and combo_conservative as exploratory probes only; require only-changed audit plus a single-task online package before promoting them.",
            "evidence": {
                "combo_variant": {
                    "full_examples_delta_vs_baseline": combo_variant["full_examples_delta_vs_baseline"],
                    "holdout_losses": combo_variant["holdout_losses"],
                    "local_gate": combo_variant["local_gate"],
                },
                "noun_variant": {
                    "full_examples_delta_vs_baseline": noun_variant["full_examples_delta_vs_baseline"],
                    "holdout_losses": noun_variant["holdout_losses"],
                    "local_gate": noun_variant["local_gate"],
                },
                "residual_bucket_pressure": {
                    "quotation_or_title_token_gap": by_bucket["quotation_or_title_token_gap"]["mainline_count"],
                    "count_rule_gap": by_bucket["count_rule_gap"]["mainline_count"],
                },
            },
        },
        {
            "priority": "low",
            "track": "defer_unstructured_buckets",
            "recommendation": "Do not spend mainline iterations on unknown_other yet; treat it as a later data-cleaning or richer-parser problem.",
            "evidence": {
                "unknown_other_remaining": by_bucket["unknown_other"]["mainline_count"],
                "unknown_other_delta": by_bucket["unknown_other"]["delta_count"],
            },
        },
    ]


def build_report() -> dict:
    baseline_audit = load_json(AUDIT_BASELINE_PATH)
    mainline_audit = load_json(AUDIT_MAINLINE_PATH)
    probe_report = load_json(PROBE_PATH)
    delta_audit = load_json(DELTA_AUDIT_PATH)
    ledger = load_json(LEDGER_PATH)
    calibration = load_json(CALIBRATION_PATH)

    bucket_deltas = summarize_bucket_deltas(baseline_audit, mainline_audit)
    variant_rows = summarize_probe_variants(probe_report)
    delta_summary = summarize_delta_audit(delta_audit)
    official_summary = summarize_official_task2_line(ledger, calibration)

    report = {
        "generated_on": "2026-03-29",
        "stable_official_baseline": official_summary["stable_official_baseline"],
        "task2_official_summary": official_summary,
        "task2_audit_comparison": {
            "baseline_avg_score": baseline_audit["full_examples"]["avg_score"],
            "mainline_avg_score": mainline_audit["full_examples"]["avg_score"],
            "avg_score_delta": round(
                mainline_audit["full_examples"]["avg_score"] - baseline_audit["full_examples"]["avg_score"],
                6,
            ),
            "baseline_mismatch_count": baseline_audit["full_examples"]["mismatch_count"],
            "mainline_mismatch_count": mainline_audit["full_examples"]["mismatch_count"],
            "mismatch_delta": mainline_audit["full_examples"]["mismatch_count"] - baseline_audit["full_examples"]["mismatch_count"],
            "bucket_deltas": bucket_deltas,
        },
        "task2_probe_variant_summary": variant_rows,
        "task2_verified_rule_summary": delta_summary,
        "task2_next_step_recommendations": build_recommendations(
            bucket_deltas=bucket_deltas,
            variant_rows=variant_rows,
            delta_summary=delta_summary,
            official_summary=official_summary,
        ),
    }
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Task2 Signal Calibration",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Stable official baseline: `{report['stable_official_baseline']}`",
        "",
        "## Official Summary",
        "",
    ]
    task2_summary = report["task2_official_summary"]
    for row in task2_summary["task2_online_deltas"]:
        lines.append(
            f"- `{row['version']}` vs `{row['base_version']}`: official `{row['official_score']}` "
            f"(`delta {row['official_delta_vs_base']}`)"
        )
    lines.extend(
        [
            f"- Calibration label: `{task2_summary['calibration_label']}`",
            f"- Promotion gate: `{task2_summary['promotion_gate']}`",
            f"- Rationale: {task2_summary['rationale']}",
            "",
            "## Audit Delta",
            "",
            f"- Baseline avg: `{report['task2_audit_comparison']['baseline_avg_score']}`",
            f"- Mainline avg: `{report['task2_audit_comparison']['mainline_avg_score']}`",
            f"- Avg delta: `{report['task2_audit_comparison']['avg_score_delta']}`",
            f"- Mismatch delta: `{report['task2_audit_comparison']['mismatch_delta']}`",
            "",
            "| Error Bucket | Baseline | Mainline | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["task2_audit_comparison"]["bucket_deltas"]:
        lines.append(
            f"| `{row['error_type']}` | `{row['baseline_count']}` | `{row['mainline_count']}` | `{row['delta_count']}` |"
        )

    lines.extend(
        [
            "",
            "## Probe Variants",
            "",
            "| Variant | Full Delta | Holdout Delta Mean | Wins | Ties | Losses | Gate |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["task2_probe_variant_summary"]:
        lines.append(
            f"| `{row['variant']}` | `{row['full_examples_delta_vs_baseline']}` | "
            f"`{row['holdout_delta_mean_vs_baseline']}` | `{row['holdout_wins']}` | "
            f"`{row['holdout_ties']}` | `{row['holdout_losses']}` | `{row['local_gate']}` |"
        )

    verified = report["task2_verified_rule_summary"]
    lines.extend(
        [
            "",
            "## Verified Rule",
            "",
            f"- Variant: `{verified['variant']}`",
            f"- Changed rows: `{verified['changed_count']}`",
            f"- Fixes: `{verified['fix_count']}`",
            f"- Regressions: `{verified['regression_count']}`",
            f"- Partial only: `{verified['partial_count']}`",
            f"- Fix precision over changed: `{verified['fix_precision_over_changed']}`",
            f"- Net gain: `{verified['net_gain']}`",
            "",
            "## Recommendations",
            "",
        ]
    )
    for row in report["task2_next_step_recommendations"]:
        lines.append(f"- `{row['priority']}` / `{row['track']}`: {row['recommendation']}")
    return "\n".join(lines) + "\n"


def main():
    report = build_report()
    json_path = WORK_LOGS_DIR / "task2_signal_calibration_2026-03-29.json"
    md_path = WORK_LOGS_DIR / "task2_signal_calibration_2026-03-29.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
