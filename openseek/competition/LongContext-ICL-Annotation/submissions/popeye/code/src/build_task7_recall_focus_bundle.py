import json
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

FREEZE_PATH = WORK_LOGS_DIR / "task7_post_v27_freeze_2026-03-31.json"
TARGETED_TOPK_PATH = WORK_LOGS_DIR / "task7_targeted_topk_diagnosis_2026-03-31.json"

OUTPUT_JSON_PATH = WORK_LOGS_DIR / "task7_recall_focus_bundle_2026-04-01.json"
OUTPUT_MD_PATH = WORK_LOGS_DIR / "task7_recall_focus_bundle_2026-04-01.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict:
    freeze = load_json(FREEZE_PATH)
    targeted = load_json(TARGETED_TOPK_PATH)

    rows = []
    for row in targeted["rows"]:
        baseline_visible = row["baseline_visible"]
        secondary_only = row["secondary_only_candidates"]
        rows.append(
            {
                "id": row["id"],
                "gold": row["gold"],
                "gold_visible_in_candidates": row["gold_visible_in_candidates"],
                "baseline_visible_count": len(baseline_visible),
                "secondary_only_count": len(secondary_only),
                "baseline_visible": baseline_visible,
                "secondary_only_candidates": secondary_only,
                "best_non_gold_winners": {
                    key: {
                        "winner": value["winner"],
                        "winner_is_secondary": value["winner_is_secondary"],
                    }
                    for key, value in row["variants"].items()
                },
            }
        )

    report = {
        "generated_on": "2026-04-01",
        "base_version": "v27",
        "source_assets": {
            "freeze_report": str(FREEZE_PATH),
            "targeted_topk_diagnosis": str(TARGETED_TOPK_PATH),
        },
        "task7_line_status": "recall_first_research_only",
        "why_now": [
            "Task2 dependency-aware promotion is closed after v28 and v29.",
            "Task8 maintenance was confirmed safe-but-neutral by v30.",
            "Task7 remains the only non-task2 line with a proven online positive version and a still-open mechanism question.",
        ],
        "focus_question": "Can we raise gold_visible_in_candidates on retained hard rows before touching rerank logic again?",
        "gates": {
            "must_improve_gold_visibility": True,
            "must_not_use_threshold_micro_tuning_as_primary_change": True,
            "must_not_promote_new_task7_package_before_visibility_gain": True,
        },
        "baseline_snapshot": {
            "retained_target_count": freeze["coverage_check"]["retained_target_count"],
            "gold_missing_on_retained_targets": freeze["coverage_check"]["gold_missing_on_retained_targets"],
            "order_bias_disagreement_rate": freeze["diagnostic_findings"]["order_bias"]["disagreement_rate"],
            "pairwise_secondary_strong_win_rate": freeze["diagnostic_findings"]["pairwise"]["secondary_strong_win_rate"],
        },
        "retained_targets": rows,
        "recommended_first_experiments": [
            {
                "name": "category-aware candidate expansion",
                "goal": "Add broader alias and entity-surface candidates for the retained rows without changing judge logic.",
            },
            {
                "name": "title/place/name recall probe",
                "goal": "Target the task7 families most likely to miss canonical names in the current candidate set.",
            },
            {
                "name": "retained-row offline visibility check",
                "goal": "Measure gold_visible_in_candidates before and after candidate-generation changes on the frozen retained bundle.",
            },
        ],
        "do_not_return_to": freeze["final_status"]["do_not_return_to"],
    }
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Recall Focus Bundle",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Base version: `{report['base_version']}`",
        f"- Line status: `{report['task7_line_status']}`",
        f"- Focus question: {report['focus_question']}",
        "",
        "## Why This Bundle Exists",
        "",
    ]
    for item in report["why_now"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Baseline Snapshot",
            "",
            f"- Retained targets: `{report['baseline_snapshot']['retained_target_count']}`",
            f"- Gold missing on retained targets: `{report['baseline_snapshot']['gold_missing_on_retained_targets']}`",
            f"- Order-bias disagreement rate: `{report['baseline_snapshot']['order_bias_disagreement_rate']}`",
            f"- Pairwise secondary strong-win rate: `{report['baseline_snapshot']['pairwise_secondary_strong_win_rate']}`",
            "",
            "## Retained Targets",
            "",
        ]
    )

    for row in report["retained_targets"]:
        lines.extend(
            [
                f"- `{row['id']}` gold=`{row['gold']}` visible_in_candidates=`{row['gold_visible_in_candidates']}`",
                f"  - baseline_visible_count=`{row['baseline_visible_count']}` secondary_only_count=`{row['secondary_only_count']}`",
                f"  - baseline_visible={row['baseline_visible']}",
                f"  - secondary_only_candidates={row['secondary_only_candidates']}",
            ]
        )

    lines.extend(["", "## Recommended First Experiments", ""])
    for item in report["recommended_first_experiments"]:
        lines.append(f"- `{item['name']}`: {item['goal']}")

    lines.extend(["", "## Guardrails", ""])
    for item in report["do_not_return_to"]:
        lines.append(f"- Do not return to `{item}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    OUTPUT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
