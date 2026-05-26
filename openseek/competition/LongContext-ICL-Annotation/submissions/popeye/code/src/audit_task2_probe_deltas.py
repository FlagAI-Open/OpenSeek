import argparse
import json
import random
from collections import Counter
from pathlib import Path

from audit_task2_structured import classify_task2_gap
from main import TASK_FILES
from probe_task2_rule_patches import build_example_baseline_predictions, variant_prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--variant", type=str, default="verb_rel_or_have")
    parser.add_argument(
        "--baseline_variant",
        type=str,
        default="mainline",
        help="Example-side baseline variant. Use 'mainline' for solve_task2_structured_count, or a named probe variant such as 'verb_mistagged_guarded'.",
    )
    parser.add_argument("--sample_limit", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 123, 2026])
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_examples(task_id: int) -> list[dict]:
    task_dict = json.loads(Path(TASK_FILES[task_id]).read_text(encoding="utf-8"))
    return list(task_dict["examples"])


def audit_subset(
    samples: list[dict],
    variant: str,
    baseline_predictions: dict[str, str | None],
) -> dict:
    changed_rows = []
    fixes = []
    regressions = []
    changed_error_counter = Counter()
    fix_counter = Counter()
    regression_counter = Counter()

    for sample in samples:
        sample_id = sample["id"]
        baseline_pred = baseline_predictions[sample_id]
        variant_pred = variant_prediction(sample["input"], variant, baseline_pred)
        if variant_pred == baseline_pred:
            continue

        gold = sample["output"][0]
        variant_gap = classify_task2_gap(sample["input"], gold, variant_pred)
        baseline_gap = classify_task2_gap(sample["input"], gold, baseline_pred) if baseline_pred != gold else None

        row = {
            "id": sample_id,
            "gold": gold,
            "baseline_prediction": baseline_pred,
            "variant_prediction": variant_pred,
            "input": sample["input"],
            "variant_error_type": variant_gap["error_type"],
            "variant_priority_bucket": variant_gap["priority_bucket"],
            "baseline_error_type": baseline_gap["error_type"] if baseline_gap else None,
            "status": "fix" if baseline_pred != gold and variant_pred == gold else "regression" if baseline_pred == gold and variant_pred != gold else "changed_but_still_wrong",
        }
        changed_rows.append(row)
        changed_error_counter[row["variant_error_type"]] += 1

        if row["status"] == "fix":
            fixes.append(row)
            if baseline_gap is not None:
                fix_counter[baseline_gap["error_type"]] += 1
        elif row["status"] == "regression":
            regressions.append(row)
            regression_counter[row["variant_error_type"]] += 1

    return {
        "sample_count": len(samples),
        "changed_count": len(changed_rows),
        "fix_count": len(fixes),
        "regression_count": len(regressions),
        "changed_error_counter": dict(changed_error_counter),
        "fixed_baseline_error_counter": dict(fix_counter),
        "regression_error_counter": dict(regression_counter),
        "changed_examples": changed_rows[:20],
        "fix_examples": fixes[:20],
        "regression_examples": regressions[:20],
    }


def main():
    args = parse_args()
    examples = load_examples(args.task_id)
    baseline_predictions = build_example_baseline_predictions(examples, args.baseline_variant)

    report = {
        "task_id": args.task_id,
        "variant": args.variant,
        "baseline_variant": args.baseline_variant,
        "sample_limit": args.sample_limit,
        "seeds": args.seeds,
        "full_examples": audit_subset(examples, args.variant, baseline_predictions),
        "holdout_runs": [],
    }

    for seed in args.seeds:
        rng = random.Random(seed + args.task_id)
        holdout = rng.sample(examples, min(args.sample_limit, len(examples)))
        section = audit_subset(holdout, args.variant, baseline_predictions)
        section["seed"] = seed
        report["holdout_runs"].append(section)

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
