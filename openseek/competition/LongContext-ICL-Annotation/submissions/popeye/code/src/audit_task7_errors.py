import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_files",
        type=str,
        nargs="+",
        required=True,
        help="Task7 holdout result JSON files produced by validate_holdout.py or similar scripts.",
    )
    parser.add_argument(
        "--task7_data",
        type=str,
        default="data/openseek-7_jeopardy_answer_generation_all.json",
        help="Path to the task7 dataset JSON.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the audit report as JSON.",
    )
    return parser.parse_args()


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_task7_fields(text: str) -> tuple[str, str]:
    category_match = re.search(r"Category:\s*(.*?)\s*Clue:", text, flags=re.I | re.S)
    clue_match = re.search(r"Clue:\s*(.*)", text, flags=re.I | re.S)
    category = category_match.group(1).strip() if category_match else ""
    clue = clue_match.group(1).strip() if clue_match else text.strip()
    return category, clue


def classify_mismatch(gold: str, prediction: str | None) -> str:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return "empty"
    if pred_norm == gold_norm:
        return "exact"
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return "substring"
    gold_tokens = set(gold_norm.split())
    pred_tokens = set(pred_norm.split())
    if gold_tokens and pred_tokens and gold_tokens == pred_tokens:
        return "token_reorder"
    if gold_tokens & pred_tokens:
        return "partial_overlap"
    return "entity_confusion"


def reconstruct_holdout(examples: list[dict], seed: int, sample_limit: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 7)
    holdout = rng.sample(examples, min(sample_limit, len(examples)))
    holdout_ids = {example["id"] for example in holdout}
    pool = [example for example in examples if example["id"] not in holdout_ids]
    return holdout, pool


def audit_result_file(result_path: Path, examples: list[dict]) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_rows = payload.get("results", [])
    task7_row = None
    for row in result_rows:
        if row.get("task_id") == 7:
            task7_row = row
            break
    if task7_row is None:
        raise ValueError(f"No task7 row found in {result_path}")

    seed = int(payload["seed"])
    sample_limit = int(payload["sample_limit"])
    holdout, pool = reconstruct_holdout(examples, seed=seed, sample_limit=sample_limit)
    pool_category_counter = Counter(parse_task7_fields(example["input"])[0].strip().lower() for example in pool)
    pool_answer_counter = Counter(example["output"][0].strip().lower() for example in pool)

    mismatch_type_counter = Counter()
    category_presence_counter = Counter()
    answer_repeat_counter = Counter()
    mismatch_rows = []

    mismatch_map = {row["id"]: row for row in task7_row.get("mismatch_examples", [])}
    for sample in holdout:
        category, clue = parse_task7_fields(sample["input"])
        gold = sample["output"][0]
        category_key = category.strip().lower()
        gold_key = gold.strip().lower()

        category_status = "no_same_category"
        if pool_category_counter[category_key] > 0:
            category_status = "same_category_present"
        if pool_category_counter[category_key] > 1:
            category_status = "same_category_multi"
        category_presence_counter[category_status] += 1

        answer_status = "answer_unique_in_pool"
        if pool_answer_counter[gold_key] > 0:
            answer_status = "answer_seen_in_pool"
        if pool_answer_counter[gold_key] > 1:
            answer_status = "answer_repeated_in_pool"
        answer_repeat_counter[answer_status] += 1

        mismatch = mismatch_map.get(sample["id"])
        if mismatch is None:
            continue
        mismatch_type = classify_mismatch(gold, mismatch.get("prediction"))
        mismatch_type_counter[mismatch_type] += 1
        mismatch_rows.append(
            {
                "id": sample["id"],
                "category": category,
                "clue": clue,
                "gold": gold,
                "prediction": mismatch.get("prediction"),
                "mismatch_type": mismatch_type,
                "category_status": category_status,
                "answer_status": answer_status,
            }
        )

    return {
        "file": result_path.name,
        "seed": seed,
        "sample_limit": sample_limit,
        "avg_score": task7_row["avg_score"],
        "sample_count": task7_row["sample_count"],
        "mismatch_count": task7_row["mismatch_count"],
        "mismatch_type_counter": dict(mismatch_type_counter),
        "category_presence_counter": dict(category_presence_counter),
        "answer_repeat_counter": dict(answer_repeat_counter),
        "mismatch_examples": mismatch_rows[:20],
    }


def main():
    args = parse_args()
    task7_payload = json.loads(Path(args.task7_data).read_text(encoding="utf-8"))
    examples = list(task7_payload["examples"])

    report = {
        "task7_data": args.task7_data,
        "result_files": args.result_files,
        "reports": [],
    }
    aggregate_mismatch_types = Counter()
    aggregate_category_status = Counter()
    aggregate_answer_status = Counter()

    for result_file in args.result_files:
        file_report = audit_result_file(Path(result_file), examples)
        report["reports"].append(file_report)
        aggregate_mismatch_types.update(file_report["mismatch_type_counter"])
        aggregate_category_status.update(file_report["category_presence_counter"])
        aggregate_answer_status.update(file_report["answer_repeat_counter"])

    report["aggregate"] = {
        "mismatch_type_counter": dict(aggregate_mismatch_types),
        "category_presence_counter": dict(aggregate_category_status),
        "answer_repeat_counter": dict(aggregate_answer_status),
    }

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
