import argparse
import json
import random
from collections import Counter
from pathlib import Path

from main import TASK_FILES
from method import (
    build_task6_retrieval_context,
    build_task7_retrieval_context,
    get_example_pool_limit,
    get_profile_name,
    get_profile_retrieval_tasks,
    parse_task6_fields,
    parse_task7_fields,
    reorder_task6_examples_by_genre_retrieval,
    reorder_task7_examples_by_lexical_retrieval,
    select_examples_with_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="*", default=[6, 7])
    parser.add_argument("--profiles", type=str, nargs="*", default=["baseline", "frontier_task6_task7"])
    parser.add_argument("--sample_limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--examples_limit", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def summarize_selected_examples(task_id: int, ordered_examples: list[dict], used_examples: int, sample_input: str) -> dict:
    selected = ordered_examples[:used_examples]
    label_counter = Counter(example["output"][0] for example in selected)

    if task_id == 6:
        _, _, target_genre = parse_task6_fields(sample_input)
        genre_counter = Counter()
        same_genre = 0
        for example in selected:
            _, _, genre = parse_task6_fields(example["input"])
            genre_counter[genre] += 1
            same_genre += int(genre == target_genre)
        return {
            "target_genre": target_genre,
            "same_genre_count": same_genre,
            "label_counter": dict(label_counter),
            "top_genres": genre_counter.most_common(8),
        }

    category, _ = parse_task7_fields(sample_input)
    category_counter = Counter()
    same_category = 0
    for example in selected:
        ex_category, _ = parse_task7_fields(example["input"])
        category_counter[ex_category] += 1
        same_category += int(ex_category == category)
    return {
        "target_category": category,
        "same_category_count": same_category,
        "label_counter": dict(label_counter),
        "top_categories": category_counter.most_common(8),
    }


def audit_task_profile(task_id: int, profile_name: str, sample_limit: int, seed: int, examples_limit: int | None) -> dict:
    task_dict = json.loads(Path(TASK_FILES[task_id]).read_text(encoding="utf-8"))
    examples = list(task_dict["examples"])
    test_samples = list(task_dict["test_samples"])
    rng = random.Random(seed + task_id)
    sampled_tests = rng.sample(test_samples, min(sample_limit, len(test_samples)))

    resolved_examples_limit = get_example_pool_limit(
        task_id=task_id,
        examples_limit=examples_limit,
        profile_name=profile_name,
    )
    icl_examples = examples[:resolved_examples_limit]
    retrieval_tasks = get_profile_retrieval_tasks(profile_name)
    retrieval_context = None
    if task_id == 6 and task_id in retrieval_tasks:
        retrieval_context = build_task6_retrieval_context(icl_examples)
    elif task_id == 7 and task_id in retrieval_tasks:
        retrieval_context = build_task7_retrieval_context(icl_examples)

    rows = []
    for sample in sampled_tests:
        ordered_examples = icl_examples
        if task_id == 6 and retrieval_context is not None:
            ordered_examples = reorder_task6_examples_by_genre_retrieval(icl_examples, sample["input"], retrieval_context)
        elif task_id == 7 and retrieval_context is not None:
            ordered_examples = reorder_task7_examples_by_lexical_retrieval(icl_examples, sample["input"], retrieval_context)

        meta = select_examples_with_metadata(
            ordered_examples,
            task_dict["Definition"][0],
            sample["input"],
            task_id=task_id,
            profile_name=profile_name,
        )
        summary = summarize_selected_examples(task_id, ordered_examples, meta["used_examples"], sample["input"])
        rows.append(
            {
                "id": sample["id"],
                "used_examples": meta["used_examples"],
                "example_tokens": meta["example_tokens"],
                "budget_tokens": meta["budget_tokens"],
                "pool_size": meta["pool_size"],
                "truncated": meta["truncated"],
                **summary,
            }
        )

    return {
        "task_id": task_id,
        "task_name": task_dict["task_name"],
        "profile": profile_name,
        "sample_limit": len(rows),
        "example_pool_limit": resolved_examples_limit,
        "retrieval_enabled": task_id in retrieval_tasks,
        "avg_used_examples": round(sum(row["used_examples"] for row in rows) / max(len(rows), 1), 2),
        "avg_example_tokens": round(sum(row["example_tokens"] for row in rows) / max(len(rows), 1), 2),
        "rows": rows,
    }


def main():
    args = parse_args()
    profiles = [get_profile_name(profile) for profile in args.profiles]
    report = {
        "tasks": args.tasks,
        "profiles": profiles,
        "sample_limit": args.sample_limit,
        "seed": args.seed,
        "results": [],
    }
    for profile_name in profiles:
        for task_id in args.tasks:
            report["results"].append(
                audit_task_profile(
                    task_id=task_id,
                    profile_name=profile_name,
                    sample_limit=args.sample_limit,
                    seed=args.seed,
                    examples_limit=args.examples_limit,
                )
            )

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
