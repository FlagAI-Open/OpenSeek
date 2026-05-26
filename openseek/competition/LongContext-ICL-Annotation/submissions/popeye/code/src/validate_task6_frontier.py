import argparse
import json
import random
from pathlib import Path

from main import TASK_FILES
from method import (
    annotate_nvidia as annotate,
    build_prompt,
    build_task6_retrieval_context,
    get_example_pool_limit,
    get_profile_name,
    get_profile_retrieval_tasks,
    get_tokenizer,
    parse_task6_fields,
    reorder_task6_examples_by_genre_retrieval,
    select_examples_with_metadata,
    solve_task_locally,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--profiles", type=str, nargs="*", default=["baseline", "frontier_task6_task7"])
    parser.add_argument("--examples_limit", type=int, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_input_length", type=int, default=128000)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def normalize_binary_label(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = str(text).strip().upper()
    if cleaned in {"Y", "N"}:
        return cleaned
    return None


def count_same_genre(examples: list[dict], target_genre: str) -> int:
    target = target_genre.lower()
    matched = 0
    for example in examples:
        try:
            _, _, genre = parse_task6_fields(example["input"])
        except Exception:
            continue
        if genre.lower() == target:
            matched += 1
    return matched


def build_profile_report(
    *,
    profile_name: str,
    holdout: list[dict],
    train_examples: list[dict],
    task_description: str,
    tokenizer,
    max_input_length: int,
    examples_limit: int | None,
) -> dict:
    resolved_examples_limit = get_example_pool_limit(
        task_id=6,
        examples_limit=examples_limit,
        profile_name=profile_name,
    )
    icl_pool = train_examples[:resolved_examples_limit]
    retrieval_tasks = get_profile_retrieval_tasks(profile_name)
    retrieval_context = build_task6_retrieval_context(icl_pool) if 6 in retrieval_tasks else None

    rows = []
    correct = 0
    positives = 0
    prompt_skips = 0
    same_genre_selected_total = 0
    same_genre_selected_ratio_total = 0.0
    same_genre_top20_total = 0
    same_genre_top20_ratio_total = 0.0
    example_tokens_total = 0
    used_examples_total = 0
    prompt_tokens_total = 0

    for sample in holdout:
        text2annotate = sample["input"]
        gold = sample["output"][0]
        _, _, genre = parse_task6_fields(text2annotate)

        ordered_examples = list(icl_pool)
        if retrieval_context is not None:
            ordered_examples = reorder_task6_examples_by_genre_retrieval(
                ordered_examples,
                text2annotate,
                retrieval_context,
            )

        selection = select_examples_with_metadata(
            ordered_examples,
            task_description,
            text2annotate,
            task_id=6,
            profile_name=profile_name,
        )
        used_examples = ordered_examples[: selection["used_examples"]]
        same_genre_selected = count_same_genre(used_examples, genre)
        top20_examples = ordered_examples[:20]
        same_genre_top20 = count_same_genre(top20_examples, genre)

        prompt = build_prompt(
            task_description,
            text2annotate,
            task_id=6,
            profile_name=profile_name,
        ).replace("[[EXAMPLES]]\n\n", selection["examples_str"] + "\n\n")

        prompt_tokens = None
        skipped_due_to_length = False
        if tokenizer is not None:
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens > max_input_length:
                skipped_due_to_length = True

        if skipped_due_to_length:
            prompt_skips += 1
            prediction = None
            source = "skip_prompt_too_long"
        else:
            local_prediction = solve_task_locally(6, text2annotate)
            if local_prediction is not None:
                prediction = local_prediction
                source = "local"
            else:
                prediction = annotate(prompt, task_id=6)
                source = "llm"

        prediction = normalize_binary_label(prediction)
        ok = prediction == gold
        correct += int(ok)
        positives += int(prediction == "Y")
        used_count = len(used_examples)
        same_genre_selected_ratio = (same_genre_selected / used_count) if used_count else 0.0
        same_genre_top20_ratio = (same_genre_top20 / len(top20_examples)) if top20_examples else 0.0

        same_genre_selected_total += same_genre_selected
        same_genre_selected_ratio_total += same_genre_selected_ratio
        same_genre_top20_total += same_genre_top20
        same_genre_top20_ratio_total += same_genre_top20_ratio
        example_tokens_total += selection["example_tokens"]
        used_examples_total += selection["used_examples"]
        if prompt_tokens is not None:
            prompt_tokens_total += prompt_tokens

        rows.append(
            {
                "id": sample["id"],
                "genre": genre,
                "gold": gold,
                "prediction": prediction,
                "correct": ok,
                "source": source,
                "used_examples": selection["used_examples"],
                "example_tokens": selection["example_tokens"],
                "prompt_tokens": prompt_tokens,
                "same_genre_selected": same_genre_selected,
                "same_genre_selected_ratio": same_genre_selected_ratio,
                "same_genre_top20": same_genre_top20,
                "same_genre_top20_ratio": same_genre_top20_ratio,
                "truncated": selection["truncated"],
            }
        )

    sample_count = len(holdout)
    return {
        "profile": profile_name,
        "example_pool_limit": resolved_examples_limit,
        "retrieval_enabled": 6 in retrieval_tasks,
        "sample_limit": sample_count,
        "accuracy": correct / sample_count if sample_count else 0.0,
        "positives_ratio": positives / sample_count if sample_count else 0.0,
        "prompt_skip_count": prompt_skips,
        "avg_used_examples": used_examples_total / sample_count if sample_count else 0.0,
        "avg_example_tokens": example_tokens_total / sample_count if sample_count else 0.0,
        "avg_prompt_tokens": prompt_tokens_total / sample_count if sample_count and tokenizer is not None else None,
        "same_genre_summary": {
            "avg_same_genre_selected": same_genre_selected_total / sample_count if sample_count else 0.0,
            "avg_same_genre_selected_ratio": same_genre_selected_ratio_total / sample_count if sample_count else 0.0,
            "avg_same_genre_top20": same_genre_top20_total / sample_count if sample_count else 0.0,
            "avg_same_genre_top20_ratio": same_genre_top20_ratio_total / sample_count if sample_count else 0.0,
        },
        "rows": rows,
    }


def main():
    args = parse_args()
    tokenizer = get_tokenizer(args.tokenizer_path) if args.tokenizer_path else get_tokenizer()

    task_dict = json.loads(Path(TASK_FILES[6]).read_text(encoding="utf-8"))
    examples = list(task_dict["examples"])
    task_description = task_dict["Definition"][0]

    rng = random.Random(args.seed + 6)
    holdout = rng.sample(examples, min(args.sample_limit, len(examples)))
    holdout_ids = {example["id"] for example in holdout}
    train_examples = [example for example in examples if example["id"] not in holdout_ids]

    reports = []
    for profile in args.profiles:
        reports.append(
            build_profile_report(
                profile_name=get_profile_name(profile),
                holdout=holdout,
                train_examples=train_examples,
                task_description=task_description,
                tokenizer=tokenizer,
                max_input_length=args.max_input_length,
                examples_limit=args.examples_limit,
            )
        )

    report = {
        "seed": args.seed,
        "sample_limit": len(holdout),
        "profiles": reports,
    }

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
