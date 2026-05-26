import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from main import TASK_FILES
from method import (
    CHAT_THINKING_TASKS,
    TASK_CHAT_SYSTEM,
    build_chat_examples_with_metadata,
    build_prompt,
    build_task5_retrieval_context,
    build_task6_retrieval_context,
    build_task7_retrieval_context,
    get_example_pool_limit,
    get_profile_name,
    get_profile_retrieval_tasks,
    get_tokenizer,
    reorder_task5_examples_by_lexical_retrieval,
    reorder_task6_examples_by_genre_retrieval,
    reorder_task7_examples_by_lexical_retrieval,
    select_examples_with_metadata,
    should_use_chat_retrieval,
)
from task8_retrieval import build_task8_lexical_retrieval_context, reorder_task8_examples_by_lexical_retrieval


TARGET_RANGES = {
    "default": (28000, 32000),
    8: (15000, 18000),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(1, 9)))
    parser.add_argument("--profiles", type=str, nargs="*", default=["baseline", "long_context"])
    parser.add_argument("--sample_limit", type=int, default=10)
    parser.add_argument("--examples_limit", type=int, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--chat_tasks",
        type=int,
        nargs="*",
        default=sorted(CHAT_THINKING_TASKS),
        help="Tasks treated as chat-thinking during audit.",
    )
    parser.add_argument(
        "--retrieval_tasks",
        type=int,
        nargs="*",
        default=None,
        help="Optional override for retrieval tasks. If omitted, use the active profile defaults.",
    )
    return parser.parse_args()


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)


def get_target_range(task_id: int) -> tuple[int, int]:
    return TARGET_RANGES.get(task_id, TARGET_RANGES["default"])


def build_retrieval_context(task_id: int, examples: list[dict], enable_retrieval: bool):
    if task_id == 8:
        return build_task8_lexical_retrieval_context(examples)
    if task_id == 5 and enable_retrieval:
        return build_task5_retrieval_context(examples)
    if task_id == 6 and enable_retrieval:
        return build_task6_retrieval_context(examples)
    if task_id == 7 and enable_retrieval:
        return build_task7_retrieval_context(examples)
    return None


def maybe_reorder_examples(task_id: int, text2annotate: str, examples: list[dict], retrieval_context, enable_retrieval: bool):
    if task_id == 8:
        return reorder_task8_examples_by_lexical_retrieval(examples, text2annotate, retrieval_context)
    if task_id == 5 and enable_retrieval and retrieval_context is not None:
        return reorder_task5_examples_by_lexical_retrieval(examples, text2annotate, retrieval_context)
    if task_id == 6 and enable_retrieval and retrieval_context is not None:
        return reorder_task6_examples_by_genre_retrieval(examples, text2annotate, retrieval_context)
    if task_id == 7 and enable_retrieval and retrieval_context is not None:
        return reorder_task7_examples_by_lexical_retrieval(examples, text2annotate, retrieval_context)
    return examples


def resolve_retrieval_tasks(retrieval_tasks: list[int] | None, profile_name: str) -> set[int]:
    if retrieval_tasks is None:
        return get_profile_retrieval_tasks(profile_name)
    return set(retrieval_tasks)


def audit_task(task_id: int, profile_name: str, sample_limit: int, examples_limit: int, tokenizer, chat_tasks: set[int], retrieval_tasks: set[int]) -> dict:
    task_dict = json.loads(Path(TASK_FILES[task_id]).read_text(encoding="utf-8"))
    task_description = task_dict["Definition"][0]
    resolved_examples_limit = get_example_pool_limit(
        task_id=task_id,
        examples_limit=examples_limit,
        profile_name=profile_name,
    )
    examples = task_dict["examples"][:resolved_examples_limit]
    test_samples = task_dict["test_samples"][:sample_limit]
    retrieval_enabled = task_id in retrieval_tasks or task_id == 8 or (task_id == 5 and should_use_chat_retrieval(task_id))
    retrieval_context = build_retrieval_context(task_id, examples, retrieval_enabled)

    rows = []
    for sample in test_samples:
        text2annotate = sample["input"]
        ordered_examples = maybe_reorder_examples(task_id, text2annotate, examples, retrieval_context, retrieval_enabled)
        if task_id in chat_tasks:
            examples_meta = build_chat_examples_with_metadata(
                ordered_examples,
                task_id=task_id,
                examples_limit=examples_limit,
                profile_name=profile_name,
            )
            system_msg = TASK_CHAT_SYSTEM.get(task_id, "You are a precise annotation system.")
            user_msg = f"Task: {task_description}\n\nExamples:\n{examples_meta['examples_str']}\nInput: {text2annotate}\nAnswer:"
            total_tokens = count_tokens(tokenizer, system_msg) + count_tokens(tokenizer, user_msg)
        else:
            prompt = build_prompt(task_description, text2annotate, task_id=task_id, profile_name=profile_name)
            examples_meta = select_examples_with_metadata(
                ordered_examples,
                task_description,
                text2annotate,
                task_id=task_id,
                profile_name=profile_name,
            )
            input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_meta["examples_str"] + "\n\n")
            total_tokens = count_tokens(tokenizer, input_prompt)

        target_min, target_max = get_target_range(task_id)
        rows.append(
            {
                "id": sample["id"],
                "profile": profile_name,
                "task_id": task_id,
                "total_tokens": total_tokens,
                "example_tokens": examples_meta["example_tokens"],
                "render_style": examples_meta["render_style"],
                "used_examples": examples_meta["used_examples"],
                "budget_tokens": examples_meta["budget_tokens"],
                "example_pool_limit": resolved_examples_limit,
                "pool_size": examples_meta["pool_size"],
                "pool_limited": examples_meta["pool_limited"],
                "truncated": examples_meta["truncated"],
                "within_target_range": target_min <= total_tokens <= target_max,
                "target_min": target_min,
                "target_max": target_max,
            }
        )

    sample_count = max(len(rows), 1)
    within_target = sum(1 for row in rows if row["within_target_range"])
    return {
        "task_id": task_id,
        "task_name": task_dict["task_name"],
        "profile": profile_name,
        "sample_count": len(rows),
        "avg_total_tokens": round(sum(row["total_tokens"] for row in rows) / sample_count, 2),
        "avg_example_tokens": round(sum(row["example_tokens"] for row in rows) / sample_count, 2),
        "avg_used_examples": round(sum(row["used_examples"] for row in rows) / sample_count, 2),
        "render_style": rows[0]["render_style"] if rows else "baseline",
        "example_pool_limit": resolved_examples_limit,
        "within_target_count": within_target,
        "within_target_ratio": round(within_target / sample_count, 4),
        "pool_limited_count": sum(1 for row in rows if row["pool_limited"]),
        "truncated_count": sum(1 for row in rows if row["truncated"]),
        "target_range": list(get_target_range(task_id)),
        "samples": rows,
    }


def main():
    args = parse_args()
    tokenizer = None
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = get_tokenizer()

    profiles = [get_profile_name(profile) for profile in args.profiles]
    chat_tasks = set(args.chat_tasks)
    results = []
    for profile_name in profiles:
        retrieval_tasks = resolve_retrieval_tasks(args.retrieval_tasks, profile_name)
        for task_id in args.tasks:
            results.append(
                audit_task(
                    task_id=task_id,
                    profile_name=profile_name,
                    sample_limit=args.sample_limit,
                    examples_limit=args.examples_limit,
                    tokenizer=tokenizer,
                    chat_tasks=chat_tasks,
                    retrieval_tasks=retrieval_tasks,
                )
            )

    report = {
        "tasks": args.tasks,
        "profiles": profiles,
        "sample_limit": args.sample_limit,
        "examples_limit": args.examples_limit,
        "retrieval_tasks": args.retrieval_tasks,
        "resolved_retrieval_tasks_by_profile": {
            profile_name: sorted(resolve_retrieval_tasks(args.retrieval_tasks, profile_name))
            for profile_name in profiles
        },
        "results": results,
    }
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
