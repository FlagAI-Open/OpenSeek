import argparse
import json
from collections import Counter
from pathlib import Path

from main import TASK_FILES
from method import (
    _request_nvidia_completions,
    build_prompt,
    build_task7_retrieval_context,
    build_task7_rerank_generation_prompt,
    count_answer,
    detect_task7_secondary_family,
    get_example_pool_limit,
    get_profile_name,
    get_profile_retrieval_tasks,
    normalize_task7_answer,
    parse_task7_fields,
    reorder_task7_examples_by_lexical_retrieval,
    select_examples,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_FOCUS_BUNDLE = WORK_LOGS_DIR / "task7_recall_focus_bundle_2026-04-01.json"
DEFAULT_SUPPORT_EXAMPLES = WORK_LOGS_DIR / "task7_recall_support_examples_2026-04-01.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_bundle", type=str, default=str(DEFAULT_FOCUS_BUNDLE))
    parser.add_argument("--support_examples_path", type=str, default=str(DEFAULT_SUPPORT_EXAMPLES))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n_candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline_off,baseline_full,baseline_typed,baseline_recall,retrieval_off,retrieval_full,retrieval_typed,retrieval_recall,support_recall",
        help="Comma-separated variant names to run.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_map() -> dict[str, dict]:
    task7 = load_json(Path(TASK_FILES[7]))
    sample_map = {}
    for section in ("test_samples", "examples"):
        for sample in task7[section]:
            sample_map[sample["id"]] = sample
    return sample_map, list(task7["examples"])


def build_variant_specs(raw_variants: str) -> list[dict]:
    variant_specs = {
        "baseline_off": {
            "profile": "baseline",
            "retrieval_mode": "none",
            "hint_mode": "off",
        },
        "baseline_full": {
            "profile": "baseline",
            "retrieval_mode": "none",
            "hint_mode": "full",
        },
        "baseline_typed": {
            "profile": "baseline",
            "retrieval_mode": "none",
            "hint_mode": "typed",
        },
        "baseline_recall": {
            "profile": "baseline",
            "retrieval_mode": "none",
            "hint_mode": "recall",
        },
        "retrieval_off": {
            "profile": "long_context_task7_retrieval",
            "retrieval_mode": "lexical",
            "hint_mode": "off",
        },
        "retrieval_full": {
            "profile": "long_context_task7_retrieval",
            "retrieval_mode": "lexical",
            "hint_mode": "full",
        },
        "retrieval_typed": {
            "profile": "long_context_task7_retrieval",
            "retrieval_mode": "lexical",
            "hint_mode": "typed",
        },
        "retrieval_recall": {
            "profile": "long_context_task7_retrieval",
            "retrieval_mode": "lexical",
            "hint_mode": "recall",
        },
        "support_recall": {
            "profile": "baseline",
            "retrieval_mode": "none",
            "hint_mode": "recall",
            "example_source": "support",
        },
    }
    names = [token.strip() for token in raw_variants.split(",") if token.strip()]
    return [{"name": name, **variant_specs[name]} for name in names]


def build_task7_input_prompt(
    *,
    examples: list[dict],
    task_description: str,
    text2annotate: str,
    profile_name: str,
    retrieval_mode: str,
) -> tuple[str, list[dict]]:
    prompt = build_prompt(task_description, text2annotate, task_id=7, profile_name=profile_name)
    ordered_examples = examples
    use_retrieval = retrieval_mode == "lexical" or 7 in get_profile_retrieval_tasks(profile_name)
    if use_retrieval:
        retrieval_context = build_task7_retrieval_context(examples)
        ordered_examples = reorder_task7_examples_by_lexical_retrieval(
            examples,
            text2annotate,
            retrieval_context,
        )
    examples_str = select_examples(
        ordered_examples,
        task_description,
        text2annotate,
        task_id=7,
        profile_name=profile_name,
    )
    input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")
    return input_prompt, ordered_examples


def collect_candidates(
    *,
    input_prompt: str,
    text2annotate: str,
    rounds: int,
    n_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    hint_mode: str,
) -> tuple[list[str], list[str]]:
    raw_choices = []
    for _ in range(max(1, rounds)):
        prompt = (
            build_task7_rerank_generation_prompt(input_prompt, text2annotate, hint_mode=hint_mode)
            if hint_mode != "off"
            else input_prompt
        )
        raw_choices.extend(
            _request_nvidia_completions(
                prompt,
                task_id=7,
                n=n_candidates,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )
    deduped = []
    seen = set()
    for raw in raw_choices:
        answer = count_answer(raw, task_id=7)
        normalized = normalize_task7_answer(answer)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(answer.strip())
    return raw_choices, deduped


def summarize_variant(
    *,
    row: dict,
    sample_input: str,
    task_description: str,
    all_examples: list[dict],
    support_example_ids: list[str] | None,
    variant: dict,
    rounds: int,
    n_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict:
    profile_name = get_profile_name(variant["profile"])
    examples_limit = get_example_pool_limit(task_id=7, examples_limit=None, profile_name=profile_name)
    if variant.get("example_source") == "support" and support_example_ids:
        support_id_set = set(support_example_ids)
        icl_examples = [example for example in all_examples if example["id"] in support_id_set][:examples_limit]
    else:
        icl_examples = [example for example in all_examples if example["id"] != row["id"]][:examples_limit]
    input_prompt, ordered_examples = build_task7_input_prompt(
        examples=icl_examples,
        task_description=task_description,
        text2annotate=sample_input,
        profile_name=profile_name,
        retrieval_mode=variant["retrieval_mode"],
    )
    raw_choices, deduped_candidates = collect_candidates(
        input_prompt=input_prompt,
        text2annotate=sample_input,
        rounds=rounds,
        n_candidates=n_candidates,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        hint_mode=variant["hint_mode"],
    )
    counts = Counter(normalize_task7_answer(count_answer(raw, task_id=7)) for raw in raw_choices)
    counts.pop("", None)
    gold_norm = normalize_task7_answer(row["gold"])
    visible = gold_norm in {normalize_task7_answer(candidate) for candidate in deduped_candidates}
    category, clue = parse_task7_fields(sample_input)
    return {
        "variant": variant["name"],
        "profile": profile_name,
        "retrieval_mode": variant["retrieval_mode"],
        "hint_mode": variant["hint_mode"],
        "detected_family": detect_task7_secondary_family(category, clue, typed_route="auto"),
        "raw_choice_count": len(raw_choices),
        "unique_candidate_count": len(deduped_candidates),
        "gold_visible_in_candidates": visible,
        "gold": row["gold"],
        "gold_rank": next(
            (
                idx + 1
                for idx, candidate in enumerate(deduped_candidates)
                if normalize_task7_answer(candidate) == gold_norm
            ),
            None,
        ),
        "top_candidates": deduped_candidates[:10],
        "candidate_vote_counts": {
            candidate: counts.get(normalize_task7_answer(candidate), 0)
            for candidate in deduped_candidates[:10]
        },
        "example_pool_limit": examples_limit,
        "example_source": variant.get("example_source", "default"),
        "first_retrieved_example_ids": [example["id"] for example in ordered_examples[:5]],
    }


def main():
    args = parse_args()
    focus_bundle = load_json(Path(args.focus_bundle))
    support_examples = load_json(Path(args.support_examples_path))
    test_map, all_examples = load_test_map()
    task7 = load_json(Path(TASK_FILES[7]))
    task_description = task7["Definition"][0]
    variants = build_variant_specs(args.variants)
    support_map = {
        row["target_id"]: [item["id"] for item in row["top_support_examples"]]
        for row in support_examples["results"]
    }

    report_rows = []
    variant_aggregate = {
        variant["name"]: {
            "row_count": 0,
            "gold_visible_count": 0,
        }
        for variant in variants
    }

    for row in focus_bundle["retained_targets"]:
        sample = test_map[row["id"]]
        row_report = {
            "id": row["id"],
            "gold": row["gold"],
            "input": sample["input"],
            "variants": [],
        }
        for variant in variants:
            variant_report = summarize_variant(
                row=row,
                sample_input=sample["input"],
                task_description=task_description,
                all_examples=all_examples,
                support_example_ids=support_map.get(row["id"]),
                variant=variant,
                rounds=args.rounds,
                n_candidates=args.n_candidates,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            row_report["variants"].append(variant_report)
            variant_aggregate[variant["name"]]["row_count"] += 1
            variant_aggregate[variant["name"]]["gold_visible_count"] += int(
                variant_report["gold_visible_in_candidates"]
            )
        report_rows.append(row_report)

    summary = []
    for variant in variants:
        agg = variant_aggregate[variant["name"]]
        row_count = max(agg["row_count"], 1)
        summary.append(
            {
                "variant": variant["name"],
                "profile": variant["profile"],
                "retrieval_mode": variant["retrieval_mode"],
                "hint_mode": variant["hint_mode"],
                "gold_visible_count": agg["gold_visible_count"],
                "row_count": agg["row_count"],
                "gold_visible_rate": round(agg["gold_visible_count"] / row_count, 4),
            }
        )

    report = {
        "generated_on": "2026-04-01",
        "focus_bundle": str(Path(args.focus_bundle)),
        "rounds": args.rounds,
        "n_candidates": args.n_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "summary": summary,
        "rows": report_rows,
    }

    output_path = Path(args.output_path) if args.output_path else WORK_LOGS_DIR / "task7_recall_probe_2026-04-01.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
