import argparse
import ast
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from main import TASK_FILES
from method import (
    CHAT_THINKING_TASKS,
    TASK_SAMPLE_ATTEMPTS,
    build_chat_examples,
    build_prompt,
    build_task5_retrieval_context,
    build_task7_retrieval_context,
    get_example_pool_limit,
    get_profile_name,
    get_profile_retrieval_tasks,
    parse_task7_fields,
    reorder_task5_examples_by_lexical_retrieval,
    reorder_task7_examples_by_lexical_retrieval,
    select_examples,
    solve_task_locally,
)
from method import annotate_chat_thinking
from method import annotate_nvidia as annotate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=[2, 5, 6, 7],
        help="Task ids to validate on holdout examples.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=40,
        help="Number of held-out examples per task.",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=None,
        help="Maximum number of ICL examples to consider. Defaults to the active profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for holdout sampling.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional local tokenizer/model path for prompt length checks.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=128000,
        help="Maximum prompt token length allowed before skipping a sample.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the validation report as JSON.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        nargs="*",
        default=["baseline"],
        help="Execution profiles to compare, e.g. baseline long_context.",
    )
    parser.add_argument(
        "--chat_tasks",
        type=int,
        nargs="*",
        default=sorted(CHAT_THINKING_TASKS),
        help="Tasks that should use chat+thinking in validation.",
    )
    parser.add_argument(
        "--retrieval_tasks",
        type=int,
        nargs="*",
        default=None,
        help="Optional override for retrieval tasks. If omitted, use the active profile defaults.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_qa_answer(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def build_retrieval_context(examples: list[dict], task_id: int) -> dict | None:
    if task_id == 5:
        return build_task5_retrieval_context(examples)
    if task_id == 7:
        return build_task7_retrieval_context(examples)
    return None


def reorder_examples_by_retrieval(
    examples: list[dict],
    text2annotate: str,
    task_id: int,
    retrieval_context: dict | None,
) -> list[dict]:
    if retrieval_context is None:
        return examples
    if task_id == 5:
        return reorder_task5_examples_by_lexical_retrieval(examples, text2annotate, retrieval_context)
    if task_id == 7:
        return reorder_task7_examples_by_lexical_retrieval(examples, text2annotate, retrieval_context)
    return examples


def resolve_retrieval_tasks(retrieval_tasks: list[int] | None, profile_name: str) -> set[int]:
    if retrieval_tasks is None:
        return get_profile_retrieval_tasks(profile_name)
    return set(retrieval_tasks)


def normalize_class_label(task_id: int, text: str) -> str:
    if task_id == 5:
        low = normalize_text(text)
        if "not sad" in low:
            return "not sad"
        if "sad" in low:
            return "sad"
    if task_id == 6:
        low = normalize_text(text)
        if low in {"y", "n"}:
            return low
    return normalize_text(text)


def score_prediction(task_id: int, prediction: str | None, gold: str) -> dict:
    gold = gold if isinstance(gold, str) else str(gold)
    if prediction is None:
        return {"score": 0.0, "reason": "null"}

    pred = str(prediction).strip()
    if task_id in {2, 5, 6, 7}:
        if task_id in {5, 6}:
            pred_norm = normalize_class_label(task_id, pred)
            gold_norm = normalize_class_label(task_id, gold)
        elif task_id == 7:
            pred_norm = normalize_qa_answer(pred)
            gold_norm = normalize_qa_answer(gold)
        else:
            pred_norm = normalize_text(pred)
            gold_norm = normalize_text(gold)
        return {
            "score": 1.0 if pred_norm == gold_norm else 0.0,
            "reason": "exact" if pred_norm == gold_norm else "mismatch",
        }

    if task_id == 8:
        pred_text = pred.strip()
        syntax_ok = False
        try:
            ast.parse(pred_text)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        expected_name = extract_wrapper_name(gold)
        has_wrapper = bool(expected_name and re.search(rf"def\s+{re.escape(expected_name)}\s*\(", pred_text))
        score = 0.0
        if syntax_ok:
            score += 0.6
        if has_wrapper:
            score += 0.4
        return {
            "score": score,
            "reason": f"syntax={syntax_ok},wrapper={has_wrapper}",
        }

    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    return {
        "score": 1.0 if pred_norm == gold_norm else 0.0,
        "reason": "exact" if pred_norm == gold_norm else "mismatch",
    }


def extract_wrapper_name(text: str) -> str | None:
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if match:
        return match.group(1)
    return None


def predict_example(
    task_id: int,
    sample: dict,
    icl_examples: list[dict],
    task_description: str,
    tokenizer,
    max_input_length: int,
    examples_limit: int,
    chat_examples_cache: dict[int, str],
    chat_tasks: set[int],
    retrieval_tasks: set[int],
    retrieval_context: dict | None,
    profile_name: str,
) -> str | None:
    text2annotate = sample["input"]

    local_prediction = solve_task_locally(task_id, text2annotate)
    if local_prediction is not None:
        return local_prediction

    if task_id in chat_tasks:
        if task_id in retrieval_tasks and retrieval_context is not None:
            reordered_examples = reorder_examples_by_retrieval(
                icl_examples,
                text2annotate,
                task_id,
                retrieval_context,
            )
            examples_str = build_chat_examples(
                reordered_examples,
                task_id=task_id,
                examples_limit=examples_limit,
                profile_name=profile_name,
            )
            return annotate_chat_thinking(
                examples_str,
                text2annotate,
                task_id,
                task_description,
                profile_name=profile_name,
            )

        cache_key = (task_id, profile_name)
        if cache_key not in chat_examples_cache:
            chat_examples_cache[cache_key] = build_chat_examples(
                icl_examples,
                task_id=task_id,
                examples_limit=examples_limit,
                profile_name=profile_name,
            )
        return annotate_chat_thinking(
            chat_examples_cache[cache_key],
            text2annotate,
            task_id,
            task_description,
            profile_name=profile_name,
        )

    prompt = build_prompt(task_description, text2annotate, task_id=task_id, profile_name=profile_name)
    reordered_examples = (
        reorder_examples_by_retrieval(icl_examples, text2annotate, task_id, retrieval_context)
        if task_id in retrieval_tasks
        else icl_examples
    )
    examples_str = select_examples(
        reordered_examples[:examples_limit],
        task_description,
        text2annotate,
        task_id=task_id,
        profile_name=profile_name,
    )
    input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")

    if tokenizer is not None:
        tokenized_input = tokenizer(input_prompt, return_tensors="pt")
        if tokenized_input["input_ids"].shape[1] > max_input_length:
            return None

    prediction = None
    for _ in range(TASK_SAMPLE_ATTEMPTS.get(task_id, 1)):
        prediction = annotate(input_prompt, task_id=task_id)
        if prediction is not None:
            break
    return prediction


def validate_task(
    task_id: int,
    sample_limit: int,
    examples_limit: int,
    seed: int,
    tokenizer,
    max_input_length: int,
    chat_tasks: set[int],
    retrieval_tasks: set[int],
    profile_name: str,
) -> dict:
    task_file = TASK_FILES[task_id]
    task_dict = json.loads(Path(task_file).read_text(encoding="utf-8"))
    all_examples = list(task_dict["examples"])
    resolved_examples_limit = get_example_pool_limit(
        task_id=task_id,
        examples_limit=examples_limit,
        profile_name=profile_name,
    )
    rng = random.Random(seed + task_id)
    holdout = rng.sample(all_examples, min(sample_limit, len(all_examples)))
    holdout_ids = {example["id"] for example in holdout}
    icl_pool = [example for example in all_examples if example["id"] not in holdout_ids]
    icl_pool = icl_pool[:resolved_examples_limit]
    retrieval_context = build_retrieval_context(icl_pool, task_id) if task_id in retrieval_tasks else None

    results = []
    chat_examples_cache = {}
    for sample in tqdm(holdout, desc=f"Holdout Task {task_id}", leave=False):
        gold = sample["output"][0]
        prediction = predict_example(
            task_id=task_id,
            sample=sample,
            icl_examples=icl_pool,
            task_description=task_dict["Definition"][0],
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            examples_limit=examples_limit,
            chat_examples_cache=chat_examples_cache,
            chat_tasks=chat_tasks,
            retrieval_tasks=retrieval_tasks,
            retrieval_context=retrieval_context,
            profile_name=profile_name,
        )
        scored = score_prediction(task_id, prediction, gold)
        results.append(
            {
                "id": sample["id"],
                "gold": gold,
                "prediction": prediction,
                "score": scored["score"],
                "reason": scored["reason"],
            }
        )

    avg_score = sum(item["score"] for item in results) / len(results) if results else 0.0
    mismatches = [item for item in results if item["score"] < 1.0]
    return {
        "task_id": task_id,
        "profile": profile_name,
        "task_name": task_dict["task_name"],
        "example_pool_limit": resolved_examples_limit,
        "sample_count": len(results),
        "avg_score": avg_score,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:10],
    }


def main():
    args = parse_args()
    tokenizer = None
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    report = {
        "tasks": args.tasks,
        "profiles": [get_profile_name(profile) for profile in args.profiles],
        "chat_tasks": sorted(args.chat_tasks),
        "retrieval_tasks": args.retrieval_tasks,
        "sample_limit": args.sample_limit,
        "examples_limit": args.examples_limit,
        "seed": args.seed,
        "results": [],
    }
    chat_tasks = set(args.chat_tasks)
    for profile_name in report["profiles"]:
        retrieval_tasks = resolve_retrieval_tasks(args.retrieval_tasks, profile_name)
        for task_id in args.tasks:
            report["results"].append(
                validate_task(
                    task_id=task_id,
                    sample_limit=args.sample_limit,
                    examples_limit=args.examples_limit,
                    seed=args.seed,
                    tokenizer=tokenizer,
                    max_input_length=args.max_input_length,
                    chat_tasks=chat_tasks,
                    retrieval_tasks=retrieval_tasks,
                    profile_name=profile_name,
                )
            )
    report["resolved_retrieval_tasks_by_profile"] = {
        profile_name: sorted(resolve_retrieval_tasks(args.retrieval_tasks, profile_name))
        for profile_name in report["profiles"]
    }

    scored_tasks = [task["avg_score"] for task in report["results"]]
    report["macro_avg_score"] = sum(scored_tasks) / len(scored_tasks) if scored_tasks else 0.0
    report["macro_avg_score_by_profile"] = {}
    for profile_name in report["profiles"]:
        profile_scores = [task["avg_score"] for task in report["results"] if task["profile"] == profile_name]
        report["macro_avg_score_by_profile"][profile_name] = (
            sum(profile_scores) / len(profile_scores) if profile_scores else 0.0
        )

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
