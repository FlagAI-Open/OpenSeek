import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from main import TASK_FILES
from method import build_prompt, select_examples
from method import annotate_nvidia as annotate
from task8_retrieval import (
    build_task8_lexical_retrieval_context,
    build_task8_retrieval_context,
    extract_task8_features,
    reorder_task8_examples_by_family_lexical_retrieval,
    reorder_task8_examples_by_lexical_retrieval,
    reorder_task8_examples_by_retrieval,
)
from task8_semantics import score_task8_proxy_v1, score_task8_proxy_v2


TASK_ID = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_limit", type=int, default=8, help="Number of held-out examples to score.")
    parser.add_argument("--examples_limit", type=int, default=100, help="Maximum examples available for selection.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout split.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path for prompt length checks.")
    parser.add_argument("--max_input_length", type=int, default=128000, help="Maximum prompt token length.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional JSON report output path.")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="none",
        choices=["none", "lexical", "family_lexical", "structured", "hybrid"],
        help="Example retrieval strategy used before prompt construction.",
    )
    parser.add_argument("--retrieval", action="store_true", help="Deprecated alias for --retrieval_mode lexical.")
    return parser.parse_args()
def predict_example(
    sample: dict,
    icl_examples: list[dict],
    task_description: str,
    tokenizer,
    max_input_length: int,
    examples_limit: int,
    retrieval_mode: str,
    retrieval_context: dict | None,
) -> str | None:
    text2annotate = sample["input"]
    prompt = build_prompt(task_description, text2annotate, task_id=TASK_ID)
    if retrieval_mode in {"structured", "hybrid"}:
        ordered_examples = reorder_task8_examples_by_retrieval(
            icl_examples,
            text2annotate,
            retrieval_context,
            mode=retrieval_mode,
        )
    elif retrieval_mode == "family_lexical":
        ordered_examples = reorder_task8_examples_by_family_lexical_retrieval(
            icl_examples,
            text2annotate,
            retrieval_context,
        )
    else:
        ordered_examples = reorder_task8_examples_by_lexical_retrieval(icl_examples, text2annotate, retrieval_context)
    examples_str = select_examples(
        ordered_examples[:examples_limit],
        task_description,
        text2annotate,
        task_id=TASK_ID,
    )
    input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")

    if tokenizer is not None:
        tokenized = tokenizer(input_prompt, return_tensors="pt")
        if tokenized["input_ids"].shape[1] > max_input_length:
            return None

    return annotate(input_prompt, task_id=TASK_ID)


def main():
    args = parse_args()
    if args.retrieval and args.retrieval_mode == "none":
        args.retrieval_mode = "lexical"

    task_dict = json.loads(TASK_FILES[TASK_ID].read_text(encoding="utf-8"))
    all_examples = task_dict["examples"]
    task_description = task_dict["Definition"][0]

    rng = random.Random(args.seed + TASK_ID)
    holdout = rng.sample(all_examples, min(args.sample_limit, len(all_examples)))
    holdout_ids = {sample["id"] for sample in holdout}
    icl_pool = [example for example in all_examples if example["id"] not in holdout_ids]
    icl_pool = icl_pool[: args.examples_limit]

    tokenizer = None
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    retrieval_context = None
    if args.retrieval_mode in {"lexical", "family_lexical"}:
        retrieval_context = build_task8_lexical_retrieval_context(icl_pool)
    elif args.retrieval_mode in {"structured", "hybrid"}:
        retrieval_context = build_task8_retrieval_context(icl_pool)

    rows = []
    scores_v1 = []
    scores_v2 = []
    family_breakdown = defaultdict(
        lambda: {
            "count": 0,
            "proxy_v1_total": 0.0,
            "proxy_v2_total": 0.0,
            "hard_gate_total": 0.0,
            "family_total": 0.0,
            "struct_total": 0.0,
        }
    )
    for sample in tqdm(holdout, desc="Task 8 Proxy Holdout"):
        prediction = predict_example(
            sample=sample,
            icl_examples=icl_pool,
            task_description=task_description,
            tokenizer=tokenizer,
            max_input_length=args.max_input_length,
            examples_limit=args.examples_limit,
            retrieval_mode=args.retrieval_mode,
            retrieval_context=retrieval_context,
        )
        gold = sample["output"][0]
        scored_v1 = score_task8_proxy_v1(prediction, gold)
        scored_v2 = score_task8_proxy_v2(sample["input"], prediction, gold=gold)
        scores_v1.append(scored_v1["score"])
        scores_v2.append(scored_v2["score"])
        input_family = extract_task8_features(sample["input"])["primary_family"] or "unknown"
        family_stats = family_breakdown[input_family]
        family_stats["count"] += 1
        family_stats["proxy_v1_total"] += scored_v1["score"]
        family_stats["proxy_v2_total"] += scored_v2["score"]
        family_stats["hard_gate_total"] += scored_v2["hard_gate_score"]
        family_stats["family_total"] += scored_v2["family_semantics_score"]
        family_stats["struct_total"] += scored_v2["structural_similarity_score"]
        rows.append(
            {
                "id": sample["id"],
                "prediction": prediction,
                "input_family": input_family,
                "gold_wrapper": scored_v2["audit"]["expected_wrapper"],
                "pred_wrapper": scored_v2["audit"]["wrapper_name"],
                "proxy_v1_score": scored_v1["score"],
                "proxy_v1_reason": scored_v1["reason"],
                "proxy_v2_score": scored_v2["score"],
                "proxy_v2_reason": scored_v2["reason"],
                "hard_gate_score": scored_v2["hard_gate_score"],
                "family_semantics_score": scored_v2["family_semantics_score"],
                "structural_similarity_score": scored_v2["structural_similarity_score"],
                "blocker_count": scored_v2["audit"]["blocker_count"],
                "warning_count": scored_v2["audit"]["warning_count"],
                "audit": scored_v2["audit"],
                "family": scored_v2["family"],
                "structural": scored_v2["structural"],
            }
        )

    family_report = {}
    for family, stats in family_breakdown.items():
        count = max(stats["count"], 1)
        family_report[family] = {
            "count": stats["count"],
            "avg_proxy_v1": round(stats["proxy_v1_total"] / count, 4),
            "avg_proxy_v2": round(stats["proxy_v2_total"] / count, 4),
            "avg_hard_gate": round(stats["hard_gate_total"] / count, 4),
            "avg_family_semantics": round(stats["family_total"] / count, 4),
            "avg_structural_similarity": round(stats["struct_total"] / count, 4),
        }

    report = {
        "task_id": TASK_ID,
        "sample_limit": args.sample_limit,
        "examples_limit": args.examples_limit,
        "seed": args.seed,
        "retrieval_mode": args.retrieval_mode,
        "avg_proxy_score": round(sum(scores_v2) / max(len(scores_v2), 1), 4),
        "avg_proxy_v1_score": round(sum(scores_v1) / max(len(scores_v1), 1), 4),
        "avg_proxy_v2_score": round(sum(scores_v2) / max(len(scores_v2), 1), 4),
        "family_breakdown": family_report,
        "results": rows,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
