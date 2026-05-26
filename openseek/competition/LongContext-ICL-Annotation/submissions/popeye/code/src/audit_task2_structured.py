import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from main import TASK_FILES
from method import get_task2_tagger, parse_task2_fields, solve_task2_structured_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--sample_limit", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 123])
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def classify_task2_gap(text2annotate: str, gold: str, prediction: str | None) -> dict:
    parsed = parse_task2_fields(text2annotate)
    gold_int = int(gold) if gold is not None else None
    pred_int = int(prediction) if prediction is not None and str(prediction).isdigit() else None
    tokenizer, pos_tag = get_task2_tagger()

    if parsed is None:
        return {
            "error_type": "parse_failure",
            "priority_bucket": "high_value_low_risk",
            "rationale": "The structured solver could not parse the fixed task2 template.",
        }

    sentence, target = parsed
    if tokenizer is None or pos_tag is None:
        return {
            "error_type": "parse_failure",
            "priority_bucket": "unknown_other",
            "rationale": "The task2 POS tagger is unavailable, so the structured solver cannot run deterministically.",
        }

    try:
        tokens = tokenizer.tokenize(sentence)
        tags = pos_tag(tokens)
    except Exception:
        return {
            "error_type": "parse_failure",
            "priority_bucket": "unknown_other",
            "rationale": "The task2 POS tagging call failed for this sentence.",
        }

    token_lows = [token.lower() for token in tokens]
    joined_low = " ".join(token_lows)
    has_quotes = any(mark in sentence for mark in ['"', "''", "``"])
    has_hyphen = "-" in sentence or "multi color" in joined_low or "hotdog" in joined_low
    has_apostrophe = "'" in sentence or "n't" in joined_low

    verb_like_tokens = [word.lower() for word, tag in tags if tag.startswith("VB")]
    noun_like_tokens = [word.lower() for word, tag in tags if tag in {"NN", "NNS"}]
    has_vbg = any(tag == "VBG" for _, tag in tags)
    has_vbz = any(tag == "VBZ" for _, tag in tags)

    if target == "nouns":
        if has_quotes:
            return {
                "error_type": "quotation_or_title_token_gap",
                "priority_bucket": "high_value_low_risk",
                "rationale": "Quoted title-like spans can change whether words are counted as standalone nouns.",
            }
        if has_hyphen:
            return {
                "error_type": "punctuation_or_hyphen_gap",
                "priority_bucket": "high_value_low_risk",
                "rationale": "Hyphenation or compound-token formatting likely changes the intended noun counting rule.",
            }
        if has_apostrophe:
            return {
                "error_type": "contraction_or_apostrophe_gap",
                "priority_bucket": "medium_value_verifiable",
                "rationale": "Apostrophes can alter token boundaries and the counting rule for noun phrases.",
            }
        if pred_int is not None and gold_int is not None and abs(pred_int - gold_int) == 1:
            if any(tag in {"NNP", "NNPS"} for _, tag in tags) or any(token.istitle() for token in tokens):
                return {
                    "error_type": "quotation_or_title_token_gap",
                    "priority_bucket": "medium_value_verifiable",
                    "rationale": "Proper-name or title-style tokens likely explain a one-count noun mismatch.",
                }
            return {
                "error_type": "count_rule_gap",
                "priority_bucket": "medium_value_verifiable",
                "rationale": "The sentence parses, but the noun counting convention differs from the current rule set.",
            }
        return {
            "error_type": "unknown_other",
            "priority_bucket": "defer",
            "rationale": "No specific deterministic noun-count rule gap matched this error.",
        }

    if target == "verbs":
        if has_apostrophe:
            return {
                "error_type": "contraction_or_apostrophe_gap",
                "priority_bucket": "medium_value_verifiable",
                "rationale": "Apostrophes or contractions can change how verbal forms are split and counted.",
            }
        if has_quotes:
            return {
                "error_type": "quotation_or_title_token_gap",
                "priority_bucket": "medium_value_verifiable",
                "rationale": "Quoted spans can cause title-like text to be counted incorrectly as verb material.",
            }
        if has_hyphen:
            return {
                "error_type": "punctuation_or_hyphen_gap",
                "priority_bucket": "medium_value_verifiable",
                "rationale": "Hyphenation or compound formatting can affect whether verbal tokens are split correctly.",
            }
        if pred_int == 0 and gold_int and gold_int > 0 and (has_vbg or has_vbz):
            return {
                "error_type": "tokenization_gap",
                "priority_bucket": "high_value_low_risk",
                "rationale": "The tagger saw verb-like surface forms, but the current structured counter filtered them out.",
            }
        if any(word in {"has", "have", "had", "is", "are", "was", "were"} for word in verb_like_tokens):
            return {
                "error_type": "count_rule_gap",
                "priority_bucket": "high_value_low_risk",
                "rationale": "The dataset appears to count some auxiliary/copular verbs differently from the current heuristic.",
            }
        if has_vbg or has_vbz:
            return {
                "error_type": "count_rule_gap",
                "priority_bucket": "medium_value_verifiable",
                "rationale": "Gerund or present-tense forms suggest a label-rule mismatch rather than a parse failure.",
            }
        return {
            "error_type": "unknown_other",
            "priority_bucket": "defer",
            "rationale": "No specific deterministic verb-count rule gap matched this error.",
        }

    return {
        "error_type": "unknown_other",
        "priority_bucket": "defer",
        "rationale": "Unexpected task2 target label.",
    }


def evaluate_samples(samples: list[dict]) -> dict:
    mismatches = []
    correct = 0
    parse_failure_count = 0
    for sample in samples:
        prediction = solve_task2_structured_count(sample["input"])
        gold = sample["output"][0]
        ok = prediction == gold
        correct += int(ok)
        if not ok:
            gap_info = classify_task2_gap(sample["input"], gold, prediction)
            if gap_info["error_type"] == "parse_failure":
                parse_failure_count += 1
            mismatches.append(
                {
                    "id": sample["id"],
                    "gold": gold,
                    "prediction": prediction,
                    "input": sample["input"],
                    "error_type": gap_info["error_type"],
                    "priority_bucket": gap_info["priority_bucket"],
                    "rationale": gap_info["rationale"],
                }
            )
    total = len(samples)
    error_type_counter = Counter(row["error_type"] for row in mismatches)
    priority_counter = Counter(row["priority_bucket"] for row in mismatches)
    return {
        "sample_count": total,
        "avg_score": correct / total if total else 0.0,
        "parse_status": {
            "parse_success": total - parse_failure_count,
            "parse_failure": parse_failure_count,
        },
        "mismatch_count": len(mismatches),
        "error_type_counter": dict(error_type_counter),
        "error_type_ratio": {
            key: round(value / len(mismatches), 6) for key, value in error_type_counter.items()
        } if mismatches else {},
        "priority_bucket_counter": dict(priority_counter),
        "priority_bucket_ratio": {
            key: round(value / len(mismatches), 6) for key, value in priority_counter.items()
        } if mismatches else {},
        "mismatch_examples": mismatches[:10],
        "all_mismatches": mismatches,
    }


def main():
    args = parse_args()
    task_dict = json.loads(Path(TASK_FILES[args.task_id]).read_text(encoding="utf-8"))
    examples = list(task_dict["examples"])

    report = {
        "task_id": args.task_id,
        "task_name": task_dict["task_name"],
        "full_examples": evaluate_samples(examples),
        "holdout_runs": [],
    }

    for seed in args.seeds:
        rng = random.Random(seed + args.task_id)
        holdout = rng.sample(examples, min(args.sample_limit, len(examples)))
        run_report = evaluate_samples(holdout)
        run_report["seed"] = seed
        report["holdout_runs"].append(run_report)

    aggregate_error_counter = Counter(report["full_examples"]["error_type_counter"])
    aggregate_priority_counter = Counter(report["full_examples"]["priority_bucket_counter"])
    for run in report["holdout_runs"]:
        aggregate_error_counter.update(run["error_type_counter"])
        aggregate_priority_counter.update(run["priority_bucket_counter"])

    total_mismatches = sum(aggregate_error_counter.values())
    priority_summary = {
        "high_value_low_risk": [],
        "medium_value_verifiable": [],
        "defer": [],
    }
    seen = set()
    for section in [report["full_examples"], *report["holdout_runs"]]:
        for row in section["all_mismatches"]:
            mismatch_id = row["id"]
            if mismatch_id in seen:
                continue
            seen.add(mismatch_id)
            bucket = row["priority_bucket"]
            if bucket in priority_summary and len(priority_summary[bucket]) < 8:
                priority_summary[bucket].append(
                    {
                        "id": row["id"],
                        "error_type": row["error_type"],
                        "gold": row["gold"],
                        "prediction": row["prediction"],
                        "input": row["input"],
                    }
                )

    report["aggregate"] = {
        "error_type_counter": dict(aggregate_error_counter),
        "error_type_ratio": {
            key: round(value / total_mismatches, 6) for key, value in aggregate_error_counter.items()
        } if total_mismatches else {},
        "priority_bucket_counter": dict(aggregate_priority_counter),
        "priority_bucket_ratio": {
            key: round(value / total_mismatches, 6) for key, value in aggregate_priority_counter.items()
        } if total_mismatches else {},
        "error_type_coverage": total_mismatches,
        "priority_gap_recommendations": {
            "high_value_low_risk": priority_summary["high_value_low_risk"],
            "medium_value_verifiable": priority_summary["medium_value_verifiable"],
            "defer": priority_summary["defer"],
        },
    }

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
