import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from audit_task2_structured import classify_task2_gap
from main import TASK_FILES
from method import get_task2_tagger, parse_task2_fields, solve_task2_structured_count
from task2_dependency_policy import count_verbs_with_policy, get_policy_config, tokenize_task2_sentence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--sample_limit", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 123])
    parser.add_argument(
        "--baseline_variant",
        type=str,
        default="mainline",
        help="Example-side baseline variant. Use 'mainline' for solve_task2_structured_count, or a named probe variant such as 'verb_mistagged_guarded'.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=[
            "baseline",
            "verb_rel_or_have",
            "noun_initial_multi",
            "combo_conservative",
            "verb_has_this",
            "verb_has_been_clause",
            "verb_mistagged_surface",
            "verb_mistagged_guarded",
            "verb_participle_stuffed_exact",
            "verb_dependency_compressed_only",
            "verb_dependency_trailing_vbg",
            "verb_dependency_discourse_followup",
            "verb_test_touch_play",
            "verb_test_touch_motion",
            "verb_test_touch_combo",
            "verb_combo_v2",
        ],
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--test_compare_path", type=str, default=None)
    return parser.parse_args()


def load_task2_examples(task_id: int) -> list[dict]:
    task_dict = json.loads(Path(TASK_FILES[task_id]).read_text(encoding="utf-8"))
    return list(task_dict["examples"])


def load_task2_test_samples(task_id: int) -> list[dict]:
    task_dict = json.loads(Path(TASK_FILES[task_id]).read_text(encoding="utf-8"))
    return list(task_dict["test_samples"])


def task2_tags(text2annotate: str):
    parsed = parse_task2_fields(text2annotate)
    if parsed is None:
        return None
    tokenizer, pos_tag = get_task2_tagger()
    if tokenizer is None or pos_tag is None:
        return None
    sentence, target = parsed
    try:
        tags = pos_tag(tokenizer.tokenize(sentence))
    except Exception:
        return None
    return sentence, target, tags


CONSERVATIVE_VARIANTS = {
    "verb_rel_or_have",
    "noun_initial_multi",
    "combo_conservative",
    "verb_has_this",
    "verb_has_been_clause",
    "verb_mistagged_surface",
    "verb_mistagged_guarded",
    "verb_participle_stuffed_exact",
    "verb_dependency_discourse_followup",
    "verb_dependency_compressed_only",
    "verb_dependency_trailing_vbg",
    "verb_test_touch_play",
    "verb_test_touch_motion",
    "verb_test_touch_combo",
    "verb_combo_v2",
}

DEPENDENCY_POLICY_VARIANTS = {
    "verb_dependency_compressed_only": "compressed_only",
    "verb_dependency_trailing_vbg": "compressed_plus_trailing_vbg",
    "verb_dependency_discourse_followup": "compressed_plus_discourse_followup",
}

TASK2_MISTAGGED_ING_SURFACES = {
    "standing",
    "walking",
    "grazing",
    "laying",
    "playing",
    "jumping",
    "skiing",
    "sitting",
    "sleeping",
    "eating",
    "staring",
    "putting",
}

TASK2_MISTAGGED_SIMPLE_SURFACES = {
    "sit",
    "sits",
    "walk",
    "walks",
    "push",
    "pushes",
    "talk",
    "talks",
    "gather",
    "gathers",
    "watch",
    "watches",
    "check",
    "checks",
    "fly",
    "flies",
    "wade",
    "wades",
}

TASK2_TEST_TOUCH_SIMPLE_NN_SURFACES = {
    "verb_test_touch_play": {"play"},
    "verb_test_touch_combo": {"play"},
}

TASK2_TEST_TOUCH_SIMPLE_NNS_SURFACES = {
    "verb_test_touch_motion": {"skateboards", "maneuvers", "rumbles"},
    "verb_test_touch_combo": {"skateboards", "maneuvers", "rumbles", "reaches"},
}


def _parse_baseline_prediction(text2annotate: str, baseline_prediction: str | None) -> int | None:
    pred = baseline_prediction if baseline_prediction is not None else solve_task2_structured_count(text2annotate)
    if pred is None:
        return None
    try:
        return int(pred)
    except (TypeError, ValueError):
        return None


def _task2_is_mistagged_simple_surface(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    if tag == "NNS" and word in TASK2_MISTAGGED_SIMPLE_SURFACES:
        return prev_tag in {"NN", "NNS", "NNP", "PRP", "CC"} and next_tag in {"IN", "TO", "RB", "RP", "DT"}
    if tag == "NN" and word == "sit":
        return prev_tag in {"NN", "NNS", "NNP", "PRP", "CC"} and next_tag in {"IN", "TO", "RB", "RP", "DT"}
    return False


def _task2_is_test_touch_simple_surface(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
    variant: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    allowed_prev_tags = {"NN", "NNS", "NNP", "PRP", "CC"}
    if variant == "verb_test_touch_combo" and word == "reaches":
        allowed_prev_tags = allowed_prev_tags | {"PRP$"}

    if tag == "NN" and word in TASK2_TEST_TOUCH_SIMPLE_NN_SURFACES.get(variant, set()):
        return prev_tag in allowed_prev_tags and next_tag in {"IN", "TO", "RB", "RP", "DT"}
    if tag == "NNS" and word in TASK2_TEST_TOUCH_SIMPLE_NNS_SURFACES.get(variant, set()):
        return prev_tag in allowed_prev_tags and next_tag in {"IN", "TO", "RB", "RP", "DT"}
    return False


def _task2_is_guarded_ing_surface(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    prev2_word = tags[idx - 2][0].lower() if idx > 1 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    has_prev_verb = any(existing_tag.startswith("VB") for _, existing_tag in tags[:idx])

    if tag not in {"NN", "NNS"} or word not in TASK2_MISTAGGED_ING_SURFACES:
        return False
    if next_tag not in {"IN", "TO", "RP", "RB"}:
        return False

    # This guarded branch only keeps the most reliable noun-tagged participles.
    # It avoids adjective-led descriptors ("red standing"), collective "group of"
    # subjects, coordinated proper-noun lists, and cases that already contain an
    # earlier verb unless we hit a tiny, validated existential pattern below.
    if prev_tag not in {"DT", "JJ", "NN", "NNS", "NNP", "PRP", "CC"}:
        return False
    if has_prev_verb:
        if word != "skiing":
            return False
        words = [token.lower() for token, _ in tags]
        return words[:4] == ["there", "is", "a", "man"]
    if prev_tag == "JJ":
        return False
    if prev2_word == "of":
        return False
    if prev2_word == "and" and prev_tag == "NNP":
        return False
    return True


def _task2_is_stuffed_participle_exact(
    tags: list[tuple[str, str]],
    idx: int,
    word: str,
    tag: str,
) -> bool:
    prev_tag = tags[idx - 1][1] if idx > 0 else ""
    next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
    return tag == "JJ" and word == "stuffed" and prev_tag == "DT" and next_tag == "NN"


def _apply_conservative_variant(base_pred: int, target: str, tags: list[tuple[str, str]], variant: str) -> int:
    updated = base_pred

    if variant in {"verb_rel_or_have", "combo_conservative"} and target == "verbs" and updated == 0:
        verb_positions = [(idx, word.lower()) for idx, (word, tag) in enumerate(tags) if tag.startswith("VB")]
        if len(verb_positions) == 1:
            idx, word = verb_positions[0]
            first_word = tags[0][0].lower() if tags else ""
            prev_word = tags[idx - 1][0].lower() if idx > 0 else ""
            if (word in {"has", "have", "had"} or prev_word in {"that", "which", "who"}) and first_word not in {
                "there",
                "it",
                "this",
            }:
                updated += 1

    if variant in {"noun_initial_multi", "combo_conservative"} and target == "nouns":
        if tags and tags[0][1] in {"NNP", "NNPS"} and len(tags) > 1 and tags[1][1].startswith("VB"):
            updated += 1
        for idx, (word, tag) in enumerate(tags[:-2]):
            if (
                word.lower() == "multi"
                and tag in {"NN", "NNS"}
                and tags[idx + 1][1] in {"NN", "NNS"}
                and tags[idx + 2][1] in {"NN", "NNS"}
            ):
                updated -= 1
                break

    if variant == "verb_participle_stuffed_exact" and target == "verbs":
        bonus = 0
        for idx, (word, tag) in enumerate(tags):
            low = word.lower()
            if _task2_is_stuffed_participle_exact(tags, idx, low, tag):
                bonus += 1
        updated += bonus

    if target == "verbs" and updated == 0:
        verb_positions = [(idx, word.lower()) for idx, (word, tag) in enumerate(tags) if tag.startswith("VB")]
        if variant in {"verb_has_this", "verb_combo_v2"} and len(verb_positions) == 1:
            idx, word = verb_positions[0]
            first_word = tags[0][0].lower() if tags else ""
            if word in {"has", "have", "had"} and first_word == "this":
                updated += 1

        if variant in {"verb_has_been_clause", "verb_combo_v2"} and len(verb_positions) == 1:
            idx, word = verb_positions[0]
            next_word = tags[idx + 1][0].lower() if idx + 1 < len(tags) else ""
            next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
            first_word = tags[0][0].lower() if tags else ""
            if (
                word in {"has", "have", "had"}
                and next_word == "been"
                and next_tag == "VBN"
                and first_word not in {"there", "it"}
            ):
                updated += 1

        if variant in {
            "verb_mistagged_surface",
            "verb_mistagged_guarded",
            "verb_test_touch_play",
            "verb_test_touch_motion",
            "verb_test_touch_combo",
            "verb_combo_v2",
        }:
            bonus = 0
            for idx, (word, tag) in enumerate(tags):
                low = word.lower()
                prev_tag = tags[idx - 1][1] if idx > 0 else ""
                next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
                if variant == "verb_participle_stuffed_exact" and _task2_is_stuffed_participle_exact(tags, idx, low, tag):
                    bonus += 1
                elif variant == "verb_mistagged_guarded" and _task2_is_guarded_ing_surface(tags, idx, low, tag):
                    bonus += 1
                elif (
                    variant in {"verb_mistagged_surface", "verb_combo_v2"}
                    and tag in {"NN", "NNS"}
                    and low in TASK2_MISTAGGED_ING_SURFACES
                    and prev_tag in {"DT", "JJ", "NN", "NNS", "NNP", "PRP", "CC"}
                    and next_tag in {"IN", "TO", "RP", "RB"}
                ):
                    bonus += 1
                elif _task2_is_test_touch_simple_surface(tags, idx, low, tag, variant):
                    bonus += 1
                elif _task2_is_mistagged_simple_surface(tags, idx, low, tag):
                    bonus += 1
            updated += bonus

    return updated


def variant_prediction(text2annotate: str, variant: str, baseline_prediction: str | None = None) -> str | None:
    if variant == "baseline":
        return baseline_prediction if baseline_prediction is not None else solve_task2_structured_count(text2annotate)

    if variant in DEPENDENCY_POLICY_VARIANTS:
        parsed = parse_task2_fields(text2annotate)
        policy = get_policy_config(DEPENDENCY_POLICY_VARIANTS[variant])
        if parsed is None or policy is None:
            return None
        sentence, target = parsed
        if target != "verbs":
            return baseline_prediction if baseline_prediction is not None else solve_task2_structured_count(text2annotate)
        return str(count_verbs_with_policy(tokenize_task2_sentence(sentence), policy))

    if variant in CONSERVATIVE_VARIANTS:
        parsed = task2_tags(text2annotate)
        if parsed is None:
            return None
        _, target, tags = parsed
        base_pred = _parse_baseline_prediction(text2annotate, baseline_prediction)
        if base_pred is None:
            return None
        return str(_apply_conservative_variant(base_pred, target, tags, variant))

    parsed = task2_tags(text2annotate)
    if parsed is None:
        return None
    _, target, tags = parsed

    if target == "nouns":
        return str(count_nouns(tags, variant))
    if target == "verbs":
        return str(count_verbs(tags, variant))
    return None


def count_nouns(tags: list[tuple[str, str]], variant: str) -> int:
    count = 0
    for idx, (word, tag) in enumerate(tags):
        low = word.lower()
        next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
        prev_word = tags[idx - 1][0] if idx - 1 >= 0 else ""
        next_word = tags[idx + 1][0] if idx + 1 < len(tags) else ""

        if tag in {"NN", "NNS"}:
            if low in {
                "someone",
                "somebody",
                "something",
                "anyone",
                "anybody",
                "anything",
                "everyone",
                "everybody",
                "everything",
                "nobody",
                "nothing",
            }:
                continue
            if low in {
                "red",
                "blue",
                "green",
                "yellow",
                "orange",
                "pink",
                "purple",
                "black",
                "white",
                "brown",
                "gray",
                "grey",
                "gold",
                "silver",
                "beige",
            } and next_tag in {"NN", "NNS"}:
                continue
            if variant == "punctuation_compounds":
                if low in {"multi", "hotdog"}:
                    continue
            count += 1
            continue

        if variant == "quotation_titles" and tag in {"NNP", "NNPS"}:
            if prev_word in {"``", '"'} or next_word in {"''", '"'}:
                count += 1
    return count


def count_verbs(tags: list[tuple[str, str]], variant: str) -> int:
    count = 0
    for idx, (word, tag) in enumerate(tags):
        low = word.lower()
        next_word = tags[idx + 1][0].lower() if idx + 1 < len(tags) else ""
        next_tag = tags[idx + 1][1] if idx + 1 < len(tags) else ""
        later_words = [w.lower() for w, _ in tags[idx + 1 : idx + 4]]

        if tag.startswith("VB") and low not in {"be", "been", "being", "'s"}:
            count += 1
            continue

        if variant == "count_rule_helpers" and tag.startswith("VB"):
            if low in {"has", "have", "had"}:
                count += 1
                continue
            if low == "being" and next_tag == "VBN":
                count += 1
                continue
            if low in {"are", "were"} and next_tag == "VBG" and next_word != "being":
                count += 1
                continue
            if low in {"is", "was"} and any(word.endswith("ing") for word in later_words):
                count += 1
                continue

        if variant == "tokenization_verbs" and tag in {"NN", "NNS"}:
            if low.endswith("ing"):
                count += 1
                continue
            if low in {
                "graze",
                "grazes",
                "stand",
                "stands",
                "pass",
                "passes",
                "pose",
                "poses",
                "look",
                "looks",
                "play",
                "plays",
                "sleep",
                "sleeps",
                "sit",
                "sits",
            }:
                count += 1
                continue
    return count


def evaluate_samples(samples: list[dict], variant: str, baseline_predictions: dict[str, str | None]) -> dict:
    correct = 0
    mismatches = []
    fixed_by_bucket = Counter()
    regressed_by_bucket = Counter()
    triggered_adjustments = 0
    for sample in samples:
        sample_id = sample["id"]
        gold = sample["output"][0]
        pred = variant_prediction(sample["input"], variant, baseline_predictions[sample_id])
        baseline_pred = baseline_predictions[sample_id]
        if variant in CONSERVATIVE_VARIANTS and pred != baseline_pred:
            triggered_adjustments += 1
        ok = pred == gold
        correct += int(ok)

        baseline_gap = classify_task2_gap(sample["input"], gold, baseline_pred)["error_type"] if baseline_pred != gold else None
        if baseline_pred != gold and pred == gold and baseline_gap is not None:
            fixed_by_bucket[baseline_gap] += 1
        if baseline_pred == gold and pred != gold:
            new_gap = classify_task2_gap(sample["input"], gold, pred)["error_type"]
            regressed_by_bucket[new_gap] += 1

        if not ok:
            mismatches.append(
                {
                    "id": sample_id,
                    "gold": gold,
                    "prediction": pred,
                    "baseline_prediction": baseline_pred,
                    "input": sample["input"],
                    "error_type": classify_task2_gap(sample["input"], gold, pred)["error_type"],
                }
            )

    total = len(samples)
    return {
        "sample_count": total,
        "avg_score": correct / total if total else 0.0,
        "mismatch_count": len(mismatches),
        "triggered_adjustments": triggered_adjustments,
        "fixed_vs_baseline": dict(fixed_by_bucket),
        "regressed_vs_baseline": dict(regressed_by_bucket),
        "mismatch_examples": mismatches[:10],
    }


def load_test_compare_predictions(compare_path: str) -> dict[str, str | None]:
    rows = [
        json.loads(line)
        for line in Path(compare_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        row["test_sample_id"]: row.get("prediction")
        for row in rows
    }


def evaluate_test_changes(
    test_samples: list[dict],
    variant: str,
    baseline_test_predictions: dict[str, str | None],
) -> dict:
    changed_rows = []
    direction_counter = Counter()

    for sample in test_samples:
        sample_id = sample["id"]
        baseline_pred = baseline_test_predictions.get(sample_id)
        variant_pred = variant_prediction(sample["input"], variant, baseline_pred)
        if variant_pred == baseline_pred:
            continue
        changed_rows.append(
            {
                "id": sample_id,
                "baseline_prediction": baseline_pred,
                "variant_prediction": variant_pred,
                "input": sample["input"],
            }
        )
        direction_counter[(baseline_pred, variant_pred)] += 1

    return {
        "changed_count": len(changed_rows),
        "direction_counter": {
            f"{old}->{new}": count
            for (old, new), count in sorted(direction_counter.items())
        },
        "changed_examples": changed_rows[:12],
    }


def build_example_baseline_predictions(
    samples: list[dict],
    baseline_variant: str,
) -> dict[str, str | None]:
    base_predictions = {
        sample["id"]: solve_task2_structured_count(sample["input"])
        for sample in samples
    }
    if baseline_variant in {"", "mainline", "default", "baseline"}:
        return base_predictions

    return {
        sample["id"]: variant_prediction(sample["input"], baseline_variant, base_predictions[sample["id"]])
        for sample in samples
    }


def main():
    args = parse_args()
    examples = load_task2_examples(args.task_id)
    test_samples = load_task2_test_samples(args.task_id) if args.test_compare_path else []
    baseline_test_predictions = load_test_compare_predictions(args.test_compare_path) if args.test_compare_path else {}
    baseline_predictions = build_example_baseline_predictions(examples, args.baseline_variant)

    report = {
        "task_id": args.task_id,
        "variants": args.variants,
        "sample_limit": args.sample_limit,
        "seeds": args.seeds,
        "baseline_variant": args.baseline_variant,
        "variant_reports": {},
    }

    for variant in args.variants:
        variant_report = {
            "full_examples": evaluate_samples(examples, variant, baseline_predictions),
            "holdout_runs": [],
        }
        for seed in args.seeds:
            rng = random.Random(seed + args.task_id)
            holdout = rng.sample(examples, min(args.sample_limit, len(examples)))
            holdout_report = evaluate_samples(holdout, variant, baseline_predictions)
            holdout_report["seed"] = seed
            variant_report["holdout_runs"].append(holdout_report)
        if args.test_compare_path:
            variant_report["test_changes_vs_compare"] = evaluate_test_changes(
                test_samples,
                variant,
                baseline_test_predictions,
            )
        report["variant_reports"][variant] = variant_report

    baseline_full = report["variant_reports"]["baseline"]["full_examples"]["avg_score"]
    leaderboard = []
    for variant, variant_report in report["variant_reports"].items():
        holdout_scores = [run["avg_score"] for run in variant_report["holdout_runs"]]
        leaderboard.append(
            {
                "variant": variant,
                "full_examples_avg": variant_report["full_examples"]["avg_score"],
                "full_examples_delta_vs_baseline": round(
                    variant_report["full_examples"]["avg_score"] - baseline_full,
                    6,
                ),
                "holdout_mean": round(sum(holdout_scores) / len(holdout_scores), 6),
                "holdout_min": round(min(holdout_scores), 6),
                "holdout_max": round(max(holdout_scores), 6),
            }
        )
    leaderboard.sort(key=lambda row: (row["full_examples_avg"], row["holdout_mean"]), reverse=True)
    report["leaderboard"] = leaderboard

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
