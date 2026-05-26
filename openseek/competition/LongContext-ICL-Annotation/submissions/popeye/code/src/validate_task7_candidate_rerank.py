import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import requests

from main import TASK_FILES
from method import (
    DEFAULT_CHAT_URL,
    DEFAULT_COMPLETION_URL,
    DEFAULT_REQUEST_TIMEOUT,
    TASK_REQUEST_ATTEMPTS,
    TASK_STOPS,
    build_task7_append_unique_judge_candidates,
    build_task7_rerank_generation_prompt,
    build_prompt,
    build_task7_retrieval_context,
    count_answer,
    detect_task7_secondary_family,
    extract_task7_category_constraints,
    get_example_pool_limit,
    get_task7_constraint_subtypes,
    get_profile_name,
    get_profile_retrieval_tasks,
    get_served_model_name,
    is_task7_constraint_secondary_allowed,
    is_task7_secondary_family_allowed,
    parse_task7_fields,
    reorder_task7_examples_by_lexical_retrieval,
    score_task7_candidate_constraints,
    select_examples,
)
from probe_task7_direct_fact_expansion import (
    aggregate_direct_fact_candidates,
    build_direct_fact_questions,
    collect_question_candidates,
    detect_direct_fact_bucket,
)
from task7_direct_fact_projection import (
    build_author_answer_catalog,
    project_direct_fact_candidates_to_author_catalog,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_limit", type=int, default=8)
    parser.add_argument(
        "--direct_fact_bucket_filter",
        type=str,
        choices=["all", "quoted_work_author_relation", "legacy_show_host", "explicit_company_publisher_book", "none"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--profile", type=str, default="baseline")
    parser.add_argument("--examples_limit", type=int, default=None)
    parser.add_argument("--retrieval_mode", type=str, choices=["none", "lexical"], default="none")
    parser.add_argument("--n_candidates", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_candidates_for_judge", type=int, default=6)
    parser.add_argument("--judge_mode", type=str, choices=["completion", "chat"], default="completion")
    parser.add_argument(
        "--constraint_rerank",
        type=str,
        choices=["off", "soft", "hard_gate"],
        default="off",
    )
    parser.add_argument(
        "--generation_hint_mode",
        type=str,
        choices=["off", "full", "constraint_only", "typed", "recall"],
        default="off",
    )
    parser.add_argument("--use_generation_hints", action="store_true")
    parser.add_argument("--secondary_profile", type=str, default=None)
    parser.add_argument("--secondary_retrieval_mode", type=str, choices=["none", "lexical"], default=None)
    parser.add_argument("--secondary_n_candidates", type=int, default=0)
    parser.add_argument(
        "--secondary_source",
        type=str,
        choices=["completion", "direct_fact_projected"],
        default="completion",
    )
    parser.add_argument("--secondary_temperature", type=float, default=None)
    parser.add_argument("--secondary_top_p", type=float, default=None)
    parser.add_argument("--secondary_use_generation_hints", action="store_true")
    parser.add_argument(
        "--secondary_generation_hint_mode",
        type=str,
        choices=["off", "full", "constraint_only", "typed", "recall"],
        default="off",
    )
    parser.add_argument(
        "--secondary_typed_route",
        type=str,
        choices=["off", "auto"],
        default="off",
    )
    parser.add_argument("--secondary_merge_mode", type=str, choices=["counts", "append_unique"], default="counts")
    parser.add_argument(
        "--secondary_gate",
        type=str,
        choices=["off", "entropy_or_unique", "unique_only", "entropy_only"],
        default="off",
    )
    parser.add_argument("--secondary_min_unique", type=int, default=4)
    parser.add_argument("--secondary_min_entropy", type=float, default=1.3)
    parser.add_argument("--secondary_family_allowlist", type=str, default=None)
    parser.add_argument("--secondary_constraint_allowlist", type=str, default=None)
    parser.add_argument("--append_unique_reserved_secondary_slots", type=int, default=2)
    parser.add_argument("--pairwise_triggered_mode", action="store_true")
    parser.add_argument("--pairwise_triggered_max_secondary", type=int, default=3)
    parser.add_argument("--diagnosis_mode", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def normalize_qa_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def request_completion_choices(
    prompt: str,
    task_id: int,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 64,
) -> list[str]:
    data = {
        "model": get_served_model_name(),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }
    completion_seed = os.environ.get("OPENSEEK_COMPLETION_SEED", "").strip()
    if completion_seed:
        try:
            data["seed"] = int(completion_seed)
        except ValueError:
            pass
    stop = TASK_STOPS.get(task_id)
    if stop:
        data["stop"] = stop

    attempts = TASK_REQUEST_ATTEMPTS.get(task_id, 1)
    for attempt in range(attempts):
        try:
            resp = requests.post(DEFAULT_COMPLETION_URL, json=data, timeout=DEFAULT_REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            return [choice.get("text", "") for choice in payload.get("choices", [])]
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    return []


def request_chat_choice(
    system_msg: str,
    user_msg: str,
    max_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    data = {
        "model": get_served_model_name(),
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    attempts = TASK_REQUEST_ATTEMPTS.get(7, 1)
    for attempt in range(attempts):
        try:
            resp = requests.post(DEFAULT_CHAT_URL, json=data, timeout=DEFAULT_REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            return payload["choices"][0]["message"]["content"]
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    return ""


def build_judge_prompt(category: str, clue: str, candidates: list[str]) -> str:
    candidate_lines = [f"{idx + 1}. {candidate}" for idx, candidate in enumerate(candidates)]
    return (
        "You are selecting the best Jeopardy answer from candidate options.\n\n"
        "Rules:\n"
        "1. Choose the single candidate that best matches both the category and the clue.\n"
        "2. Prefer the canonical Jeopardy answer: the exact entity or phrase, not a paraphrase, broader class, or clue fragment.\n"
        "3. If one option is a short incomplete fragment and another is the fully specified answer, choose the fully specified answer.\n"
        "4. If one option is an expanded team/place name and another is the shorter canonical answer used in Jeopardy, choose the canonical answer.\n"
        "5. Return only the selected option number inside <label>...</label>.\n\n"
        f"Category: {category}\n"
        f"Clue: {clue}\n\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\nAnswer:\n<label>"
    )


def build_chat_judge_messages(category: str, clue: str, candidates: list[str]) -> tuple[str, str]:
    system_msg = (
        "You select the best Jeopardy answer from candidate options. "
        "Think carefully about category specificity and clue meaning, then return only the chosen option number."
    )
    user_msg = build_judge_prompt(category, clue, candidates)
    return system_msg, user_msg


def extract_option_index(text: str) -> int | None:
    if not text:
        return None
    match = re.search(r"\b(\d+)\b", text)
    if not match:
        return None
    return int(match.group(1)) - 1


def choose_vote_candidate(candidates: list[str], counts: Counter) -> str | None:
    if not candidates:
        return None
    scored = []
    for candidate in candidates:
        norm = normalize_qa_answer(candidate)
        scored.append((counts[norm], -len(norm.split()), -len(norm), candidate))
    scored.sort(reverse=True)
    return scored[0][3]


def rerank_candidates_by_constraints(
    candidates: list[str],
    counts: Counter,
    *,
    category: str,
    mode: str,
) -> tuple[list[str], list[dict]]:
    if mode == "off" or not candidates:
        return candidates, []

    scored_rows = []
    for candidate in candidates:
        norm = normalize_qa_answer(candidate)
        constraint_info = score_task7_candidate_constraints(candidate, category)
        scored_rows.append(
            {
                "candidate": candidate,
                "normalized": norm,
                "count": counts.get(norm, 0),
                "constraint_score": constraint_info["score"],
                "matched_constraints": constraint_info["matched_constraints"],
                "hard_violations": constraint_info["hard_violations"],
                "constraint_count": constraint_info["constraint_count"],
                "constraint_details": constraint_info["details"],
            }
        )

    eligible_rows = scored_rows
    if mode == "hard_gate":
        filtered_rows = [
            row for row in scored_rows
            if row["constraint_count"] > 0 and row["hard_violations"] == 0
        ]
        if filtered_rows:
            eligible_rows = filtered_rows

    eligible_rows.sort(
        key=lambda row: (
            -row["constraint_score"],
            -row["matched_constraints"],
            row["hard_violations"],
            -row["count"],
            len(row["normalized"].split()),
            len(row["normalized"]),
        )
    )
    return [row["candidate"] for row in eligible_rows], scored_rows


def build_task7_prompt(
    task_description: str,
    text2annotate: str,
    icl_examples: list[dict],
    profile_name: str,
    retrieval_mode: str,
) -> str:
    prompt = build_prompt(task_description, text2annotate, task_id=7, profile_name=profile_name)
    current_examples = icl_examples
    profile_retrieval_tasks = get_profile_retrieval_tasks(profile_name)
    use_retrieval = retrieval_mode == "lexical" or 7 in profile_retrieval_tasks
    if use_retrieval:
        retrieval_context = build_task7_retrieval_context(icl_examples)
        current_examples = reorder_task7_examples_by_lexical_retrieval(
            icl_examples,
            text2annotate,
            retrieval_context,
        )
    examples_str = select_examples(
        current_examples,
        task_description,
        text2annotate,
        task_id=7,
        profile_name=profile_name,
    )
    return prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")


def build_branch_icl_pool(
    examples: list[dict],
    holdout_ids: set[str],
    profile_name: str,
    examples_limit: int | None,
) -> tuple[list[dict], int]:
    resolved_examples_limit = get_example_pool_limit(
        task_id=7,
        examples_limit=examples_limit,
        profile_name=profile_name,
    )
    icl_pool = [example for example in examples if example["id"] not in holdout_ids][:resolved_examples_limit]
    return icl_pool, resolved_examples_limit


def collect_branch_choices(
    prompt: str,
    text2annotate: str,
    n_candidates: int,
    temperature: float,
    top_p: float,
    use_generation_hints: bool,
    generation_hint_mode: str,
) -> tuple[str, list[str]]:
    hint_mode = generation_hint_mode
    if use_generation_hints and hint_mode == "off":
        hint_mode = "full"
    generation_prompt = (
        build_task7_rerank_generation_prompt(prompt, text2annotate, hint_mode=hint_mode)
        if hint_mode != "off"
        else prompt
    )
    raw_choices = request_completion_choices(
        generation_prompt,
        task_id=7,
        n=n_candidates,
        temperature=temperature,
        top_p=top_p,
        max_tokens=64,
    )
    return generation_prompt, raw_choices


def resolve_generation_hint_mode(use_generation_hints: bool, generation_hint_mode: str) -> str:
    if generation_hint_mode != "off":
        return generation_hint_mode
    return "full" if use_generation_hints else "off"


def parse_token_allowlist(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return set()
    return {
        token.strip().lower()
        for token in raw.split(",")
        if token.strip()
    }


def classify_direct_fact_bucket_for_sample(sample: dict, typed_route: str = "auto") -> str:
    category, clue = parse_task7_fields(sample["input"])
    family = detect_task7_secondary_family(category, clue, typed_route=typed_route)
    return detect_direct_fact_bucket(category, clue, family) or "none"


def build_task7_author_catalog(examples: list[dict]) -> dict:
    author_answers = []
    for sample in examples:
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        if detect_direct_fact_bucket(category, clue, family) == "quoted_work_author_relation":
            author_answers.append(sample["output"][0])
    return build_author_answer_catalog(author_answers)


def collect_direct_fact_projected_candidates(
    *,
    category: str,
    clue: str,
    family: str,
    bucket: str,
    author_catalog: dict | None,
    rounds: int,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[list[str], list[str], list[dict]]:
    reports = []
    for question in build_direct_fact_questions(category, clue, family):
        reports.append(
            collect_question_candidates(
                question=question,
                family=family,
                rounds=rounds,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )
    raw_candidates = aggregate_direct_fact_candidates(reports)
    projected_candidates = list(raw_candidates)
    projection_trace = []
    if bucket == "quoted_work_author_relation" and author_catalog:
        projected_candidates, projection_trace = project_direct_fact_candidates_to_author_catalog(
            raw_candidates,
            author_catalog,
        )
    return raw_candidates, projected_candidates, projection_trace


def init_metric_bucket() -> dict:
    return {
        "count": 0,
        "secondary_triggered": 0,
        "primary_oracle_hit": 0,
        "oracle_hit": 0,
        "primary_judge_correct": 0,
        "judge_correct": 0,
        "oracle_gain_count": 0,
        "judge_gain_count": 0,
        "judge_secondary_visible_count": 0,
        "judge_selected_from_secondary_count": 0,
    }


def finalize_metric_bucket(bucket: dict) -> dict:
    count = bucket["count"]
    triggered = bucket["secondary_triggered"]
    bucket["oracle_hit_rate"] = bucket["oracle_hit"] / count if count else 0.0
    bucket["judge_accuracy"] = bucket["judge_correct"] / count if count else 0.0
    bucket["primary_oracle_hit_rate"] = bucket["primary_oracle_hit"] / count if count else 0.0
    bucket["primary_judge_accuracy"] = bucket["primary_judge_correct"] / count if count else 0.0
    bucket["oracle_gain_on_triggered"] = bucket["oracle_gain_count"] / triggered if triggered else 0.0
    bucket["judge_gain_on_triggered"] = bucket["judge_gain_count"] / triggered if triggered else 0.0
    bucket["judge_secondary_visible_rate"] = (
        bucket["judge_secondary_visible_count"] / triggered if triggered else 0.0
    )
    bucket["judge_selected_from_secondary_rate"] = (
        bucket["judge_selected_from_secondary_count"] / triggered if triggered else 0.0
    )
    return bucket


def dedupe_candidates(raw_choices: list[str]) -> tuple[list[str], Counter]:
    counts = Counter()
    normalized_to_candidate = {}
    for raw in raw_choices:
        answer = count_answer(raw, task_id=7)
        normalized = normalize_qa_answer(answer)
        if not normalized:
            continue
        counts[normalized] += 1
        normalized_to_candidate.setdefault(normalized, answer.strip())
    ordered = [normalized_to_candidate[norm] for norm, _ in counts.most_common()]
    return ordered, counts


def append_unique_candidates(primary_candidates: list[str], secondary_candidates: list[str]) -> list[str]:
    merged = list(primary_candidates)
    seen = {normalize_qa_answer(candidate) for candidate in primary_candidates}
    for candidate in secondary_candidates:
        normalized = normalize_qa_answer(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(candidate)
    return merged


def get_secondary_only_candidates(primary_candidates: list[str], secondary_candidates: list[str]) -> list[str]:
    primary_seen = {normalize_qa_answer(candidate) for candidate in primary_candidates}
    secondary_only = []
    seen = set()
    for candidate in secondary_candidates:
        normalized = normalize_qa_answer(candidate)
        if not normalized or normalized in primary_seen or normalized in seen:
            continue
        seen.add(normalized)
        secondary_only.append(candidate)
    return secondary_only


def summarize_candidate_pool(candidate_counts: Counter) -> dict:
    counts = list(candidate_counts.values())
    total = sum(counts)
    top = counts[0] if counts else 0
    second = counts[1] if len(counts) > 1 else 0
    entropy = 0.0
    if total:
        for count in counts:
            p = count / total
            entropy -= p * math.log(p, 2)
    return {
        "unique_candidates": len(counts),
        "top_share": top / total if total else 0.0,
        "margin_share": (top - second) / total if total else 0.0,
        "entropy": entropy,
        "total_votes": total,
    }


def should_trigger_secondary_gate(pool_summary: dict, gate_mode: str, min_unique: int, min_entropy: float) -> bool:
    if gate_mode == "off":
        return True
    if gate_mode == "entropy_or_unique":
        return (
            pool_summary["unique_candidates"] >= min_unique
            or pool_summary["entropy"] >= min_entropy
        )
    if gate_mode == "unique_only":
        return pool_summary["unique_candidates"] >= min_unique
    if gate_mode == "entropy_only":
        return pool_summary["entropy"] >= min_entropy
    return True


def classify_gold_presence_bucket(
    gold: str,
    primary_candidates: list[str],
    secondary_only_candidates: list[str],
) -> str:
    gold_norm = normalize_qa_answer(gold)
    if not gold_norm:
        return "absent"
    in_primary = gold_norm in {normalize_qa_answer(candidate) for candidate in primary_candidates}
    in_secondary_only = gold_norm in {normalize_qa_answer(candidate) for candidate in secondary_only_candidates}
    if in_primary and in_secondary_only:
        return "both"
    if in_primary:
        return "primary_only"
    if in_secondary_only:
        return "secondary_only"
    return "absent"


def build_order_sensitivity_markers(
    primary_judge_candidates: list[str],
    judge_candidates_visible: list[str],
    secondary_only_candidates: list[str],
) -> dict:
    judge_order = list(judge_candidates_visible)
    secondary_only_norms = {normalize_qa_answer(candidate) for candidate in secondary_only_candidates}
    secondary_positions = [
        idx
        for idx, candidate in enumerate(judge_order)
        if normalize_qa_answer(candidate) in secondary_only_norms
    ]
    return {
        "judge_candidate_order": judge_order,
        "secondary_only_candidate_norms": sorted(secondary_only_norms),
        "secondary_only_visible_count": len(secondary_positions),
        "secondary_only_positions": secondary_positions,
        "secondary_only_lead_position": secondary_positions[0] if secondary_positions else None,
        "secondary_only_in_top2": any(position < 2 for position in secondary_positions),
        "secondary_only_in_top3": any(position < 3 for position in secondary_positions),
    }


def run_judge(
    *,
    category: str,
    clue: str,
    candidates: list[str],
    judge_mode: str,
) -> tuple[str | None, str]:
    if not candidates:
        return None, ""

    judge_raw = ""
    if judge_mode == "chat":
        system_msg, user_msg = build_chat_judge_messages(category, clue, candidates)
        judge_raw = request_chat_choice(
            system_msg=system_msg,
            user_msg=user_msg,
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
    else:
        judge_prompt = build_judge_prompt(category, clue, candidates)
        judge_choices = request_completion_choices(
            judge_prompt,
            task_id=7,
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
        )
        judge_raw = judge_choices[0] if judge_choices else ""

    judge_index = extract_option_index(judge_raw)
    if judge_index is not None and 0 <= judge_index < len(candidates):
        return candidates[judge_index], judge_raw
    return None, judge_raw


def run_pairwise_triggered_rerank(
    *,
    category: str,
    clue: str,
    primary_candidate: str | None,
    secondary_only_candidates: list[str],
    judge_mode: str,
    max_secondary: int,
) -> tuple[str | None, dict]:
    if not primary_candidate:
        return None, {
            "enabled": True,
            "ran": False,
            "reason": "no_primary_candidate",
            "comparisons": [],
        }
    if not secondary_only_candidates:
        return primary_candidate, {
            "enabled": True,
            "ran": False,
            "reason": "no_secondary_only_candidates",
            "comparisons": [],
        }

    comparisons = []
    best_candidate = primary_candidate
    best_score = 0
    limited_secondary = secondary_only_candidates[: max(0, max_secondary)] if max_secondary > 0 else []
    if not limited_secondary:
        return primary_candidate, {
            "enabled": True,
            "ran": False,
            "reason": "max_secondary_zero",
            "comparisons": [],
        }

    for secondary_candidate in limited_secondary:
        ab_candidates = [primary_candidate, secondary_candidate]
        ab_choice, ab_raw = run_judge(
            category=category,
            clue=clue,
            candidates=ab_candidates,
            judge_mode=judge_mode,
        )
        if ab_choice is None:
            ab_choice = primary_candidate

        ba_candidates = [secondary_candidate, primary_candidate]
        ba_choice, ba_raw = run_judge(
            category=category,
            clue=clue,
            candidates=ba_candidates,
            judge_mode=judge_mode,
        )
        if ba_choice is None:
            ba_choice = primary_candidate

        secondary_norm = normalize_qa_answer(secondary_candidate)
        ab_prefers_secondary = normalize_qa_answer(ab_choice) == secondary_norm
        ba_prefers_secondary = normalize_qa_answer(ba_choice) == secondary_norm
        secondary_win_count = int(ab_prefers_secondary) + int(ba_prefers_secondary)
        qualifies_for_takeover = secondary_win_count == 2
        rejected_split_takeover = secondary_win_count == 1
        if qualifies_for_takeover and secondary_win_count > best_score:
            best_score = secondary_win_count
            best_candidate = secondary_candidate

        comparisons.append(
            {
                "secondary_candidate": secondary_candidate,
                "ab": {
                    "candidates": ab_candidates,
                    "judge_raw": ab_raw,
                    "winner": ab_choice,
                    "winner_is_secondary": ab_prefers_secondary,
                },
                "ba": {
                    "candidates": ba_candidates,
                    "judge_raw": ba_raw,
                    "winner": ba_choice,
                    "winner_is_secondary": ba_prefers_secondary,
                },
                "secondary_win_count": secondary_win_count,
                "secondary_preference_rate": secondary_win_count / 2.0,
                "qualifies_for_takeover": qualifies_for_takeover,
                "rejected_split_takeover": rejected_split_takeover,
            }
        )

    return best_candidate, {
        "enabled": True,
        "ran": True,
        "reason": None,
        "primary_candidate": primary_candidate,
        "selected_candidate": best_candidate,
        "selected_candidate_source": (
            "secondary_only" if normalize_qa_answer(best_candidate) != normalize_qa_answer(primary_candidate) else "primary"
        ),
        "selected_candidate_win_count": best_score,
        "takeover_requires_unanimous_pairwise_win": True,
        "split_takeovers_rejected_count": sum(int(comp["rejected_split_takeover"]) for comp in comparisons),
        "comparisons": comparisons,
    }


def main():
    args = parse_args()
    profile_name = get_profile_name(args.profile)

    task_dict = json.loads(Path(TASK_FILES[7]).read_text(encoding="utf-8"))
    examples = list(task_dict["examples"])

    rng = random.Random(args.seed + 7)
    sampled_examples = examples
    if args.direct_fact_bucket_filter != "all":
        sampled_examples = [
            example
            for example in examples
            if classify_direct_fact_bucket_for_sample(example, typed_route="auto")
            == args.direct_fact_bucket_filter
        ]
    holdout = rng.sample(sampled_examples, min(args.sample_limit, len(sampled_examples)))
    holdout_ids = {example["id"] for example in holdout}
    primary_icl_pool, resolved_examples_limit = build_branch_icl_pool(
        examples,
        holdout_ids,
        profile_name=profile_name,
        examples_limit=args.examples_limit,
    )

    rows = []
    vote_correct = 0
    judge_correct = 0
    oracle_hits = 0
    primary_judge_correct = 0
    primary_oracle_hits = 0
    secondary_triggered = 0
    secondary_skipped = 0
    judge_secondary_visible_count = 0
    judge_selected_from_secondary_count = 0
    oracle_gain_count = 0
    judge_gain_count = 0
    typed_family_breakdown = {
        family: init_metric_bucket()
        for family in ("constraint", "numeric", "person_entity", "title_or_place", "generic")
    }
    direct_fact_bucket_breakdown = {
        bucket: init_metric_bucket()
        for bucket in ("quoted_work_author_relation", "other", "none")
    }
    constraint_subtype_breakdown = {
        subtype: init_metric_bucket()
        for subtype in (
            "letter_count",
            "quoted_fragment",
            "starts_with_options",
            "before_after",
            "title_character",
            "team_name",
            "person_name",
        )
    }

    secondary_enabled = args.secondary_n_candidates > 0 or args.secondary_source == "direct_fact_projected"
    secondary_profile_name = get_profile_name(args.secondary_profile) if args.secondary_profile else profile_name
    secondary_retrieval_mode = args.secondary_retrieval_mode or args.retrieval_mode
    secondary_temperature = args.secondary_temperature if args.secondary_temperature is not None else args.temperature
    secondary_top_p = args.secondary_top_p if args.secondary_top_p is not None else args.top_p
    secondary_family_allowlist = parse_token_allowlist(args.secondary_family_allowlist)
    secondary_constraint_allowlist = parse_token_allowlist(args.secondary_constraint_allowlist)
    secondary_icl_pool = None
    secondary_examples_limit = None
    author_catalog = build_task7_author_catalog(examples) if args.secondary_source == "direct_fact_projected" else None
    if secondary_enabled:
        secondary_icl_pool, secondary_examples_limit = build_branch_icl_pool(
            examples,
            holdout_ids,
            profile_name=secondary_profile_name,
            examples_limit=args.examples_limit,
        )

    for sample in holdout:
        text2annotate = sample["input"]
        gold = sample["output"][0]
        category, clue = parse_task7_fields(text2annotate)
        category_constraints = extract_task7_category_constraints(category)
        constraint_subtypes = get_task7_constraint_subtypes(category)
        secondary_family = detect_task7_secondary_family(
            category,
            clue,
            typed_route=("auto" if args.secondary_source == "direct_fact_projected" else args.secondary_typed_route),
        )
        direct_fact_bucket = detect_direct_fact_bucket(category, clue, secondary_family) or "none"
        if direct_fact_bucket != "quoted_work_author_relation":
            direct_fact_bucket = "other" if direct_fact_bucket != "none" else "none"
        typed_family_breakdown.setdefault(secondary_family, init_metric_bucket())
        typed_family_breakdown[secondary_family]["count"] += 1
        direct_fact_bucket_breakdown.setdefault(direct_fact_bucket, init_metric_bucket())
        direct_fact_bucket_breakdown[direct_fact_bucket]["count"] += 1
        for subtype in constraint_subtypes:
            constraint_subtype_breakdown.setdefault(subtype, init_metric_bucket())
            constraint_subtype_breakdown[subtype]["count"] += 1
        prompt = build_task7_prompt(
            task_description=task_dict["Definition"][0],
            text2annotate=text2annotate,
            icl_examples=primary_icl_pool,
            profile_name=profile_name,
            retrieval_mode=args.retrieval_mode,
        )
        primary_generation_prompt, primary_raw_choices = collect_branch_choices(
            prompt=prompt,
            text2annotate=text2annotate,
            n_candidates=args.n_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            use_generation_hints=args.use_generation_hints,
            generation_hint_mode=args.generation_hint_mode,
        )

        branch_details = {
            "primary": {
                "retrieval_mode": args.retrieval_mode,
                "profile": profile_name,
                "example_pool_limit": resolved_examples_limit,
                "n_candidates": args.n_candidates,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "use_generation_hints": args.use_generation_hints,
                "generation_hint_mode": resolve_generation_hint_mode(
                    args.use_generation_hints,
                    args.generation_hint_mode,
                ),
                "raw_choices": primary_raw_choices,
                "generation_prompt_preview": primary_generation_prompt[-400:],
            }
        }
        raw_choices = list(primary_raw_choices)
        primary_unique_candidates, primary_candidate_counts = dedupe_candidates(primary_raw_choices)
        primary_pool_summary = summarize_candidate_pool(primary_candidate_counts)
        secondary_raw_choices = []
        secondary_unique_candidates = []
        secondary_candidate_counts = Counter()
        secondary_only_candidates = []
        gate_triggered = False
        secondary_skip_reason = None

        if secondary_enabled:
            gate_triggered = should_trigger_secondary_gate(
                primary_pool_summary,
                gate_mode=args.secondary_gate,
                min_unique=args.secondary_min_unique,
                min_entropy=args.secondary_min_entropy,
            )
            branch_details["secondary_gate"] = {
                "mode": args.secondary_gate,
                "triggered": gate_triggered,
                "min_unique": args.secondary_min_unique,
                "min_entropy": args.secondary_min_entropy,
                "primary_pool_summary": primary_pool_summary,
            }
            if gate_triggered:
                skip_constraint_only_secondary = (
                    args.secondary_generation_hint_mode == "constraint_only"
                    and not category_constraints
                )
                skip_typed_generic_secondary = (
                    args.secondary_generation_hint_mode == "typed"
                    and secondary_family == "generic"
                )
                skip_family_allowlist_secondary = (
                    args.secondary_generation_hint_mode == "typed"
                    and not is_task7_secondary_family_allowed(
                        secondary_family,
                        allowlist=secondary_family_allowlist,
                    )
                )
                skip_constraint_allowlist_secondary = (
                    args.secondary_generation_hint_mode == "typed"
                    and secondary_family == "constraint"
                    and not is_task7_constraint_secondary_allowed(
                        category,
                        allowlist=secondary_constraint_allowlist,
                    )
                )
                if (
                    skip_constraint_only_secondary
                    or skip_typed_generic_secondary
                    or skip_family_allowlist_secondary
                    or skip_constraint_allowlist_secondary
                ):
                    secondary_skipped += 1
                    gate_triggered = False
                    secondary_generation_prompt = ""
                    secondary_skip_reason = "no_category_constraints"
                    if skip_typed_generic_secondary:
                        secondary_skip_reason = "generic_family"
                    elif skip_family_allowlist_secondary:
                        secondary_skip_reason = "family_allowlist"
                    elif skip_constraint_allowlist_secondary:
                        secondary_skip_reason = "constraint_allowlist"
                    branch_details["secondary_gate"]["triggered"] = False
                    branch_details["secondary_gate"]["skip_reason"] = secondary_skip_reason
                else:
                    if args.secondary_source == "direct_fact_projected":
                        secondary_raw_choices, secondary_unique_candidates, projection_trace = collect_direct_fact_projected_candidates(
                            category=category,
                            clue=clue,
                            family=secondary_family,
                            bucket=direct_fact_bucket if direct_fact_bucket != "other" else "none",
                            author_catalog=author_catalog,
                            rounds=2,
                            n=max(args.secondary_n_candidates, 1),
                            temperature=secondary_temperature,
                            top_p=secondary_top_p,
                            max_tokens=32,
                        )
                        secondary_candidate_counts = Counter()
                        for candidate in secondary_unique_candidates:
                            normalized = normalize_qa_answer(candidate)
                            if normalized:
                                secondary_candidate_counts[normalized] = 1
                        secondary_only_candidates = get_secondary_only_candidates(
                            primary_unique_candidates,
                            secondary_unique_candidates,
                        )
                        secondary_generation_prompt = ""
                    else:
                        secondary_prompt = build_task7_prompt(
                            task_description=task_dict["Definition"][0],
                            text2annotate=text2annotate,
                            icl_examples=secondary_icl_pool,
                            profile_name=secondary_profile_name,
                            retrieval_mode=secondary_retrieval_mode,
                        )
                        secondary_generation_prompt, secondary_raw_choices = collect_branch_choices(
                            prompt=secondary_prompt,
                            text2annotate=text2annotate,
                            n_candidates=args.secondary_n_candidates,
                            temperature=secondary_temperature,
                            top_p=secondary_top_p,
                            use_generation_hints=args.secondary_use_generation_hints,
                            generation_hint_mode=args.secondary_generation_hint_mode,
                        )
                        secondary_unique_candidates, secondary_candidate_counts = dedupe_candidates(secondary_raw_choices)
                        secondary_only_candidates = get_secondary_only_candidates(
                            primary_unique_candidates,
                            secondary_unique_candidates,
                        )
                        projection_trace = []
                    secondary_triggered += 1
                    typed_family_breakdown[secondary_family]["secondary_triggered"] += 1
                    direct_fact_bucket_breakdown[direct_fact_bucket]["secondary_triggered"] += 1
                    for subtype in constraint_subtypes:
                        constraint_subtype_breakdown[subtype]["secondary_triggered"] += 1
            else:
                secondary_skipped += 1
                secondary_generation_prompt = ""
                secondary_skip_reason = "primary_pool_gate"
            branch_details["secondary"] = {
                "retrieval_mode": secondary_retrieval_mode,
                "profile": secondary_profile_name,
                "example_pool_limit": secondary_examples_limit,
                "n_candidates": args.secondary_n_candidates,
                "temperature": secondary_temperature,
                "top_p": secondary_top_p,
                "use_generation_hints": args.secondary_use_generation_hints,
                "generation_hint_mode": resolve_generation_hint_mode(
                    args.secondary_use_generation_hints,
                    args.secondary_generation_hint_mode,
                ),
                "secondary_source": args.secondary_source,
                "typed_family": secondary_family,
                "constraint_subtypes": constraint_subtypes,
                "family_allowlist": sorted(secondary_family_allowlist) if secondary_family_allowlist is not None else None,
                "constraint_allowlist": sorted(secondary_constraint_allowlist) if secondary_constraint_allowlist is not None else None,
                "raw_choices": secondary_raw_choices,
                "unique_candidates": secondary_unique_candidates,
                "secondary_only_candidates": secondary_only_candidates,
                "generation_prompt_preview": secondary_generation_prompt[-400:],
            }
            branch_details["secondary_gate"]["skip_reason"] = secondary_skip_reason
            if gate_triggered and args.secondary_merge_mode == "counts":
                raw_choices.extend(secondary_raw_choices)

        if args.secondary_merge_mode == "append_unique" and secondary_enabled and gate_triggered:
            unique_candidates = append_unique_candidates(primary_unique_candidates, secondary_unique_candidates)
            candidate_counts = Counter(primary_candidate_counts)
            for normalized, count in secondary_candidate_counts.items():
                candidate_counts.setdefault(normalized, count)
            vote_candidate = choose_vote_candidate(primary_unique_candidates, primary_candidate_counts)
        else:
            unique_candidates, candidate_counts = dedupe_candidates(raw_choices)
            vote_candidate = choose_vote_candidate(unique_candidates, candidate_counts)

        primary_ranked_candidates, primary_constraint_debug = rerank_candidates_by_constraints(
            primary_unique_candidates,
            primary_candidate_counts,
            category=category,
            mode=args.constraint_rerank,
        )
        primary_judge_candidates = primary_ranked_candidates[: args.max_candidates_for_judge]

        judge_candidates, constraint_debug = rerank_candidates_by_constraints(
            unique_candidates,
            candidate_counts,
            category=category,
            mode=args.constraint_rerank,
        )
        if args.secondary_merge_mode == "append_unique" and secondary_enabled and gate_triggered:
            truncated_candidates = build_task7_append_unique_judge_candidates(
                primary_ranked_candidates,
                secondary_unique_candidates,
                max_candidates=args.max_candidates_for_judge,
                reserved_secondary_slots=args.append_unique_reserved_secondary_slots,
            )
        else:
            truncated_candidates = judge_candidates[: args.max_candidates_for_judge]

        vote_ok = normalize_qa_answer(vote_candidate) == normalize_qa_answer(gold)
        vote_correct += int(vote_ok)

        primary_oracle_hit = normalize_qa_answer(gold) in {
            normalize_qa_answer(candidate) for candidate in primary_unique_candidates
        }
        primary_oracle_hits += int(primary_oracle_hit)
        oracle_hit = normalize_qa_answer(gold) in {normalize_qa_answer(candidate) for candidate in unique_candidates}
        oracle_hits += int(oracle_hit)
        typed_family_breakdown[secondary_family]["primary_oracle_hit"] += int(primary_oracle_hit)
        typed_family_breakdown[secondary_family]["oracle_hit"] += int(oracle_hit)
        direct_fact_bucket_breakdown[direct_fact_bucket]["primary_oracle_hit"] += int(primary_oracle_hit)
        direct_fact_bucket_breakdown[direct_fact_bucket]["oracle_hit"] += int(oracle_hit)
        for subtype in constraint_subtypes:
            constraint_subtype_breakdown[subtype]["primary_oracle_hit"] += int(primary_oracle_hit)
            constraint_subtype_breakdown[subtype]["oracle_hit"] += int(oracle_hit)

        primary_judge_candidate, primary_judge_raw = run_judge(
            category=category,
            clue=clue,
            candidates=primary_judge_candidates,
            judge_mode=args.judge_mode,
        )
        if primary_judge_candidate is None:
            primary_judge_candidate = choose_vote_candidate(primary_unique_candidates, primary_candidate_counts)
        primary_judge_ok = normalize_qa_answer(primary_judge_candidate) == normalize_qa_answer(gold)
        primary_judge_correct += int(primary_judge_ok)
        typed_family_breakdown[secondary_family]["primary_judge_correct"] += int(primary_judge_ok)
        direct_fact_bucket_breakdown[direct_fact_bucket]["primary_judge_correct"] += int(primary_judge_ok)
        for subtype in constraint_subtypes:
            constraint_subtype_breakdown[subtype]["primary_judge_correct"] += int(primary_judge_ok)

        judge_candidate, judge_raw = run_judge(
            category=category,
            clue=clue,
            candidates=truncated_candidates,
            judge_mode=args.judge_mode,
        )
        if judge_candidate is None:
            judge_candidate = vote_candidate

        pairwise_triggered_debug = {
            "enabled": args.pairwise_triggered_mode,
            "ran": False,
            "reason": "disabled",
            "comparisons": [],
        }
        pairwise_triggered_candidate = judge_candidate
        if args.pairwise_triggered_mode and gate_triggered:
            pairwise_triggered_candidate, pairwise_triggered_debug = run_pairwise_triggered_rerank(
                category=category,
                clue=clue,
                primary_candidate=judge_candidate,
                secondary_only_candidates=secondary_only_candidates,
                judge_mode=args.judge_mode,
                max_secondary=args.pairwise_triggered_max_secondary,
            )
            if pairwise_triggered_candidate is None:
                pairwise_triggered_candidate = judge_candidate
            judge_candidate = pairwise_triggered_candidate

        judge_ok = normalize_qa_answer(judge_candidate) == normalize_qa_answer(gold)
        judge_correct += int(judge_ok)
        typed_family_breakdown[secondary_family]["judge_correct"] += int(judge_ok)
        direct_fact_bucket_breakdown[direct_fact_bucket]["judge_correct"] += int(judge_ok)
        for subtype in constraint_subtypes:
            constraint_subtype_breakdown[subtype]["judge_correct"] += int(judge_ok)

        judge_secondary_visible = False
        judge_selected_from_secondary = False
        secondary_only_norms = {normalize_qa_answer(candidate) for candidate in secondary_only_candidates}
        if gate_triggered and secondary_only_norms:
            judge_secondary_visible = any(
                normalize_qa_answer(candidate) in secondary_only_norms for candidate in truncated_candidates
            )
            judge_selected_from_secondary = normalize_qa_answer(judge_candidate) in secondary_only_norms
            judge_secondary_visible_count += int(judge_secondary_visible)
            judge_selected_from_secondary_count += int(judge_selected_from_secondary)
            typed_family_breakdown[secondary_family]["judge_secondary_visible_count"] += int(judge_secondary_visible)
            typed_family_breakdown[secondary_family]["judge_selected_from_secondary_count"] += int(
                judge_selected_from_secondary
            )
            direct_fact_bucket_breakdown[direct_fact_bucket]["judge_secondary_visible_count"] += int(
                judge_secondary_visible
            )
            direct_fact_bucket_breakdown[direct_fact_bucket]["judge_selected_from_secondary_count"] += int(
                judge_selected_from_secondary
            )
            for subtype in constraint_subtypes:
                constraint_subtype_breakdown[subtype]["judge_secondary_visible_count"] += int(judge_secondary_visible)
                constraint_subtype_breakdown[subtype]["judge_selected_from_secondary_count"] += int(
                    judge_selected_from_secondary
                )

        gold_presence_bucket = classify_gold_presence_bucket(
            gold,
            primary_unique_candidates,
            secondary_only_candidates,
        )
        order_sensitivity = build_order_sensitivity_markers(
            primary_judge_candidates,
            truncated_candidates,
            secondary_only_candidates,
        )
        selection_risk_marker = "secondary_not_visible"
        if gate_triggered and not secondary_only_candidates:
            selection_risk_marker = "no_secondary_unique"
        elif gate_triggered and judge_secondary_visible and judge_selected_from_secondary:
            selection_risk_marker = "secondary_selected"
        elif gate_triggered and judge_secondary_visible:
            selection_risk_marker = "secondary_visible_not_selected"
        elif gate_triggered:
            selection_risk_marker = "secondary_not_visible"

        row_oracle_gain = int(oracle_hit) - int(primary_oracle_hit) if gate_triggered else 0
        row_judge_gain = int(judge_ok) - int(primary_judge_ok) if gate_triggered else 0
        oracle_gain_count += row_oracle_gain
        judge_gain_count += row_judge_gain
        if gate_triggered:
            typed_family_breakdown[secondary_family]["oracle_gain_count"] += row_oracle_gain
            typed_family_breakdown[secondary_family]["judge_gain_count"] += row_judge_gain
            direct_fact_bucket_breakdown[direct_fact_bucket]["oracle_gain_count"] += row_oracle_gain
            direct_fact_bucket_breakdown[direct_fact_bucket]["judge_gain_count"] += row_judge_gain
            for subtype in constraint_subtypes:
                constraint_subtype_breakdown[subtype]["oracle_gain_count"] += row_oracle_gain
                constraint_subtype_breakdown[subtype]["judge_gain_count"] += row_judge_gain

        rows.append(
            {
                "id": sample["id"],
                "category": category,
                "clue": clue,
                "secondary_family": secondary_family,
                "direct_fact_bucket": direct_fact_bucket,
                "constraint_subtypes": constraint_subtypes,
                "gold": gold,
                "branch_details": branch_details,
                "primary_raw_choices": primary_raw_choices,
                "primary_unique_candidates": primary_unique_candidates,
                "primary_candidate_counts": dict(primary_candidate_counts),
                "raw_choices": raw_choices,
                "unique_candidates": unique_candidates,
                "candidate_counts": dict(candidate_counts),
                "primary_pool_summary": primary_pool_summary,
                "constraint_rerank": args.constraint_rerank,
                "primary_constraint_debug": primary_constraint_debug,
                "constraint_debug": constraint_debug,
                "primary_judge_candidates": primary_judge_candidates,
                "judge_candidates": judge_candidates,
                "judge_candidates_visible": truncated_candidates,
                "secondary_gate_triggered": gate_triggered,
                "vote_candidate": vote_candidate,
                "vote_correct": vote_ok,
                "primary_judge_raw": primary_judge_raw,
                "primary_judge_candidate": primary_judge_candidate,
                "primary_judge_correct": primary_judge_ok,
                "judge_raw": judge_raw,
                "judge_candidate": judge_candidate,
                "judge_correct": judge_ok,
                "pairwise_triggered_debug": pairwise_triggered_debug,
                "primary_oracle_hit": primary_oracle_hit,
                "oracle_hit": oracle_hit,
                "secondary_only_candidates": secondary_only_candidates,
                "judge_secondary_visible": judge_secondary_visible,
                "judge_selected_from_secondary": judge_selected_from_secondary,
                "oracle_gain_on_triggered": row_oracle_gain,
                "judge_gain_on_triggered": row_judge_gain,
                **(
                    {
                        "gold_presence_bucket": gold_presence_bucket,
                        "selection_risk_marker": selection_risk_marker,
                        **order_sensitivity,
                    }
                    if args.diagnosis_mode
                    else {}
                ),
            }
        )

    for bucket in typed_family_breakdown.values():
        finalize_metric_bucket(bucket)
    for bucket in direct_fact_bucket_breakdown.values():
        finalize_metric_bucket(bucket)
    for bucket in constraint_subtype_breakdown.values():
        finalize_metric_bucket(bucket)

    report = {
        "seed": args.seed,
        "sample_limit": len(holdout),
        "direct_fact_bucket_filter": args.direct_fact_bucket_filter,
        "profile": profile_name,
        "retrieval_mode": args.retrieval_mode,
        "n_candidates": args.n_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "judge_mode": args.judge_mode,
        "constraint_rerank": args.constraint_rerank,
        "generation_hint_mode": (
            args.generation_hint_mode if args.generation_hint_mode != "off"
            else ("full" if args.use_generation_hints else "off")
        ),
        "use_generation_hints": args.use_generation_hints,
        "secondary_profile": secondary_profile_name if secondary_enabled else None,
        "secondary_retrieval_mode": secondary_retrieval_mode if secondary_enabled else None,
        "secondary_source": args.secondary_source if secondary_enabled else None,
        "secondary_n_candidates": args.secondary_n_candidates,
        "secondary_temperature": secondary_temperature if secondary_enabled else None,
        "secondary_top_p": secondary_top_p if secondary_enabled else None,
        "secondary_use_generation_hints": args.secondary_use_generation_hints if secondary_enabled else False,
        "secondary_generation_hint_mode": (
            resolve_generation_hint_mode(
                args.secondary_use_generation_hints,
                args.secondary_generation_hint_mode,
            )
            if secondary_enabled
            else None
        ),
        "secondary_typed_route": args.secondary_typed_route if secondary_enabled else None,
        "secondary_merge_mode": args.secondary_merge_mode if secondary_enabled else None,
        "secondary_gate": args.secondary_gate if secondary_enabled else None,
        "secondary_min_unique": args.secondary_min_unique if secondary_enabled else None,
        "secondary_min_entropy": args.secondary_min_entropy if secondary_enabled else None,
        "secondary_family_allowlist": (
            sorted(secondary_family_allowlist) if secondary_family_allowlist is not None else None
        ),
        "secondary_constraint_allowlist": (
            sorted(secondary_constraint_allowlist) if secondary_constraint_allowlist is not None else None
        ),
        "append_unique_reserved_secondary_slots": (
            args.append_unique_reserved_secondary_slots if secondary_enabled else None
        ),
        "pairwise_triggered_mode": args.pairwise_triggered_mode,
        "pairwise_triggered_max_secondary": args.pairwise_triggered_max_secondary,
        "secondary_triggered_count": secondary_triggered if secondary_enabled else 0,
        "secondary_skipped_count": secondary_skipped if secondary_enabled else 0,
        "typed_family_breakdown": typed_family_breakdown,
        "direct_fact_bucket_breakdown": direct_fact_bucket_breakdown,
        "constraint_subtype_breakdown": constraint_subtype_breakdown,
        "example_pool_limit": resolved_examples_limit,
        "vote_accuracy": vote_correct / len(holdout) if holdout else 0.0,
        "primary_judge_accuracy": primary_judge_correct / len(holdout) if holdout else 0.0,
        "judge_accuracy": judge_correct / len(holdout) if holdout else 0.0,
        "primary_oracle_hit_rate": primary_oracle_hits / len(holdout) if holdout else 0.0,
        "oracle_hit_rate": oracle_hits / len(holdout) if holdout else 0.0,
        "judge_secondary_visible_count": judge_secondary_visible_count if secondary_enabled else 0,
        "judge_selected_from_secondary_count": judge_selected_from_secondary_count if secondary_enabled else 0,
        "oracle_gain_on_triggered": oracle_gain_count / secondary_triggered if secondary_triggered else 0.0,
        "judge_gain_on_triggered": judge_gain_count / secondary_triggered if secondary_triggered else 0.0,
        "diagnosis_mode": args.diagnosis_mode,
        "rows": rows,
    }

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
