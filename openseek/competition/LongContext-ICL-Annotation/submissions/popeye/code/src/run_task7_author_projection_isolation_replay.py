import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import method
from main import TASK_FILES
from method import (
    _choose_task7_vote_candidate,
    _collect_task7_branch_choices,
    _dedupe_task7_candidates,
    _request_nvidia_completion,
    _summarize_task7_candidate_pool,
    build_prompt,
    build_task7_retrieval_context,
    count_answer,
    detect_task7_secondary_family,
    extract_task7_category_constraints,
    get_example_pool_limit,
    get_profile_name,
    get_profile_retrieval_tasks,
    get_task7_secondary_only_candidates,
    normalize_task7_answer,
    parse_task7_fields,
    reorder_task7_examples_by_lexical_retrieval,
    resolve_task7_rerank_judge,
    select_examples_with_metadata,
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


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_TARGET_REPRO_PATH = WORK_LOGS_DIR / "task7_author_projection_target_repro_ledger_2026-04-09.json"
DEFAULT_TRACE_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_author_row_trace_2026-04-08.json"
DEFAULT_DIFF_AUDIT_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_diff_audit_2026-04-08.json"
DEFAULT_ISOLATION_AUDIT_PATH = WORK_LOGS_DIR / "task7_author_projection_refresh_seed2026_isolation_audit_2026-04-08.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_author_projection_instrumented_isolation_replay_2026-04-09.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_author_projection_instrumented_isolation_replay_2026-04-09.md"
DEFAULT_REFRESH_PROFILE = "frontier_task6_task7"
DEFAULT_REFRESH_SECONDARY_PROFILE = "long_context"

TASK7_RELEVANT_ENV_VARS = [
    "OPENSEEK_PROFILE",
    "OPENSEEK_EXAMPLES_LIMIT",
    "OPENSEEK_CONTEXT_BUDGET",
    "OPENSEEK_VLLM_URL",
    "OPENSEEK_MODEL_NAME",
    "OPENSEEK_COMPLETION_SEED",
    "OPENSEEK_REQUEST_TIMEOUT",
    "OPENSEEK_TASK7_RERANK",
    "OPENSEEK_TASK7_RERANK_RETRIEVAL_MODE",
    "OPENSEEK_TASK7_RERANK_CANDIDATES",
    "OPENSEEK_TASK7_RERANK_MAX_JUDGE",
    "OPENSEEK_TASK7_RERANK_TEMPERATURE",
    "OPENSEEK_TASK7_RERANK_TOP_P",
    "OPENSEEK_TASK7_RERANK_HINTS",
    "OPENSEEK_TASK7_RERANK_HINT_MODE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_PROFILE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_RETRIEVAL_MODE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_CANDIDATES",
    "OPENSEEK_TASK7_RERANK_SECONDARY_TEMPERATURE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_TOP_P",
    "OPENSEEK_TASK7_RERANK_SECONDARY_HINTS",
    "OPENSEEK_TASK7_RERANK_SECONDARY_HINT_MODE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_TYPED_ROUTE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_MERGE_MODE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_GATE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_MIN_UNIQUE",
    "OPENSEEK_TASK7_RERANK_SECONDARY_MIN_ENTROPY",
    "OPENSEEK_TASK7_RERANK_SECONDARY_FAMILY_ALLOWLIST",
    "OPENSEEK_TASK7_RERANK_SECONDARY_CONSTRAINT_ALLOWLIST",
    "OPENSEEK_TASK7_RERANK_APPEND_UNIQUE_SECONDARY_JUDGE_SLOTS",
    "OPENSEEK_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT",
    "OPENSEEK_TASK7_AUTHOR_PROJECTION_DECISION_MODE",
    "OPENSEEK_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS",
    "OPENSEEK_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE",
    "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_PROJECTION",
    "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_ROUNDS",
    "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_N",
    "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_MAX_TOKENS",
    "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_TEMPERATURE",
    "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_TOP_P",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_repro_path", type=str, default=str(DEFAULT_TARGET_REPRO_PATH))
    parser.add_argument("--trace_path", type=str, default=str(DEFAULT_TRACE_PATH))
    parser.add_argument("--diff_audit_path", type=str, default=str(DEFAULT_DIFF_AUDIT_PATH))
    parser.add_argument("--isolation_audit_path", type=str, default=str(DEFAULT_ISOLATION_AUDIT_PATH))
    parser.add_argument("--task7_data_path", type=str, default=str(TASK_FILES[7]))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--profile", type=str, default=DEFAULT_REFRESH_PROFILE)
    parser.add_argument("--examples_limit", type=int, default=None)
    parser.add_argument("--primary_retrieval_mode", type=str, default="none", choices=["none", "lexical"])
    parser.add_argument("--primary_candidates", type=int, default=8)
    parser.add_argument("--primary_temperature", type=float, default=0.9)
    parser.add_argument("--primary_top_p", type=float, default=0.95)
    parser.add_argument("--primary_use_hints", action="store_true")
    parser.add_argument("--primary_hint_mode", type=str, default="off")
    parser.add_argument("--secondary_profile", type=str, default=DEFAULT_REFRESH_SECONDARY_PROFILE)
    parser.add_argument("--secondary_retrieval_mode", type=str, default="none", choices=["none", "lexical"])
    parser.add_argument("--secondary_candidates", type=int, default=4)
    parser.add_argument("--secondary_temperature", type=float, default=0.9)
    parser.add_argument("--secondary_top_p", type=float, default=0.95)
    parser.add_argument("--secondary_use_hints", action="store_true")
    parser.add_argument("--secondary_hint_mode", type=str, default="off")
    parser.add_argument("--secondary_typed_route", type=str, default="off")
    parser.add_argument("--secondary_merge_mode", type=str, default="append_unique", choices=["counts", "append_unique"])
    parser.add_argument("--secondary_gate", type=str, default="unique_only")
    parser.add_argument("--secondary_min_unique", type=int, default=7)
    parser.add_argument("--secondary_min_entropy", type=float, default=1.3)
    parser.add_argument("--max_judge_candidates", type=int, default=8)
    parser.add_argument("--reserved_secondary_slots", type=int, default=3)
    parser.add_argument(
        "--author_projection_judge_layout",
        type=str,
        default="baseline",
        choices=["baseline", "author_secondary_first", "author_primary_anchor_top1", "author_primary_anchor_top2"],
    )
    parser.add_argument(
        "--author_projection_decision_mode",
        type=str,
        default="baseline",
        choices=["baseline", "anchor1", "anchor1_stability_gate"],
    )
    parser.add_argument("--author_projection_stability_repeats", type=int, default=3)
    parser.add_argument(
        "--author_projection_pairwise_gate",
        type=str,
        default="strong_only",
        choices=["strong_only"],
    )
    parser.add_argument("--author_direct_fact_rounds", type=int, default=3)
    parser.add_argument("--author_direct_fact_n", type=int, default=6)
    parser.add_argument("--author_direct_fact_max_tokens", type=int, default=32)
    parser.add_argument("--author_direct_fact_temperature", type=float, default=0.9)
    parser.add_argument("--author_direct_fact_top_p", type=float, default=0.95)
    parser.add_argument("--completion_seed", type=str, default="2026")
    parser.add_argument("--completion_url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--request_timeout", type=int, default=None)
    parser.add_argument("--sample_ids", nargs="*", default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def build_env_snapshot() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in TASK7_RELEVANT_ENV_VARS}


def take_lines(text: str, limit: int = 12) -> str:
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    return "\n".join(lines[:limit])


def build_sample_payload(sample: dict) -> dict:
    payload = dict(sample)
    payload["input_fingerprint"] = sha256_text(sample["input"])
    return payload


def build_examples_fingerprint(examples: list[dict]) -> dict:
    ids = [example["id"] for example in examples]
    author_examples = [
        {
            "id": example["id"],
            "output": example["output"][0],
        }
        for example in examples
    ]
    return {
        "count": len(examples),
        "ids_head": ids[:10],
        "ids_tail": ids[-10:] if len(ids) > 10 else ids[:],
        "ids_fingerprint": sha256_json(ids),
        "content_fingerprint": sha256_json(
            [
                {
                    "id": example["id"],
                    "input": example["input"],
                    "output": example["output"][0],
                }
                for example in examples
            ]
        ),
        "author_surface_fingerprint": sha256_json(author_examples),
    }


def build_author_catalog_summary(examples: list[dict]) -> tuple[dict, dict]:
    author_answers = []
    author_example_ids = []
    for example in examples:
        category, clue = parse_task7_fields(example["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        bucket = detect_direct_fact_bucket(category, clue, family)
        if bucket == "quoted_work_author_relation":
            author_answers.append(example["output"][0])
            author_example_ids.append(example["id"])
    catalog = build_author_answer_catalog(author_answers)
    summary = {
        "quoted_author_example_count": len(author_answers),
        "quoted_author_example_ids_head": author_example_ids[:10],
        "quoted_author_example_ids_fingerprint": sha256_json(author_example_ids),
        "catalog_counts": {
            "exact_map": len(catalog["exact_map"]),
            "initials_map": len(catalog["initials_map"]),
            "surname_only_map": len(catalog["surname_only_map"]),
        },
        "catalog_fingerprint": sha256_json(catalog),
    }
    return catalog, summary


def configure_method_runtime(args: argparse.Namespace) -> dict:
    method.DEFAULT_TASK7_RERANK_ENABLED = True
    method.DEFAULT_TASK7_RERANK_RETRIEVAL_MODE = args.primary_retrieval_mode
    method.DEFAULT_TASK7_RERANK_CANDIDATES = args.primary_candidates
    method.DEFAULT_TASK7_RERANK_MAX_JUDGE = args.max_judge_candidates
    method.DEFAULT_TASK7_RERANK_TEMPERATURE = args.primary_temperature
    method.DEFAULT_TASK7_RERANK_TOP_P = args.primary_top_p
    method.DEFAULT_TASK7_RERANK_HINTS = bool(args.primary_use_hints)
    method.DEFAULT_TASK7_RERANK_HINT_MODE = args.primary_hint_mode
    method.DEFAULT_TASK7_RERANK_SECONDARY_PROFILE = "" if args.secondary_profile in {"", "none"} else args.secondary_profile
    method.DEFAULT_TASK7_RERANK_SECONDARY_RETRIEVAL_MODE = args.secondary_retrieval_mode
    method.DEFAULT_TASK7_RERANK_SECONDARY_CANDIDATES = args.secondary_candidates
    method.DEFAULT_TASK7_RERANK_SECONDARY_TEMPERATURE = args.secondary_temperature
    method.DEFAULT_TASK7_RERANK_SECONDARY_TOP_P = args.secondary_top_p
    method.DEFAULT_TASK7_RERANK_SECONDARY_HINTS = bool(args.secondary_use_hints)
    method.DEFAULT_TASK7_RERANK_SECONDARY_HINT_MODE = args.secondary_hint_mode
    method.DEFAULT_TASK7_RERANK_SECONDARY_TYPED_ROUTE = args.secondary_typed_route
    method.DEFAULT_TASK7_RERANK_SECONDARY_MERGE_MODE = args.secondary_merge_mode
    method.DEFAULT_TASK7_RERANK_SECONDARY_GATE = args.secondary_gate
    method.DEFAULT_TASK7_RERANK_SECONDARY_MIN_UNIQUE = args.secondary_min_unique
    method.DEFAULT_TASK7_RERANK_SECONDARY_MIN_ENTROPY = args.secondary_min_entropy
    method.DEFAULT_TASK7_RERANK_APPEND_UNIQUE_SECONDARY_JUDGE_SLOTS = args.reserved_secondary_slots
    method.DEFAULT_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT = args.author_projection_judge_layout
    method.DEFAULT_TASK7_AUTHOR_PROJECTION_DECISION_MODE = args.author_projection_decision_mode
    method.DEFAULT_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS = args.author_projection_stability_repeats
    method.DEFAULT_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE = args.author_projection_pairwise_gate
    method.DEFAULT_TASK7_AUTHOR_DIRECT_FACT_ROUNDS = args.author_direct_fact_rounds
    method.DEFAULT_TASK7_AUTHOR_DIRECT_FACT_N = args.author_direct_fact_n
    method.DEFAULT_TASK7_AUTHOR_DIRECT_FACT_MAX_TOKENS = args.author_direct_fact_max_tokens
    method.DEFAULT_TASK7_AUTHOR_DIRECT_FACT_TEMPERATURE = args.author_direct_fact_temperature
    method.DEFAULT_TASK7_AUTHOR_DIRECT_FACT_TOP_P = args.author_direct_fact_top_p
    if args.completion_seed is not None:
        method.DEFAULT_COMPLETION_SEED = args.completion_seed
    if args.completion_url:
        method.DEFAULT_COMPLETION_URL = args.completion_url
    if args.model_name:
        method.DEFAULT_MODEL_NAME = args.model_name
    if args.request_timeout is not None:
        method.DEFAULT_REQUEST_TIMEOUT = args.request_timeout
    return {
        "profile": get_profile_name(args.profile),
        "examples_limit": args.examples_limit,
        "primary_retrieval_mode": args.primary_retrieval_mode,
        "primary_candidates": args.primary_candidates,
        "primary_temperature": args.primary_temperature,
        "primary_top_p": args.primary_top_p,
        "primary_use_hints": bool(args.primary_use_hints),
        "primary_hint_mode": args.primary_hint_mode,
        "secondary_profile": method.DEFAULT_TASK7_RERANK_SECONDARY_PROFILE or None,
        "secondary_retrieval_mode": args.secondary_retrieval_mode,
        "secondary_candidates": args.secondary_candidates,
        "secondary_temperature": args.secondary_temperature,
        "secondary_top_p": args.secondary_top_p,
        "secondary_use_hints": bool(args.secondary_use_hints),
        "secondary_hint_mode": args.secondary_hint_mode,
        "secondary_typed_route": args.secondary_typed_route,
        "secondary_merge_mode": args.secondary_merge_mode,
        "secondary_gate": args.secondary_gate,
        "secondary_min_unique": args.secondary_min_unique,
        "secondary_min_entropy": args.secondary_min_entropy,
        "max_judge_candidates": args.max_judge_candidates,
        "reserved_secondary_slots": args.reserved_secondary_slots,
        "author_projection_judge_layout": args.author_projection_judge_layout,
        "author_projection_decision_mode": args.author_projection_decision_mode,
        "author_projection_stability_repeats": args.author_projection_stability_repeats,
        "author_projection_pairwise_gate": args.author_projection_pairwise_gate,
        "author_direct_fact_rounds": args.author_direct_fact_rounds,
        "author_direct_fact_n": args.author_direct_fact_n,
        "author_direct_fact_max_tokens": args.author_direct_fact_max_tokens,
        "author_direct_fact_temperature": args.author_direct_fact_temperature,
        "author_direct_fact_top_p": args.author_direct_fact_top_p,
        "completion_seed": method.DEFAULT_COMPLETION_SEED,
        "completion_url": method.DEFAULT_COMPLETION_URL,
        "model_name": method.DEFAULT_MODEL_NAME,
        "request_timeout": method.DEFAULT_REQUEST_TIMEOUT,
    }


def should_try_secondary(
    *,
    category: str,
    secondary_input_prompt: str | None,
    secondary_candidates: int,
    pool_summary: dict[str, float | int],
    secondary_hint_mode: str,
    category_constraints: list[str],
    secondary_family: str,
    secondary_family_allowlist: set[str] | None,
    secondary_constraint_allowlist: set[str] | None,
) -> tuple[bool, list[str]]:
    reasons = []
    allowed = True
    if not secondary_input_prompt:
        reasons.append("missing_secondary_prompt")
        allowed = False
    if secondary_candidates <= 0:
        reasons.append("secondary_candidates_le_zero")
        allowed = False
    if allowed and not method._should_trigger_task7_secondary_gate(pool_summary):
        reasons.append("secondary_gate_blocked")
        allowed = False
    if allowed and secondary_hint_mode == "constraint_only" and not category_constraints:
        reasons.append("constraint_only_without_constraints")
        allowed = False
    if allowed and secondary_hint_mode == "typed" and secondary_family == "generic":
        reasons.append("typed_secondary_family_generic")
        allowed = False
    if allowed and secondary_hint_mode == "typed":
        if not method.is_task7_secondary_family_allowed(
            secondary_family,
            allowlist=secondary_family_allowlist,
        ):
            reasons.append("typed_family_not_allowlisted")
            allowed = False
    if allowed and secondary_hint_mode == "typed" and secondary_family == "constraint":
        if not method.is_task7_constraint_secondary_allowed(
            category,
            allowlist=secondary_constraint_allowlist,
        ):
            reasons.append("constraint_secondary_not_allowlisted")
            allowed = False
    return allowed, reasons


def build_row_context(
    *,
    task_description: str,
    text2annotate: str,
    primary_examples: list[dict],
    primary_profile: str,
    primary_retrieval_mode: str,
    secondary_examples: list[dict] | None,
    secondary_profile: str | None,
    secondary_retrieval_mode: str,
) -> dict:
    prompt_template = build_prompt(
        task_description,
        text2annotate,
        task_id=7,
        profile_name=primary_profile,
    )
    profile_retrieval_tasks = get_profile_retrieval_tasks(primary_profile)
    use_profile_retrieval = 7 in profile_retrieval_tasks
    use_primary_retrieval = primary_retrieval_mode == "lexical" or use_profile_retrieval
    ordered_primary_examples = primary_examples
    if use_primary_retrieval:
        ordered_primary_examples = reorder_task7_examples_by_lexical_retrieval(
            primary_examples,
            text2annotate,
            build_task7_retrieval_context(primary_examples),
        )
    primary_examples_meta = select_examples_with_metadata(
        ordered_primary_examples,
        task_description,
        text2annotate,
        task_id=7,
        profile_name=primary_profile,
    )
    primary_used_examples = ordered_primary_examples[: primary_examples_meta["used_examples"]]
    input_prompt = prompt_template.replace("[[EXAMPLES]]\n\n", primary_examples_meta["examples_str"] + "\n\n")

    secondary_prompt = None
    secondary_examples_meta = None
    secondary_used_examples = None
    ordered_secondary_examples = secondary_examples
    if secondary_examples and secondary_profile:
        secondary_prompt_template = build_prompt(
            task_description,
            text2annotate,
            task_id=7,
            profile_name=secondary_profile,
        )
        if secondary_retrieval_mode == "lexical":
            ordered_secondary_examples = reorder_task7_examples_by_lexical_retrieval(
                secondary_examples,
                text2annotate,
                build_task7_retrieval_context(secondary_examples),
            )
        secondary_examples_meta = select_examples_with_metadata(
            ordered_secondary_examples,
            task_description,
            text2annotate,
            task_id=7,
            profile_name=secondary_profile,
        )
        secondary_used_examples = ordered_secondary_examples[: secondary_examples_meta["used_examples"]]
        secondary_prompt = secondary_prompt_template.replace(
            "[[EXAMPLES]]\n\n",
            secondary_examples_meta["examples_str"] + "\n\n",
        )

    return {
        "input_prompt": input_prompt,
        "secondary_input_prompt": secondary_prompt,
        "primary_prompt": {
            "prompt_fingerprint": sha256_text(input_prompt),
            "prompt_head": take_lines(input_prompt),
            "selected_examples": build_examples_fingerprint(primary_used_examples),
            "selection_metadata": primary_examples_meta,
        },
        "secondary_prompt": None
        if secondary_prompt is None or secondary_examples_meta is None or secondary_used_examples is None
        else {
            "prompt_fingerprint": sha256_text(secondary_prompt),
            "prompt_head": take_lines(secondary_prompt),
            "selected_examples": build_examples_fingerprint(secondary_used_examples),
            "selection_metadata": secondary_examples_meta,
        },
    }


def run_variant(
    *,
    text2annotate: str,
    input_prompt: str,
    secondary_input_prompt: str | None,
    author_catalog: dict | None,
    enable_author_projection: bool,
    effective_config: dict,
) -> dict:
    category, clue = parse_task7_fields(text2annotate)
    category_constraints = extract_task7_category_constraints(category)
    secondary_family = detect_task7_secondary_family(
        category,
        clue,
        typed_route=effective_config["secondary_typed_route"],
    )
    direct_fact_family = detect_task7_secondary_family(category, clue, typed_route="auto")
    bucket = detect_direct_fact_bucket(category, clue, direct_fact_family)

    primary_raw_choices = _collect_task7_branch_choices(
        input_prompt,
        text2annotate,
        n_candidates=effective_config["primary_candidates"],
        temperature=effective_config["primary_temperature"],
        top_p=effective_config["primary_top_p"],
        use_hints=effective_config["primary_use_hints"],
        hint_mode=effective_config["primary_hint_mode"],
    )
    primary_unique_candidates, primary_candidate_counts = _dedupe_task7_candidates(primary_raw_choices)
    primary_pool_summary = _summarize_task7_candidate_pool(primary_candidate_counts)
    unique_candidates = list(primary_unique_candidates)
    candidate_counts = dict(primary_candidate_counts)
    vote_candidate = _choose_task7_vote_candidate(unique_candidates, candidate_counts)

    secondary_unique_candidates: list[str] = []
    secondary_candidate_counts: dict[str, int] = {}
    secondary_raw_choices: list[str] = []
    secondary_family_allowlist = method.get_task7_rerank_secondary_family_allowlist()
    secondary_constraint_allowlist = method.get_task7_rerank_secondary_constraint_allowlist()
    secondary_allowed, secondary_skip_reasons = should_try_secondary(
        category=category,
        secondary_input_prompt=secondary_input_prompt,
        secondary_candidates=effective_config["secondary_candidates"],
        pool_summary=primary_pool_summary,
        secondary_hint_mode=effective_config["secondary_hint_mode"],
        category_constraints=category_constraints,
        secondary_family=secondary_family,
        secondary_family_allowlist=secondary_family_allowlist,
        secondary_constraint_allowlist=secondary_constraint_allowlist,
    )
    if secondary_allowed and secondary_input_prompt:
        secondary_raw_choices = _collect_task7_branch_choices(
            secondary_input_prompt,
            text2annotate,
            n_candidates=effective_config["secondary_candidates"],
            temperature=effective_config["secondary_temperature"],
            top_p=effective_config["secondary_top_p"],
            use_hints=effective_config["secondary_use_hints"],
            hint_mode=effective_config["secondary_hint_mode"],
        )
        secondary_unique_candidates, secondary_candidate_counts = _dedupe_task7_candidates(secondary_raw_choices)
        if effective_config["secondary_merge_mode"] == "append_unique":
            unique_candidates = method._append_task7_unique_candidates(primary_unique_candidates, secondary_unique_candidates)
            for normalized, count in secondary_candidate_counts.items():
                candidate_counts.setdefault(normalized, count)
            vote_candidate = _choose_task7_vote_candidate(primary_unique_candidates, primary_candidate_counts)
        else:
            combined_raw_choices = list(primary_raw_choices)
            combined_raw_choices.extend(secondary_raw_choices)
            unique_candidates, candidate_counts = _dedupe_task7_candidates(combined_raw_choices)
            vote_candidate = _choose_task7_vote_candidate(unique_candidates, candidate_counts)

    direct_fact_reports = []
    direct_fact_raw_candidates: list[str] = []
    author_projected_candidates: list[str] = []
    projection_trace: list[dict] = []
    if enable_author_projection and author_catalog and bucket == "quoted_work_author_relation":
        for question in build_direct_fact_questions(category, clue, direct_fact_family):
            report = collect_question_candidates(
                question=question,
                family=direct_fact_family,
                rounds=effective_config["author_direct_fact_rounds"],
                n=effective_config["author_direct_fact_n"],
                temperature=effective_config["author_direct_fact_temperature"],
                top_p=effective_config["author_direct_fact_top_p"],
                max_tokens=effective_config["author_direct_fact_max_tokens"],
            )
            direct_fact_reports.append(report)
        direct_fact_raw_candidates = aggregate_direct_fact_candidates(direct_fact_reports)
        author_projected_candidates, projection_trace = project_direct_fact_candidates_to_author_catalog(
            direct_fact_raw_candidates,
            author_catalog,
        )
        if author_projected_candidates:
            unique_candidates = method._append_task7_unique_candidates(unique_candidates, author_projected_candidates)
            for candidate in author_projected_candidates:
                normalized = normalize_task7_answer(candidate)
                if normalized:
                    candidate_counts.setdefault(normalized, 1)

    judge_decision = resolve_task7_rerank_judge(
        category=category,
        clue=clue,
        bucket=bucket,
        primary_candidates=primary_unique_candidates,
        secondary_candidates=secondary_unique_candidates,
        author_projected_candidates=author_projected_candidates,
        unique_candidates=unique_candidates,
        max_candidates=effective_config["max_judge_candidates"],
        reserved_secondary_slots=effective_config["reserved_secondary_slots"],
        append_unique_secondary_active=(
            secondary_allowed
            and effective_config["secondary_merge_mode"] == "append_unique"
            and bool(secondary_unique_candidates)
        ),
    )

    prediction = judge_decision["final_prediction"]
    prediction_source = "judge" if prediction is not None else None
    fallback_raw = ""
    if prediction is None and vote_candidate is not None:
        prediction = vote_candidate
        prediction_source = "vote"
    if prediction is None:
        fallback_raw = _request_nvidia_completion(input_prompt, task_id=7)
        prediction = count_answer(fallback_raw, task_id=7)
        prediction_source = "fallback"

    return {
        "enabled_author_projection": enable_author_projection,
        "effective_env_override": {
            "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_PROJECTION": "1" if enable_author_projection else "0",
        },
        "family": direct_fact_family,
        "secondary_family": secondary_family,
        "bucket": bucket,
        "category_constraints": category_constraints,
        "primary": {
            "raw_choices": primary_raw_choices,
            "unique_candidates": primary_unique_candidates,
            "candidate_counts": primary_candidate_counts,
            "pool_summary": primary_pool_summary,
            "vote_candidate": _choose_task7_vote_candidate(primary_unique_candidates, primary_candidate_counts),
        },
        "secondary": {
            "attempted": secondary_allowed,
            "skip_reasons": secondary_skip_reasons,
            "raw_choices": secondary_raw_choices,
            "unique_candidates": secondary_unique_candidates,
            "candidate_counts": secondary_candidate_counts,
        },
        "author_projection": {
            "bucket": bucket,
            "question_reports": direct_fact_reports,
            "raw_candidates": direct_fact_raw_candidates,
            "projected_candidates": author_projected_candidates,
            "projection_trace": projection_trace,
        },
        "judge": {
            "strategy": judge_decision["strategy"],
            "candidates": judge_decision["candidates"],
            "prompt_fingerprint": judge_decision["prompt_fingerprint"],
            "prompt_head": judge_decision["prompt_head"],
            "raw": judge_decision["raw"],
            "index": judge_decision["index"],
            "reserved_secondary_slots": judge_decision["reserved_secondary_slots"],
            "max_candidates": judge_decision["max_candidates"],
            "decision_mode": judge_decision["decision_mode"],
            "configured_layout_mode": judge_decision["configured_layout_mode"],
            "baseline_judge_candidates": judge_decision["baseline_judge_candidates"],
            "anchor1_judge_candidates": judge_decision["anchor1_judge_candidates"],
            "baseline_winner": judge_decision["baseline_winner"],
            "anchor1_winner": judge_decision["anchor1_winner"],
            "anchor1_repeat_winners": judge_decision["anchor1_repeat_winners"],
            "anchor1_repeat_stable": judge_decision["anchor1_repeat_stable"],
            "pairwise_gate_mode": judge_decision["pairwise_gate_mode"],
            "pairwise_gate_passed": judge_decision["pairwise_gate_passed"],
            "pairwise_gate_comparisons": judge_decision["pairwise_gate_comparisons"],
            "stability_repeats": judge_decision["stability_repeats"],
            "final_decision_source": judge_decision["final_decision_source"],
            "final_prediction": judge_decision["final_prediction"],
        },
        "prediction": prediction,
        "prediction_source": prediction_source,
        "fallback_raw": fallback_raw or None,
    }


def build_references(
    sample_id: str,
    *,
    target_repro_rows: dict[str, dict],
    trace_rows: dict[str, dict],
    diff_rows: dict[str, dict],
    isolation_rows: dict[str, dict],
) -> dict:
    target_row = target_repro_rows.get(sample_id)
    trace_row = trace_rows.get(sample_id)
    diff_row = diff_rows.get(sample_id)
    isolation_row = isolation_rows.get(sample_id)
    return {
        "target_repro": target_row,
        "trace": trace_row,
        "diff_audit": diff_row,
        "prior_isolation": isolation_row,
    }


def build_match_summary(references: dict, off_prediction: str | None, on_prediction: str | None) -> dict:
    trace_row = references.get("trace") or {}
    diff_row = references.get("diff_audit") or {}
    isolation_row = references.get("prior_isolation") or {}
    return {
        "changed_in_replay": off_prediction != on_prediction,
        "off_matches_trace_off": off_prediction == trace_row.get("off", {}).get("prediction"),
        "on_matches_trace_on": on_prediction == trace_row.get("on", {}).get("prediction"),
        "off_matches_full_run_off": off_prediction == diff_row.get("off_prediction"),
        "on_matches_full_run_on": on_prediction == diff_row.get("on_prediction"),
        "off_matches_prior_isolation_off": off_prediction == isolation_row.get("off_prediction"),
        "on_matches_prior_isolation_on": on_prediction == isolation_row.get("on_prediction"),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Author Projection Instrumented Isolation Replay",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Rows replayed: `{report['summary']['row_count']}`",
        f"- Off -> on changed rows: `{report['summary']['changed_in_replay_count']}`",
        f"- On matches trace-on: `{report['summary']['on_matches_trace_on_count']}`",
        f"- Off matches trace-off: `{report['summary']['off_matches_trace_off_count']}`",
        f"- Rows with empty replay author catalog: `{report['summary']['rows_with_empty_author_catalog_count']}`",
        "",
        "## Effective config",
        "",
    ]
    for key, value in report["effective_config"].items():
        lines.append(f"- `{key}` = `{value}`")
    lines.extend(
        [
            "",
            "## Row ledger",
            "",
        ]
    )
    for row in report["rows"]:
        lines.extend(
            [
                f"### {row['test_sample_id']}",
                "",
                f"- Category: `{row['category']}`",
                f"- Bucket: `{row['bucket']}`",
                f"- Full-run off -> on: `{row['references']['diff_audit'].get('off_prediction') if row['references']['diff_audit'] else None}` -> `{row['references']['diff_audit'].get('on_prediction') if row['references']['diff_audit'] else None}`",
                f"- Trace off -> on: `{row['references']['trace'].get('off', {}).get('prediction') if row['references']['trace'] else None}` -> `{row['references']['trace'].get('on', {}).get('prediction') if row['references']['trace'] else None}`",
                f"- Replay off -> on: `{row['off']['prediction']}` -> `{row['on']['prediction']}`",
                f"- Replay changed: `{row['match_summary']['changed_in_replay']}`",
                f"- Replay off matches trace-off: `{row['match_summary']['off_matches_trace_off']}`",
                f"- Replay on matches trace-on: `{row['match_summary']['on_matches_trace_on']}`",
                f"- Replay author projected candidates: `{', '.join(row['on']['author_projection']['projected_candidates']) if row['on']['author_projection']['projected_candidates'] else '(empty)'}`",
                f"- Replay judge candidates (on): `{', '.join(row['on']['judge']['candidates']) if row['on']['judge']['candidates'] else '(empty)'}`",
                f"- Decision mode (on): `{row['on']['judge']['decision_mode']}`",
                f"- Baseline winner (on): `{row['on']['judge']['baseline_winner']}`",
                f"- Anchor1 winner (on): `{row['on']['judge']['anchor1_winner']}`",
                f"- Anchor1 repeat winners (on): `{', '.join(w for w in row['on']['judge']['anchor1_repeat_winners'] if w) if row['on']['judge']['anchor1_repeat_winners'] else '(empty)'}`",
                f"- Anchor1 repeat stable (on): `{row['on']['judge']['anchor1_repeat_stable']}`",
                f"- Pairwise gate passed (on): `{row['on']['judge']['pairwise_gate_passed']}`",
                f"- Final decision source (on): `{row['on']['judge']['final_decision_source']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    effective_config = configure_method_runtime(args)

    task7 = load_json(args.task7_data_path)
    target_repro = load_json(args.target_repro_path)
    trace_report = load_json(args.trace_path)
    diff_audit = load_json(args.diff_audit_path)
    isolation_audit = load_json(args.isolation_audit_path)

    task_description = task7["Definition"][0]
    test_samples = {sample["id"]: sample for sample in task7["test_samples"]}
    target_repro_rows = {row["test_sample_id"]: row for row in target_repro["target_rows"]}

    replay_targets = target_repro["target_rows"]
    if args.sample_ids:
        replay_targets = []
        for sample_id in args.sample_ids:
            if sample_id not in test_samples:
                raise KeyError(f"Unknown Task7 sample_id requested: {sample_id}")
            replay_targets.append(
                target_repro_rows.get(
                    sample_id,
                    {
                        "test_sample_id": sample_id,
                        "target_bucket": False,
                        "bucket": None,
                    },
                )
            )

    resolved_examples_limit = get_example_pool_limit(
        task_id=7,
        examples_limit=args.examples_limit,
        profile_name=effective_config["profile"],
    )
    primary_examples = task7["examples"][:resolved_examples_limit]
    primary_examples_fingerprint = build_examples_fingerprint(primary_examples)
    author_catalog, author_catalog_summary = build_author_catalog_summary(primary_examples)

    secondary_profile = effective_config["secondary_profile"]
    secondary_examples = None
    secondary_examples_fingerprint = None
    if secondary_profile and effective_config["secondary_candidates"] > 0:
        secondary_limit = get_example_pool_limit(
            task_id=7,
            examples_limit=args.examples_limit,
            profile_name=secondary_profile,
        )
        secondary_examples = task7["examples"][:secondary_limit]
        secondary_examples_fingerprint = build_examples_fingerprint(secondary_examples)

    trace_rows = {row["test_sample_id"]: row for row in trace_report["rows"]}
    diff_rows = {row["id"]: row for row in diff_audit["rows"]}
    isolation_rows = {row["test_sample_id"]: row for row in isolation_audit["rows"]}

    replay_rows = []
    for target_row in replay_targets:
        sample_id = target_row["test_sample_id"]
        sample = test_samples[sample_id]
        category, clue = parse_task7_fields(sample["input"])
        bucket = detect_direct_fact_bucket(
            category,
            clue,
            detect_task7_secondary_family(category, clue, typed_route="auto"),
        )
        row_context = build_row_context(
            task_description=task_description,
            text2annotate=sample["input"],
            primary_examples=primary_examples,
            primary_profile=effective_config["profile"],
            primary_retrieval_mode=effective_config["primary_retrieval_mode"],
            secondary_examples=secondary_examples,
            secondary_profile=secondary_profile,
            secondary_retrieval_mode=effective_config["secondary_retrieval_mode"],
        )
        off_result = run_variant(
            text2annotate=sample["input"],
            input_prompt=row_context["input_prompt"],
            secondary_input_prompt=row_context["secondary_input_prompt"],
            author_catalog=None,
            enable_author_projection=False,
            effective_config=effective_config,
        )
        on_result = run_variant(
            text2annotate=sample["input"],
            input_prompt=row_context["input_prompt"],
            secondary_input_prompt=row_context["secondary_input_prompt"],
            author_catalog=author_catalog,
            enable_author_projection=True,
            effective_config=effective_config,
        )
        references = build_references(
            sample_id,
            target_repro_rows=target_repro_rows,
            trace_rows=trace_rows,
            diff_rows=diff_rows,
            isolation_rows=isolation_rows,
        )
        replay_rows.append(
            {
                "test_sample_id": sample_id,
                "category": category,
                "clue": clue,
                "bucket": bucket,
                "row_payload": build_sample_payload(sample),
                "primary_example_pool": primary_examples_fingerprint,
                "secondary_example_pool": secondary_examples_fingerprint,
                "author_catalog": author_catalog_summary,
                "prompt_context": {
                    "primary": row_context["primary_prompt"],
                    "secondary": row_context["secondary_prompt"],
                },
                "references": references,
                "off": off_result,
                "on": on_result,
                "match_summary": build_match_summary(
                    references,
                    off_result["prediction"],
                    on_result["prediction"],
                ),
            }
        )

    summary = {
        "row_count": len(replay_rows),
        "changed_in_replay_count": sum(int(row["match_summary"]["changed_in_replay"]) for row in replay_rows),
        "off_matches_trace_off_count": sum(int(row["match_summary"]["off_matches_trace_off"]) for row in replay_rows),
        "on_matches_trace_on_count": sum(int(row["match_summary"]["on_matches_trace_on"]) for row in replay_rows),
        "off_matches_full_run_off_count": sum(int(row["match_summary"]["off_matches_full_run_off"]) for row in replay_rows),
        "on_matches_full_run_on_count": sum(int(row["match_summary"]["on_matches_full_run_on"]) for row in replay_rows),
        "rows_with_empty_author_catalog_count": sum(
            int(row["author_catalog"]["quoted_author_example_count"] == 0) for row in replay_rows
        ),
    }

    report = {
        "generated_on": "2026-04-09",
        "runner": {
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "argv": sys.argv,
            "command": shlex.join([sys.executable, *sys.argv]),
        },
        "inputs": {
            "task7_data_path": str(args.task7_data_path),
            "target_repro_path": str(args.target_repro_path),
            "trace_path": str(args.trace_path),
            "diff_audit_path": str(args.diff_audit_path),
            "isolation_audit_path": str(args.isolation_audit_path),
        },
        "env_snapshot": build_env_snapshot(),
        "effective_config": effective_config,
        "assumption_note": (
            "Defaults are set to the closest reconstructible refresh-era Task7 rerank configuration from existing artifacts: "
            "frontier_task6_task7 primary path, long_context secondary path, gated append_unique judge layout, "
            "reserved secondary slots=3, and author direct-fact rounds=3."
        ),
        "summary": summary,
        "rows": replay_rows,
    }

    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
