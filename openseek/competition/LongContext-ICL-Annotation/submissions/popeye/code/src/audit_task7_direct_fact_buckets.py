import argparse
import json
from collections import defaultdict
from pathlib import Path

from main import TASK_FILES
from method import detect_task7_secondary_family, normalize_task7_answer, parse_task7_fields
from probe_task7_direct_fact_expansion import (
    aggregate_direct_fact_candidates,
    build_direct_fact_questions,
    collect_question_candidates,
    detect_direct_fact_bucket,
    load_json,
)
from validate_task7_candidate_rerank import (
    build_task7_prompt,
    collect_branch_choices,
    dedupe_candidates,
)
from task7_direct_fact_projection import (
    build_author_answer_catalog,
    project_direct_fact_candidates_to_author_catalog,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_direct_fact_bucket_audit_2026-04-01.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_direct_fact_bucket_audit_2026-04-01.md"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--profile", type=str, default="baseline")
    parser.add_argument("--retrieval_mode", type=str, default="none")
    parser.add_argument("--baseline_n_candidates", type=int, default=6)
    parser.add_argument("--direct_fact_rounds", type=int, default=2)
    parser.add_argument("--direct_fact_n", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--projection_mode", type=str, choices=["none", "author_catalog"], default="none")
    return parser.parse_args()


def init_bucket_summary() -> dict:
    return {
        "count": 0,
        "baseline_visible_count": 0,
        "direct_fact_visible_count": 0,
        "union_visible_count": 0,
        "direct_fact_gain_count": 0,
        "direct_fact_only_count": 0,
    }


def finalize_bucket_summary(summary: dict) -> dict:
    count = summary["count"]
    for key in (
        "baseline_visible_count",
        "direct_fact_visible_count",
        "union_visible_count",
        "direct_fact_gain_count",
        "direct_fact_only_count",
    ):
        rate_key = key.replace("_count", "_rate")
        summary[rate_key] = summary[key] / count if count else 0.0
    return summary


def build_baseline_candidates(
    *,
    task_description: str,
    sample: dict,
    examples: list[dict],
    profile_name: str,
    retrieval_mode: str,
    n_candidates: int,
    temperature: float,
    top_p: float,
) -> tuple[list[str], list[str]]:
    icl_examples = [example for example in examples if example["id"] != sample["id"]]
    prompt = build_task7_prompt(
        task_description=task_description,
        text2annotate=sample["input"],
        icl_examples=icl_examples,
        profile_name=profile_name,
        retrieval_mode=retrieval_mode,
    )
    _, raw_choices = collect_branch_choices(
        prompt=prompt,
        text2annotate=sample["input"],
        n_candidates=n_candidates,
        temperature=temperature,
        top_p=top_p,
        use_generation_hints=False,
        generation_hint_mode="off",
    )
    candidates, _ = dedupe_candidates(raw_choices)
    return raw_choices, candidates


def build_direct_fact_candidates(
    *,
    category: str,
    clue: str,
    family: str,
    bucket: str | None,
    author_catalog: dict | None,
    rounds: int,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[list[dict], list[str], list[str], list[dict]]:
    question_reports = []
    for question in build_direct_fact_questions(category, clue, family):
        report = collect_question_candidates(
            question=question,
            family=family,
            rounds=rounds,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        question_reports.append(report)
    candidate_pool = aggregate_direct_fact_candidates(question_reports)
    projected_pool = list(candidate_pool)
    projection_trace = []
    if bucket == "quoted_work_author_relation" and author_catalog:
        projected_pool, projection_trace = project_direct_fact_candidates_to_author_catalog(
            candidate_pool,
            author_catalog,
        )
    return question_reports, candidate_pool, projected_pool, projection_trace


def collect_author_catalog(task7: dict) -> dict:
    author_answers = []
    for sample in task7["examples"]:
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        if detect_direct_fact_bucket(category, clue, family) == "quoted_work_author_relation":
            author_answers.append(sample["output"][0])
    return build_author_answer_catalog(author_answers)


def evaluate_example_rows(args) -> tuple[dict, list[dict]]:
    task7 = load_json(Path(TASK_FILES[7]))
    examples = list(task7["examples"])
    task_description = task7["Definition"][0]
    author_catalog = collect_author_catalog(task7) if args.projection_mode == "author_catalog" else None

    bucket_summary = defaultdict(init_bucket_summary)
    rows = []
    for sample in examples:
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        bucket = detect_direct_fact_bucket(category, clue, family)
        if not bucket:
            continue

        baseline_raw, baseline_candidates = build_baseline_candidates(
            task_description=task_description,
            sample=sample,
            examples=examples,
            profile_name=args.profile,
            retrieval_mode=args.retrieval_mode,
            n_candidates=args.baseline_n_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        (
            question_reports,
            direct_fact_candidates_raw,
            direct_fact_candidates,
            projection_trace,
        ) = build_direct_fact_candidates(
            category=category,
            clue=clue,
            family=family,
            bucket=bucket,
            author_catalog=author_catalog,
            rounds=args.direct_fact_rounds,
            n=args.direct_fact_n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        gold = sample["output"][0]
        gold_norm = normalize_task7_answer(gold)
        baseline_hit = gold_norm in {normalize_task7_answer(candidate) for candidate in baseline_candidates}
        direct_fact_hit = gold_norm in {normalize_task7_answer(candidate) for candidate in direct_fact_candidates}
        union_hit = baseline_hit or direct_fact_hit

        summary = bucket_summary[bucket]
        summary["count"] += 1
        summary["baseline_visible_count"] += int(baseline_hit)
        summary["direct_fact_visible_count"] += int(direct_fact_hit)
        summary["union_visible_count"] += int(union_hit)
        summary["direct_fact_gain_count"] += int(direct_fact_hit and not baseline_hit)
        summary["direct_fact_only_count"] += int(direct_fact_hit and not baseline_hit)

        rows.append(
            {
                "id": sample["id"],
                "bucket": bucket,
                "family": family,
                "gold": gold,
                "category": category,
                "clue": clue,
                "baseline_visible": baseline_hit,
                "direct_fact_visible": direct_fact_hit,
                "union_visible": union_hit,
                "baseline_candidates": baseline_candidates,
                "direct_fact_candidates_raw": direct_fact_candidates_raw,
                "direct_fact_candidates": direct_fact_candidates,
                "projection_trace": projection_trace,
                "baseline_raw_choices": baseline_raw,
                "question_reports": question_reports,
            }
        )

    finalized = {
        bucket: finalize_bucket_summary(summary)
        for bucket, summary in bucket_summary.items()
    }
    return finalized, rows


def preview_test_rows(args) -> tuple[dict, list[dict]]:
    task7 = load_json(Path(TASK_FILES[7]))
    bucket_counts = defaultdict(int)
    author_catalog = collect_author_catalog(task7) if args.projection_mode == "author_catalog" else None
    rows = []
    for sample in task7["test_samples"]:
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        bucket = detect_direct_fact_bucket(category, clue, family)
        if not bucket:
            continue
        bucket_counts[bucket] += 1
        (
            question_reports,
            direct_fact_candidates_raw,
            direct_fact_candidates,
            projection_trace,
        ) = build_direct_fact_candidates(
            category=category,
            clue=clue,
            family=family,
            bucket=bucket,
            author_catalog=author_catalog,
            rounds=args.direct_fact_rounds,
            n=args.direct_fact_n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        rows.append(
            {
                "id": sample["id"],
                "bucket": bucket,
                "family": family,
                "category": category,
                "clue": clue,
                "direct_fact_candidates_raw": direct_fact_candidates_raw,
                "direct_fact_candidates": direct_fact_candidates,
                "projection_trace": projection_trace,
                "question_reports": question_reports,
            }
        )
    return dict(bucket_counts), rows


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Direct-Fact Bucket Audit",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Profile: `{report['profile']}`",
        f"- Retrieval mode: `{report['retrieval_mode']}`",
        f"- Projection mode: `{report['projection_mode']}`",
        (
            f"- Sampling: baseline `n={report['baseline_n_candidates']}`; "
            f"direct-fact `rounds={report['direct_fact_rounds']}` `n={report['direct_fact_n']}`"
        ),
        "",
        "## Example Bucket Summary",
        "",
    ]

    for bucket, summary in sorted(report["example_bucket_summary"].items()):
        lines.extend(
            [
                f"### `{bucket}`",
                "",
                f"- Count: `{summary['count']}`",
                f"- Baseline visible: `{summary['baseline_visible_count']}` / `{summary['count']}`",
                f"- Direct-fact visible: `{summary['direct_fact_visible_count']}` / `{summary['count']}`",
                f"- Union visible: `{summary['union_visible_count']}` / `{summary['count']}`",
                f"- Direct-fact gain rows: `{summary['direct_fact_gain_count']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Test Preview Summary",
            "",
        ]
    )
    for bucket, count in sorted(report["test_bucket_counts"].items()):
        lines.append(f"- `{bucket}`: `{count}`")
    lines.append("")

    lines.extend(
        [
            "## Example Rows",
            "",
        ]
    )
    for row in report["example_rows"]:
        lines.extend(
            [
                f"### `{row['id']}`",
                "",
                f"- Bucket: `{row['bucket']}`",
                f"- Gold: `{row['gold']}`",
                f"- Baseline visible: `{row['baseline_visible']}`",
                f"- Direct-fact visible: `{row['direct_fact_visible']}`",
                f"- Baseline candidates: `{', '.join(row['baseline_candidates'][:12]) if row['baseline_candidates'] else '(empty)'}`",
                f"- Direct-fact raw: `{', '.join(row['direct_fact_candidates_raw'][:12]) if row['direct_fact_candidates_raw'] else '(empty)'}`",
                f"- Direct-fact candidates: `{', '.join(row['direct_fact_candidates'][:12]) if row['direct_fact_candidates'] else '(empty)'}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Test Preview Rows",
            "",
        ]
    )
    for row in report["test_rows"]:
        lines.extend(
            [
                f"### `{row['id']}`",
                "",
                f"- Bucket: `{row['bucket']}`",
                f"- Category: `{row['category']}`",
                f"- Clue: {row['clue']}",
                f"- Direct-fact raw: `{', '.join(row['direct_fact_candidates_raw'][:12]) if row['direct_fact_candidates_raw'] else '(empty)'}`",
                f"- Direct-fact candidates: `{', '.join(row['direct_fact_candidates'][:12]) if row['direct_fact_candidates'] else '(empty)'}`",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    example_bucket_summary, example_rows = evaluate_example_rows(args)
    test_bucket_counts, test_rows = preview_test_rows(args)
    report = {
        "generated_on": "2026-04-01",
        "profile": args.profile,
        "retrieval_mode": args.retrieval_mode,
        "projection_mode": args.projection_mode,
        "baseline_n_candidates": args.baseline_n_candidates,
        "direct_fact_rounds": args.direct_fact_rounds,
        "direct_fact_n": args.direct_fact_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "example_bucket_summary": example_bucket_summary,
        "example_rows": example_rows,
        "test_bucket_counts": test_bucket_counts,
        "test_rows": test_rows,
    }
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
