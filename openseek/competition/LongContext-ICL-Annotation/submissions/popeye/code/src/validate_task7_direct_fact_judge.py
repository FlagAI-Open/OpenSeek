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
from task7_direct_fact_projection import (
    build_author_answer_catalog,
    project_direct_fact_candidates_to_author_catalog,
)
from validate_task7_candidate_rerank import (
    build_task7_prompt,
    build_task7_append_unique_judge_candidates,
    collect_branch_choices,
    dedupe_candidates,
    normalize_qa_answer,
    run_judge,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_direct_fact_judge_validation_2026-04-01.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_direct_fact_judge_validation_2026-04-01.md"


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
    parser.add_argument("--judge_mode", type=str, default="completion")
    parser.add_argument("--max_judge_candidates", type=int, default=6)
    parser.add_argument("--reserved_secondary_slots", type=str, default="2,3")
    parser.add_argument("--projection_mode", type=str, choices=["none", "author_catalog"], default="none")
    return parser.parse_args()


def parse_reserved_slots(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            values.append(int(token))
    return values or [2]


def init_summary_bucket() -> dict:
    return {
        "count": 0,
        "baseline_judge_correct": 0,
        "baseline_judge_visible": 0,
        "augmented_judge_correct": 0,
        "augmented_judge_visible": 0,
        "judge_gain_count": 0,
        "visibility_gain_count": 0,
    }


def finalize_summary_bucket(bucket: dict) -> dict:
    count = bucket["count"]
    for key in list(bucket):
        if key == "count" or not key.endswith(("_correct", "_visible", "_count")):
            continue
        bucket[key + "_rate"] = bucket[key] / count if count else 0.0
    return bucket


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
) -> list[str]:
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
    return candidates


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
    return reports, raw_candidates, projected_candidates, projection_trace


def collect_author_catalog(task7: dict) -> dict:
    author_answers = []
    for sample in task7["examples"]:
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        if detect_direct_fact_bucket(category, clue, family) == "quoted_work_author_relation":
            author_answers.append(sample["output"][0])
    return build_author_answer_catalog(author_answers)


def judge_visible(gold: str, candidates: list[str]) -> bool:
    gold_norm = normalize_qa_answer(gold)
    return gold_norm in {normalize_qa_answer(candidate) for candidate in candidates}


def judge_correct(gold: str, prediction: str | None) -> bool:
    return normalize_qa_answer(gold) == normalize_qa_answer(prediction)


def init_variant_bucket() -> dict:
    return {
        "count": 0,
        "baseline_judge_correct": 0,
        "baseline_judge_visible": 0,
        "variant_judge_correct": 0,
        "variant_judge_visible": 0,
        "judge_gain_count": 0,
        "visibility_gain_count": 0,
    }


def build_variant_specs(
    reserved_values: list[int],
    *,
    use_projected_variants: bool,
) -> list[dict]:
    specs = []
    for reserve in reserved_values:
        specs.append(
            {
                "name": f"append_unique_raw_r{reserve}",
                "merge_mode": "append_unique",
                "reserve": reserve,
                "candidate_source": "raw",
            }
        )
        if use_projected_variants:
            specs.append(
                {
                    "name": f"append_unique_projected_r{reserve}",
                    "merge_mode": "append_unique",
                    "reserve": reserve,
                    "candidate_source": "projected",
                }
            )
    if use_projected_variants:
        specs.append(
            {
                "name": "projected_top_only",
                "merge_mode": "top_only",
                "reserve": 0,
                "candidate_source": "projected",
            }
        )
    return specs


def evaluate(args) -> dict:
    task7 = load_json(Path(TASK_FILES[7]))
    examples = list(task7["examples"])
    task_description = task7["Definition"][0]
    reserved_values = parse_reserved_slots(args.reserved_secondary_slots)
    author_catalog = collect_author_catalog(task7) if args.projection_mode == "author_catalog" else None
    variant_specs = build_variant_specs(
        reserved_values,
        use_projected_variants=author_catalog is not None,
    )

    summary = {
        "baseline": defaultdict(init_summary_bucket),
        "variants": {
            spec["name"]: defaultdict(init_variant_bucket)
            for spec in variant_specs
        },
    }
    rows = []

    for sample in examples:
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        bucket = detect_direct_fact_bucket(category, clue, family)
        if not bucket:
            continue

        gold = sample["output"][0]
        baseline_candidates = build_baseline_candidates(
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
            direct_fact_candidates_projected,
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

        baseline_judge_candidates = baseline_candidates[: args.max_judge_candidates]
        baseline_prediction, baseline_judge_raw = run_judge(
            category=category,
            clue=clue,
            candidates=baseline_judge_candidates,
            judge_mode=args.judge_mode,
        )
        baseline_visible = judge_visible(gold, baseline_judge_candidates)
        baseline_ok = judge_correct(gold, baseline_prediction)
        base_bucket = summary["baseline"][bucket]
        base_bucket["count"] += 1
        base_bucket["baseline_judge_correct"] += int(baseline_ok)
        base_bucket["baseline_judge_visible"] += int(baseline_visible)

        variant_rows = {}
        for spec in variant_specs:
            secondary_candidates = (
                direct_fact_candidates_projected
                if spec["candidate_source"] == "projected"
                else direct_fact_candidates_raw
            )
            if spec["merge_mode"] == "top_only":
                judge_candidates = secondary_candidates[: args.max_judge_candidates]
            else:
                judge_candidates = build_task7_append_unique_judge_candidates(
                    baseline_candidates,
                    secondary_candidates,
                    max_candidates=args.max_judge_candidates,
                    reserved_secondary_slots=spec["reserve"],
                )
            prediction, judge_raw = run_judge(
                category=category,
                clue=clue,
                candidates=judge_candidates,
                judge_mode=args.judge_mode,
            )
            visible = judge_visible(gold, judge_candidates)
            ok = judge_correct(gold, prediction)

            bucket_summary = summary["variants"][spec["name"]][bucket]
            bucket_summary["count"] += 1
            bucket_summary["baseline_judge_correct"] += int(baseline_ok)
            bucket_summary["baseline_judge_visible"] += int(baseline_visible)
            bucket_summary["variant_judge_correct"] += int(ok)
            bucket_summary["variant_judge_visible"] += int(visible)
            bucket_summary["judge_gain_count"] += int(ok and not baseline_ok)
            bucket_summary["visibility_gain_count"] += int(visible and not baseline_visible)

            variant_rows[spec["name"]] = {
                "candidate_source": spec["candidate_source"],
                "merge_mode": spec["merge_mode"],
                "reserve": spec["reserve"],
                "judge_candidates": judge_candidates,
                "prediction": prediction,
                "judge_raw": judge_raw,
                "visible": visible,
                "correct": ok,
            }

        rows.append(
            {
                "id": sample["id"],
                "bucket": bucket,
                "family": family,
                "gold": gold,
                "category": category,
                "clue": clue,
                "baseline": {
                    "judge_candidates": baseline_judge_candidates,
                    "prediction": baseline_prediction,
                    "judge_raw": baseline_judge_raw,
                    "visible": baseline_visible,
                    "correct": baseline_ok,
                },
                "direct_fact_candidates_raw": direct_fact_candidates_raw,
                "direct_fact_candidates_projected": direct_fact_candidates_projected,
                "projection_trace": projection_trace,
                "question_reports": question_reports,
                "variants": variant_rows,
            }
        )

    finalized_baseline = {
        bucket: finalize_summary_bucket(bucket_summary)
        for bucket, bucket_summary in summary["baseline"].items()
    }
    finalized_variants = {
        variant_name: {
            bucket: finalize_summary_bucket(bucket_summary)
            for bucket, bucket_summary in variant_summary.items()
        }
        for variant_name, variant_summary in summary["variants"].items()
    }
    return {
        "generated_on": "2026-04-01",
        "profile": args.profile,
        "retrieval_mode": args.retrieval_mode,
        "projection_mode": args.projection_mode,
        "baseline_n_candidates": args.baseline_n_candidates,
        "direct_fact_rounds": args.direct_fact_rounds,
        "direct_fact_n": args.direct_fact_n,
        "judge_mode": args.judge_mode,
        "max_judge_candidates": args.max_judge_candidates,
        "reserved_secondary_slots": reserved_values,
        "variant_specs": variant_specs,
        "baseline_summary": finalized_baseline,
        "variant_summary": finalized_variants,
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Direct-Fact Judge Validation",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Profile: `{report['profile']}`",
        f"- Retrieval mode: `{report['retrieval_mode']}`",
        f"- Projection mode: `{report['projection_mode']}`",
        f"- Judge mode: `{report['judge_mode']}`",
        f"- Baseline candidate budget: `{report['baseline_n_candidates']}`",
        f"- Direct-fact sampling: `rounds={report['direct_fact_rounds']}` `n={report['direct_fact_n']}`",
        "",
        "## Baseline Summary",
        "",
    ]
    for bucket, summary in sorted(report["baseline_summary"].items()):
        lines.extend(
            [
                f"### `{bucket}`",
                "",
                f"- Count: `{summary['count']}`",
                f"- Baseline judge visible: `{summary['baseline_judge_visible']}` / `{summary['count']}`",
                f"- Baseline judge correct: `{summary['baseline_judge_correct']}` / `{summary['count']}`",
                "",
            ]
        )

    for variant_name, variant_summary in sorted(report["variant_summary"].items()):
        lines.extend(
            [
                f"## Variant `{variant_name}`",
                "",
            ]
        )
        for bucket, summary in sorted(variant_summary.items()):
            lines.extend(
                [
                    f"### `{bucket}`",
                    "",
                    f"- Count: `{summary['count']}`",
                    f"- Variant judge visible: `{summary['variant_judge_visible']}` / `{summary['count']}`",
                    f"- Variant judge correct: `{summary['variant_judge_correct']}` / `{summary['count']}`",
                    f"- Visibility gain rows: `{summary['visibility_gain_count']}`",
                    f"- Judge gain rows: `{summary['judge_gain_count']}`",
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
