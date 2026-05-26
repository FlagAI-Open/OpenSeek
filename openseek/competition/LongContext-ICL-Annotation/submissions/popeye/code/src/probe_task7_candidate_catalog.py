import argparse
import json
import re
from collections import Counter
from pathlib import Path

from main import TASK_FILES
from method import (
    _request_nvidia_completions,
    build_prompt,
    build_task7_generation_guidance,
    build_task7_retrieval_context,
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
    parser.add_argument("--output_path", type=str, default=str(WORK_LOGS_DIR / "task7_candidate_catalog_probe_2026-04-01.json"))
    parser.add_argument("--variants", type=str, default="baseline_catalog,retrieval_catalog,support_catalog")
    parser.add_argument("--catalog_size", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=192)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task7_assets() -> tuple[dict[str, dict], list[dict], str]:
    task7 = load_json(Path(TASK_FILES[7]))
    sample_map = {}
    for section in ("test_samples", "examples"):
        for sample in task7[section]:
            sample_map[sample["id"]] = sample
    return sample_map, list(task7["examples"]), task7["Definition"][0]


def build_variant_specs(raw_variants: str) -> list[dict]:
    specs = {
        "baseline_catalog": {
            "profile": "baseline",
            "retrieval_mode": "none",
            "example_source": "default",
            "hint_mode": "recall",
        },
        "retrieval_catalog": {
            "profile": "long_context_task7_retrieval",
            "retrieval_mode": "lexical",
            "example_source": "default",
            "hint_mode": "recall",
        },
        "support_catalog": {
            "profile": "baseline",
            "retrieval_mode": "none",
            "example_source": "support",
            "hint_mode": "recall",
        },
    }
    names = [token.strip() for token in raw_variants.split(",") if token.strip()]
    return [{"name": name, **specs[name]} for name in names]


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


def build_catalog_prompt(input_prompt: str, text2annotate: str, hint_mode: str, catalog_size: int) -> str:
    category, clue = parse_task7_fields(text2annotate)
    guidance = build_task7_generation_guidance(category, clue, hint_mode=hint_mode)
    if "Answer:\n<label>" in input_prompt:
        prefix = input_prompt.rpartition("Answer:\n<label>")[0].rstrip()
    else:
        prefix = input_prompt.rstrip()
    family = detect_task7_secondary_family(category, clue, typed_route="auto")
    extra = ""
    if family == "person_entity":
        extra = (
            "Prefer distinct historical people. Avoid repeating the same famous modern default. "
            "Include earlier or first-version figures if the clue suggests them.\n"
        )
    elif family == "organization_entity":
        extra = (
            "Prefer distinct organization or publisher names. Include canonical company names, especially alliterative or ampersand-style names when plausible.\n"
        )
    return (
        f"{prefix}\n\n"
        f"{guidance}"
        "Candidate Catalog Task:\n"
        f"Produce exactly {catalog_size} diverse plausible Jeopardy answer candidates.\n"
        "Rules:\n"
        "1. Return exactly one <catalog>...</catalog> block.\n"
        "2. Inside the block, put one candidate per line using <cand>candidate text</cand>.\n"
        "3. Candidates must be distinct.\n"
        "4. Prefer canonical entities rather than clue fragments.\n"
        "5. Do not output explanations, notes, headings, or any text outside the tags.\n"
        f"{extra}"
        "\nRequired format example:\n"
        "<catalog>\n"
        "<cand>first candidate</cand>\n"
        "<cand>second candidate</cand>\n"
        "</catalog>\n"
    )


def parse_catalog_lines(text: str) -> list[str]:
    candidates = []
    seen = set()
    cand_matches = re.findall(r"<cand>(.*?)</cand>", text, flags=re.I | re.S)
    lines = cand_matches if cand_matches else text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"</?catalog>", "", line, flags=re.I)
        line = re.sub(r"</?cand>", "", line, flags=re.I)
        if re.match(r"^(candidates?|answer|notes?)\s*[:：]?\s*$", line, flags=re.I):
            continue
        line = re.sub(r"^\d+[\).\-\s]+", "", line)
        line = re.sub(r"^[\-\*\u2022]+\s*", "", line)
        line = line.strip(" \"'`")
        if not line:
            continue
        if any(token in line.lower() for token in ("return exactly", "candidate catalog task", "required format")):
            continue
        normalized = normalize_task7_answer(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(line)
    return candidates


def collect_catalog_candidates(
    *,
    catalog_prompt: str,
    rounds: int,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[list[str], list[str]]:
    raw_texts = []
    parsed_candidates = []
    seen = set()
    for _ in range(max(1, rounds)):
        raw_texts.extend(
            _request_nvidia_completions(
                catalog_prompt,
                task_id=7,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )
    for raw in raw_texts:
        for candidate in parse_catalog_lines(raw):
            normalized = normalize_task7_answer(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            parsed_candidates.append(candidate)
    return raw_texts, parsed_candidates


def main() -> None:
    args = parse_args()
    focus_bundle = load_json(Path(args.focus_bundle))
    support_examples = load_json(Path(args.support_examples_path))
    sample_map, all_examples, task_description = load_task7_assets()
    variants = build_variant_specs(args.variants)
    support_map = {
        row["target_id"]: [item["id"] for item in row["top_support_examples"]]
        for row in support_examples["results"]
    }

    results = []
    summary = []

    for variant in variants:
        visible_count = 0
        row_reports = []
        for target in focus_bundle["retained_targets"]:
            sample = sample_map[target["id"]]
            profile_name = get_profile_name(variant["profile"])
            examples_limit = get_example_pool_limit(task_id=7, examples_limit=None, profile_name=profile_name)
            if variant["example_source"] == "support":
                support_ids = set(support_map.get(target["id"], []))
                icl_examples = [example for example in all_examples if example["id"] in support_ids][:examples_limit]
            else:
                icl_examples = [example for example in all_examples if example["id"] != target["id"]][:examples_limit]

            input_prompt, ordered_examples = build_task7_input_prompt(
                examples=icl_examples,
                task_description=task_description,
                text2annotate=sample["input"],
                profile_name=profile_name,
                retrieval_mode=variant["retrieval_mode"],
            )
            catalog_prompt = build_catalog_prompt(
                input_prompt=input_prompt,
                text2annotate=sample["input"],
                hint_mode=variant["hint_mode"],
                catalog_size=args.catalog_size,
            )
            raw_texts, catalog_candidates = collect_catalog_candidates(
                catalog_prompt=catalog_prompt,
                rounds=args.rounds,
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            gold_norm = normalize_task7_answer(target["gold"])
            visible = gold_norm in {normalize_task7_answer(candidate) for candidate in catalog_candidates}
            visible_count += int(visible)
            counts = Counter()
            for raw in raw_texts:
                for candidate in parse_catalog_lines(raw):
                    counts[normalize_task7_answer(candidate)] += 1
            row_reports.append(
                {
                    "id": target["id"],
                    "gold": target["gold"],
                    "gold_visible_in_candidates": visible,
                    "gold_rank": next(
                        (
                            idx + 1
                            for idx, candidate in enumerate(catalog_candidates)
                            if normalize_task7_answer(candidate) == gold_norm
                        ),
                        None,
                    ),
                    "catalog_candidates": catalog_candidates[:20],
                    "candidate_vote_counts": {
                        candidate: counts.get(normalize_task7_answer(candidate), 0)
                        for candidate in catalog_candidates[:20]
                    },
                    "example_source": variant["example_source"],
                    "first_example_ids": [example["id"] for example in ordered_examples[:5]],
                }
            )

        summary.append(
            {
                "variant": variant["name"],
                "gold_visible_count": visible_count,
                "row_count": len(focus_bundle["retained_targets"]),
                "gold_visible_rate": round(visible_count / max(len(focus_bundle["retained_targets"]), 1), 4),
                "profile": variant["profile"],
                "retrieval_mode": variant["retrieval_mode"],
                "example_source": variant["example_source"],
                "hint_mode": variant["hint_mode"],
            }
        )
        results.append(
            {
                "variant": variant["name"],
                "rows": row_reports,
            }
        )

    report = {
        "generated_on": "2026-04-01",
        "focus_bundle": str(Path(args.focus_bundle)),
        "support_examples_path": str(Path(args.support_examples_path)),
        "catalog_size": args.catalog_size,
        "rounds": args.rounds,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "summary": summary,
        "variants": results,
    }
    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
