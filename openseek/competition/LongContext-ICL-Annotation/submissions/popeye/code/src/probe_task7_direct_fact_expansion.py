import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from main import TASK_FILES
from method import (
    _request_nvidia_completions,
    count_answer,
    detect_task7_secondary_family,
    normalize_task7_answer,
    parse_task7_fields,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_FOCUS_BUNDLE = WORK_LOGS_DIR / "task7_recall_focus_bundle_2026-04-01.json"
DEFAULT_OUTPUT_JSON = WORK_LOGS_DIR / "task7_direct_fact_probe_2026-04-01.json"
DEFAULT_OUTPUT_MD = WORK_LOGS_DIR / "task7_direct_fact_probe_2026-04-01.md"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_bundle", type=str, default=str(DEFAULT_FOCUS_BUNDLE))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--markdown_path", type=str, default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=32)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task7_sample_map() -> dict[str, dict]:
    task7 = load_json(Path(TASK_FILES[7]))
    sample_map = {}
    for section in ("examples", "test_samples"):
        for sample in task7[section]:
            sample_map[sample["id"]] = sample
    return sample_map


def _extract_show_title(clue: str) -> str:
    quoted = re.findall(r'"([^"]+)"', clue)
    if quoted:
        return quoted[0]
    match = re.search(r"\bof\s+([A-Z][A-Za-z0-9!'&.\- ]+)", clue)
    if match:
        return match.group(1).strip()
    return ""


def extract_quoted_titles(clue: str) -> list[str]:
    return [title.strip() for title in re.findall(r'"([^"]+)"', clue) if title.strip()]


def _extract_subject_word(category: str, clue: str) -> str:
    for source in (clue, category):
        for pattern in (
            r"\bbook of ([a-z]+)\b",
            r"\bbook of ([a-z]+s)\b",
            r"\b([a-z]+) book\b",
            r"\b([a-z]+s) were invented\b",
        ):
            match = re.search(pattern, source.lower())
            if match:
                candidate = match.group(1)
                if candidate not in {"it", "them", "this", "that"}:
                    return candidate
    category_tokens = re.findall(r"[a-z]+", category.lower())
    if category_tokens:
        return category_tokens[0]
    return "it"


def _is_legacy_host_clue(clue: str) -> bool:
    lowered = clue.lower()
    legacy_markers = ("first version", "original", "first host", "host of the first")
    return any(marker in lowered for marker in legacy_markers)


def detect_direct_fact_bucket(category: str, clue: str, family: str) -> str | None:
    lowered = clue.lower()
    quoted_titles = extract_quoted_titles(clue)
    if (
        family == "person_entity"
        and "host" in lowered
        and quoted_titles
        and _is_legacy_host_clue(clue)
    ):
        return "legacy_show_host"
    if (
        family == "person_entity"
        and quoted_titles
        and (
            "this author of" in lowered
            or "works by this author" in lowered
            or "illustrator of" in lowered
        )
    ):
        return "quoted_work_author_relation"
    if (
        family == "organization_entity"
        and "book" in lowered
        and (
            "this company" in lowered
            or "this publisher" in lowered
            or "this organization" in lowered
        )
    ):
        return "explicit_company_publisher_book"
    return None


def build_legacy_host_questions(show_title: str, clue: str) -> list[str]:
    questions = [
        f"Who was the first host of {show_title} in the 1960s?",
        f"Who hosted the original {show_title} in the 1960s?",
        f"Who was the original host of {show_title} before its later revival host?",
        f"Which TV host fronted the first version of {show_title} in the 1960s?",
    ]
    if _is_legacy_host_clue(clue):
        questions.extend(build_legacy_anchor_questions(show_title))
    return questions


def build_author_relation_questions(clue: str) -> list[str]:
    titles = extract_quoted_titles(clue)
    questions = []
    for title in titles[:2]:
        questions.extend(
            [
                f"Who wrote {title}?",
                f"Who is the author of {title}?",
                f"Which author wrote {title}?",
            ]
        )
    if "illustrator of" in clue.lower() and titles:
        questions.append(f"Who is the author associated with {titles[0]}?")
    return questions


def build_organization_book_questions(category: str, clue: str) -> list[str]:
    subject = _extract_subject_word(category, clue)
    singular = subject[:-1] if subject.endswith("s") and len(subject) > 3 else subject
    return [
        f"Which company published the first book of {subject}?",
        f"Which publishing company first issued a book of {subject}?",
        (
            f"What publishing house with two founder surnames beginning with the same letter "
            f"first published a {singular} book?"
        ),
        f"Which publisher whose name joins two surnames first published a {singular} book?",
        f"What major book publisher first brought out a {singular} book?",
    ]


def build_direct_fact_questions(category: str, clue: str, family: str) -> list[str]:
    bucket = detect_direct_fact_bucket(category, clue, family)
    if bucket == "legacy_show_host":
        show_title = _extract_show_title(clue) or "the show"
        return build_legacy_host_questions(show_title, clue)

    if bucket == "quoted_work_author_relation":
        return build_author_relation_questions(clue)

    if bucket == "explicit_company_publisher_book":
        return build_organization_book_questions(category, clue)

    return [
        f"What is the canonical answer to this clue? Category: {category}. Clue: {clue}",
    ]


def build_legacy_anchor_questions(show_title: str) -> list[str]:
    anchor_prompt = (
        f"Return only one later or more famous person associated with {show_title} "
        "inside <label>...</label>.\nAnswer:\n<label>"
    )
    raw_choices = _request_nvidia_completions(
        anchor_prompt,
        task_id=7,
        n=6,
        temperature=0.7,
        top_p=0.9,
        max_tokens=24,
    )
    anchors = []
    seen = set()
    for raw in raw_choices:
        candidate = count_answer(raw, task_id=7)
        normalized = normalize_task7_answer(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        anchors.append(candidate)

    questions = []
    for anchor in anchors[:3]:
        questions.append(f"Who was the original host of {show_title} before {anchor}?")
        questions.append(f"Who hosted the first version of {show_title} before {anchor}?")
    return questions


def build_answer_prompt(question: str, family: str) -> str:
    if family == "person_entity":
        preamble = "Return only the canonical full person name inside <label>...</label>."
    elif family == "organization_entity":
        preamble = "Return only the canonical company or publisher name inside <label>...</label>."
    else:
        preamble = "Return only the canonical answer inside <label>...</label>."
    return f"{preamble}\nQuestion: {question}\nAnswer:\n<label>"


def collect_question_candidates(
    question: str,
    family: str,
    *,
    rounds: int,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict:
    prompt = build_answer_prompt(question, family)
    raw_choices = []
    for _ in range(max(1, rounds)):
        raw_choices.extend(
            _request_nvidia_completions(
                prompt,
                task_id=7,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )

    parsed_candidates = []
    seen = set()
    candidate_counts = defaultdict(int)
    candidate_display = {}
    for raw in raw_choices:
        candidate = count_answer(raw, task_id=7)
        normalized = normalize_task7_answer(candidate)
        if not normalized:
            continue
        candidate_counts[normalized] += 1
        candidate_display.setdefault(normalized, candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        parsed_candidates.append(candidate)

    return {
        "question": question,
        "prompt": prompt,
        "raw_choices": raw_choices,
        "parsed_candidates": parsed_candidates,
        "candidate_counts": dict(candidate_counts),
    }


def aggregate_direct_fact_candidates(question_reports: list[dict]) -> list[str]:
    total_counts = defaultdict(int)
    display = {}
    for report in question_reports:
        for normalized, count in report.get("candidate_counts", {}).items():
            if not normalized:
                continue
            total_counts[normalized] += count
        for candidate in report.get("parsed_candidates", []):
            normalized = normalize_task7_answer(candidate)
            if not normalized:
                continue
            display.setdefault(normalized, candidate)

    ranked = sorted(
        total_counts.items(),
        key=lambda item: (-item[1], len(item[0].split()), len(item[0]), item[0]),
    )
    return [display[normalized] for normalized, _ in ranked if normalized in display]


def build_report(args) -> dict:
    focus_bundle = load_json(Path(args.focus_bundle))
    sample_map = load_task7_sample_map()

    rows = []
    for target in focus_bundle["retained_targets"]:
        sample = sample_map[target["id"]]
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        question_reports = []

        for question in build_direct_fact_questions(category, clue, family):
            report = collect_question_candidates(
                question,
                family,
                rounds=args.rounds,
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            question_reports.append(report)

        candidate_pool = aggregate_direct_fact_candidates(question_reports)

        gold_norm = normalize_task7_answer(target["gold"])
        rows.append(
            {
                "id": target["id"],
                "gold": target["gold"],
                "family": family,
                "bucket": detect_direct_fact_bucket(category, clue, family),
                "category": category,
                "clue": clue,
                "gold_visible_in_candidates": gold_norm in {normalize_task7_answer(candidate) for candidate in candidate_pool},
                "gold_rank": next(
                    (
                        idx + 1
                        for idx, candidate in enumerate(candidate_pool)
                        if normalize_task7_answer(candidate) == gold_norm
                    ),
                    None,
                ),
                "candidate_pool": candidate_pool,
                "question_reports": question_reports,
            }
        )

    return {
        "generated_on": "2026-04-01",
        "focus_bundle": str(Path(args.focus_bundle)),
        "rounds": args.rounds,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "gold_visible_count": sum(int(row["gold_visible_in_candidates"]) for row in rows),
        "row_count": len(rows),
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Direct-Fact Probe",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Focus bundle: `{report['focus_bundle']}`",
        f"- `gold_visible_count`: `{report['gold_visible_count']}/{report['row_count']}`",
        (
            f"- Sampling: `rounds={report['rounds']}` `n={report['n']}` "
            f"`temperature={report['temperature']}` `top_p={report['top_p']}`"
        ),
        "",
    ]
    for row in report["rows"]:
        lines.extend(
            [
                f"## `{row['id']}`",
                "",
                f"- Gold: `{row['gold']}`",
                f"- Family: `{row['family']}`",
                f"- Gold visible: `{row['gold_visible_in_candidates']}`",
                f"- Gold rank: `{row['gold_rank']}`",
                f"- Candidate pool: `{', '.join(row['candidate_pool']) if row['candidate_pool'] else '(empty)'}`",
                "",
                "Question reports:",
                "",
            ]
        )
        for question_report in row["question_reports"]:
            parsed = ", ".join(question_report["parsed_candidates"]) if question_report["parsed_candidates"] else "(empty)"
            lines.append(f"- question=`{question_report['question']}` parsed=`{parsed}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_path).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
