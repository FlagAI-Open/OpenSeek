import json
import re
from pathlib import Path

from main import TASK_FILES
from method import detect_task7_secondary_family, parse_task7_fields


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

FOCUS_BUNDLE_PATH = WORK_LOGS_DIR / "task7_recall_focus_bundle_2026-04-01.json"
OUTPUT_JSON_PATH = WORK_LOGS_DIR / "task7_recall_support_examples_2026-04-01.json"
OUTPUT_MD_PATH = WORK_LOGS_DIR / "task7_recall_support_examples_2026-04-01.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.I) for pattern in patterns)


def looks_like_person_name(answer: str) -> bool:
    parts = re.findall(r"[a-z]+", answer.lower())
    return 2 <= len(parts) <= 4 and all(len(part) > 1 for part in parts)


def looks_like_organization_name(answer: str) -> bool:
    lowered = answer.lower()
    org_markers = (
        "press",
        "publishing",
        "publishers",
        "company",
        "co.",
        "inc.",
        "university",
        "college",
        "network",
        "times",
        "post",
        "journal",
    )
    return "&" in answer or any(marker in lowered for marker in org_markers)


def score_person_support(category: str, clue: str, answer: str) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    combined = f"{category}\n{clue}"
    if detect_task7_secondary_family(category, clue, typed_route="auto") == "person_entity":
        score += 4
        reasons.append("family=person_entity")
    if has_any_pattern(clue, (r"\bhost\b", r"\bhosted\b", r"\bversion\b", r"\boriginal\b", r"\bfirst\b")):
        score += 3
        reasons.append("historical_or_host_pattern")
    if has_any_pattern(combined, (r"\bjeopardy\b", r"\bgame show\b", r"\btv\b", r"\btelevision\b")):
        score += 2
        reasons.append("tv_gameshow_pattern")
    if has_any_pattern(combined, (r"\b'60s\b", r"\b1960s\b", r"\b'50s\b", r"\b1970s\b")):
        score += 1
        reasons.append("dated_era_pattern")
    if looks_like_person_name(answer):
        score += 2
        reasons.append("answer_looks_like_person_name")
    return score, reasons


def score_org_support(category: str, clue: str, answer: str) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    combined = f"{category}\n{clue}"
    if detect_task7_secondary_family(category, clue, typed_route="auto") == "organization_entity":
        score += 4
        reasons.append("family=organization_entity")
    if has_any_pattern(combined, (r"\bcompany\b", r"\bpublisher\b", r"\bpublished\b", r"\bbook\b", r"\bpress\b")):
        score += 3
        reasons.append("publisher_company_pattern")
    if has_any_pattern(combined, (r"\balliterative\b", r"\bname\b", r"\bcrossword\b")):
        score += 2
        reasons.append("naming_pattern")
    if looks_like_organization_name(answer):
        score += 2
        reasons.append("answer_looks_like_organization_name")
    if "&" in answer:
        score += 1
        reasons.append("ampersand_name")
    return score, reasons


def build_report() -> dict:
    focus_bundle = load_json(FOCUS_BUNDLE_PATH)
    task7 = load_json(Path(TASK_FILES[7]))
    all_examples = task7["examples"]

    results = []
    for target in focus_bundle["retained_targets"]:
        target_id = target["id"]
        sample = next(example for example in all_examples if example["id"] == target_id)
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")

        scored_examples = []
        for example in all_examples:
            if example["id"] == target_id:
                continue
            ex_category, ex_clue = parse_task7_fields(example["input"])
            answer = example["output"][0]
            if family == "person_entity":
                score, reasons = score_person_support(ex_category, ex_clue, answer)
            elif family == "organization_entity":
                score, reasons = score_org_support(ex_category, ex_clue, answer)
            else:
                continue
            if score <= 0:
                continue
            scored_examples.append(
                {
                    "id": example["id"],
                    "score": score,
                    "reasons": reasons,
                    "category": ex_category,
                    "clue": ex_clue,
                    "answer": answer,
                }
            )

        scored_examples.sort(
            key=lambda row: (
                -row["score"],
                len(row["answer"].split()),
                len(row["answer"]),
                row["id"],
            )
        )
        results.append(
            {
                "target_id": target_id,
                "gold": target["gold"],
                "family": family,
                "category": category,
                "clue": clue,
                "top_support_examples": scored_examples[:15],
            }
        )

    return {
        "generated_on": "2026-04-01",
        "focus_bundle_path": str(FOCUS_BUNDLE_PATH),
        "results": results,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Task7 Recall Support Examples",
        "",
        f"- Generated on: `{report['generated_on']}`",
        f"- Focus bundle: `{report['focus_bundle_path']}`",
        "",
    ]
    for result in report["results"]:
        lines.extend(
            [
                f"## `{result['target_id']}`",
                "",
                f"- Gold: `{result['gold']}`",
                f"- Family: `{result['family']}`",
                f"- Category: `{result['category']}`",
                f"- Clue: {result['clue']}",
                "",
                "Top support examples:",
                "",
            ]
        )
        for row in result["top_support_examples"][:10]:
            lines.append(
                f"- score=`{row['score']}` id=`{row['id']}` answer=`{row['answer']}` reasons=`{','.join(row['reasons'])}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    OUTPUT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
