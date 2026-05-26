import argparse
import json
from collections import Counter
from pathlib import Path

from main import TASK_FILES
from method import (
    _request_nvidia_completions,
    build_prompt,
    count_answer,
    detect_task7_secondary_family,
    get_profile_name,
    normalize_task7_answer,
    parse_task7_fields,
    select_examples,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

FOCUS_BUNDLE_PATH = WORK_LOGS_DIR / "task7_recall_focus_bundle_2026-04-01.json"
SUPPORT_EXAMPLES_PATH = WORK_LOGS_DIR / "task7_recall_support_examples_2026-04-01.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_bundle", type=str, default=str(FOCUS_BUNDLE_PATH))
    parser.add_argument("--support_examples_path", type=str, default=str(SUPPORT_EXAMPLES_PATH))
    parser.add_argument("--output_path", type=str, default=str(WORK_LOGS_DIR / "task7_slot_expansion_probe_2026-04-01.json"))
    parser.add_argument("--rounds_per_slot", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=32)
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


def get_slot_prompts(category: str, clue: str, family: str) -> list[str]:
    if family == "person_entity":
        return [
            "Return the original or earliest historical person the clue points to.",
            "Avoid later, more famous successors. Return the earlier canonical full name.",
            "Return one plausible full person name from the correct era for this clue.",
            "If this is about a TV or game-show history fact, return the earlier host rather than the modern one.",
        ]
    if family == "organization_entity":
        return [
            "Return the publisher or company name, not the author, work, or generic object.",
            "If an ampersand-style or alliterative organization name fits, prefer that kind of canonical company name.",
            "Return one plausible historical publisher or company full name for this clue.",
            "If this is about who published something first, answer with the organization itself.",
        ]
    return [
        "Return one plausible canonical Jeopardy answer candidate.",
        "Return a different plausible canonical Jeopardy answer candidate.",
    ]


def build_slot_prompt(base_prompt: str, slot_instruction: str) -> str:
    prefix = base_prompt.rpartition("Answer:\n<label>")[0].rstrip()
    return (
        f"{prefix}\n\n"
        "Candidate Expansion Slot:\n"
        f"{slot_instruction}\n"
        "Return exactly one candidate answer inside <label>...</label>.\n\n"
        "Answer:\n<label>"
    )


def main() -> None:
    args = parse_args()
    focus_bundle = load_json(Path(args.focus_bundle))
    support_examples = load_json(Path(args.support_examples_path))
    sample_map, all_examples, task_description = load_task7_assets()
    support_map = {
        row["target_id"]: [item["id"] for item in row["top_support_examples"][:8]]
        for row in support_examples["results"]
    }

    rows = []
    for target in focus_bundle["retained_targets"]:
        sample = sample_map[target["id"]]
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        support_ids = set(support_map.get(target["id"], []))
        support_examples_rows = [example for example in all_examples if example["id"] in support_ids]
        prompt = build_prompt(task_description, sample["input"], task_id=7, profile_name=get_profile_name("baseline"))
        examples_str = select_examples(
            support_examples_rows,
            task_description,
            sample["input"],
            task_id=7,
            profile_name="baseline",
        )
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")

        slot_candidates = []
        seen = set()
        slot_reports = []
        for slot_instruction in get_slot_prompts(category, clue, family):
            slot_prompt = build_slot_prompt(input_prompt, slot_instruction)
            raw_choices = []
            for _ in range(max(1, args.rounds_per_slot)):
                raw_choices.extend(
                    _request_nvidia_completions(
                        slot_prompt,
                        task_id=7,
                        n=1,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                    )
                )
            parsed = []
            for raw in raw_choices:
                candidate = count_answer(raw, task_id=7)
                norm = normalize_task7_answer(candidate)
                if not norm:
                    continue
                parsed.append(candidate)
                if norm not in seen:
                    seen.add(norm)
                    slot_candidates.append(candidate)
            slot_reports.append(
                {
                    "slot_instruction": slot_instruction,
                    "raw_choice_count": len(raw_choices),
                    "parsed_candidates": parsed,
                }
            )

        gold_norm = normalize_task7_answer(target["gold"])
        rows.append(
            {
                "id": target["id"],
                "gold": target["gold"],
                "family": family,
                "gold_visible_in_candidates": gold_norm in {normalize_task7_answer(candidate) for candidate in slot_candidates},
                "gold_rank": next(
                    (
                        idx + 1
                        for idx, candidate in enumerate(slot_candidates)
                        if normalize_task7_answer(candidate) == gold_norm
                    ),
                    None,
                ),
                "slot_candidates": slot_candidates,
                "slot_reports": slot_reports,
            }
        )

    report = {
        "generated_on": "2026-04-01",
        "focus_bundle": str(Path(args.focus_bundle)),
        "support_examples_path": str(Path(args.support_examples_path)),
        "rounds_per_slot": args.rounds_per_slot,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "gold_visible_count": sum(int(row["gold_visible_in_candidates"]) for row in rows),
        "row_count": len(rows),
        "rows": rows,
    }
    Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
