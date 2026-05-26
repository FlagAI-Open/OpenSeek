import argparse
import json
import math
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
API_URL = "http://0.0.0.0:2026/v1/completions"
MODEL_NAME = "../Qwen3-4B"

TASK_FILES = {
    2: DATA_DIR / "openseek-2_count_nouns_verbs.json",
    6: DATA_DIR / "openseek-6_mnli_same_genre_classification.json",
}


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads(TASK_FILES[task_id].read_text(encoding="utf-8"))


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def label_of(ex: dict[str, Any]) -> str:
    out = ex["output"]
    return str(out[0] if isinstance(out, list) else out)


def call_completion(prompt: str, max_tokens: int = 16) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stop": ["\n"],
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(API_URL, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as exc:  # noqa: BLE001 - long batch should not die on one request.
            last_error = exc
            time.sleep(2 + attempt)
    return f"ERROR: {last_error}"


def retrieve_examples(
    examples: list[dict[str, Any]],
    query: str,
    limit: int,
    max_chars: int,
    exclude_id: str | None = None,
    same_type_first: bool = False,
) -> str:
    q_words = words(query)
    q_lower = query.lower()
    q_type = "nouns" if "number of nouns" in q_lower else "verbs" if "number of verbs" in q_lower else ""
    q_genre_match = re.search(r"Genre:\s*([a-z0-9/\\-]+)", query)
    q_genre = q_genre_match.group(1).lower() if q_genre_match else ""

    scored = []
    for idx, ex in enumerate(examples):
        if exclude_id and ex["id"] == exclude_id:
            continue
        ex_words = words(ex["input"])
        score = len(q_words & ex_words) / math.sqrt(max(len(ex_words), 1))
        ex_lower = ex["input"].lower()
        if same_type_first and q_type and q_type in ex_lower:
            score += 2.0
        if same_type_first and q_genre and f"genre: {q_genre}" in ex_lower:
            score += 2.0
        scored.append((score, -idx, ex))

    chunks = []
    total = 0
    for _, __, ex in sorted(scored, reverse=True):
        label = label_of(ex)
        if "Sentence 1:" in ex["input"]:
            chunk = f"Input:\n{ex['input']}\nAnswer: {label}\n\n"
        else:
            chunk = f"Input:\n{ex['input']}\nAnswer: {label}\n\n"
        if chunks and total + len(chunk) > max_chars:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if len(chunks) >= limit:
            break
    return "".join(chunks)


def normalize_task2(raw: str) -> str:
    match = re.search(r"-?\d+", raw)
    if not match:
        return ""
    value = int(match.group(0))
    return str(max(0, min(6, value)))


def normalize_task6(raw: str) -> str:
    text = raw.strip().upper()
    if text.startswith("Y"):
        return "Y"
    if text.startswith("N"):
        return "N"
    return ""


def task2_prompt(examples_text: str, sample_input: str, style: str) -> str:
    if style == "rubric":
        rule = (
            "Count only the requested part of speech in the quoted sentence. "
            "For verbs, count lexical verb forms and auxiliaries only when the official examples do so. "
            "For nouns, count common/proper nouns, not determiners or adjectives. Return one integer 0 to 6."
        )
    else:
        rule = "Follow the official examples exactly. Return only one integer from 0 to 6."
    return (
        "/no_think\n"
        "Task: count nouns or verbs in a sentence.\n"
        f"{rule}\n\n"
        "Official examples:\n"
        f"{examples_text}"
        "Target input:\n"
        f"{sample_input}\n"
        "Answer:"
    )


def task6_prompt(examples_text: str, sample_input: str, style: str) -> str:
    if style == "genre":
        rule = (
            "Determine whether both sentences fit the named genre. "
            "Return Y only if both sentences plausibly belong to that genre; otherwise return N. "
            "Use the genre descriptions and official examples."
        )
    else:
        rule = "Follow the official examples. Return exactly Y or N."
    return (
        "/no_think\n"
        "Task: same-genre classification for two sentences.\n"
        f"{rule}\n\n"
        "Genre descriptions: face-to-face conversations/dialogues; government public/government information; "
        "letters philanthropic fundraising; 9/11 attack-related information; slate cultural magazine topics; "
        "telephone telephonic dialogue; travel guide information; verbatim linguistics posts; oup nonfiction textile/child development; fiction popular fiction.\n\n"
        "Official examples:\n"
        f"{examples_text}"
        "Target input:\n"
        f"{sample_input}\n"
        "Answer:"
    )


def validation_indices(examples: list[dict[str, Any]], count: int) -> list[int]:
    step = max(1, len(examples) // count)
    return list(range(0, len(examples), step))[:count]


def validate_task(task_id: int, count: int) -> dict[str, Any]:
    task = load_task(task_id)
    configs = (
        [
            {"name": "same_type_examples", "limit": 120, "style": "examples", "same_type_first": True},
            {"name": "same_type_rubric", "limit": 120, "style": "rubric", "same_type_first": True},
            {"name": "nearest_examples", "limit": 120, "style": "examples", "same_type_first": False},
        ]
        if task_id == 2
        else [
            {"name": "same_genre_examples", "limit": 120, "style": "examples", "same_type_first": True},
            {"name": "same_genre_rubric", "limit": 120, "style": "genre", "same_type_first": True},
            {"name": "nearest_examples", "limit": 120, "style": "examples", "same_type_first": False},
        ]
    )
    norm = normalize_task2 if task_id == 2 else normalize_task6
    prompt_builder = task2_prompt if task_id == 2 else task6_prompt
    results = []
    for cfg in configs:
        detail = []
        correct = 0
        for idx in validation_indices(task["examples"], count):
            ex = task["examples"][idx]
            examples_text = retrieve_examples(
                task["examples"],
                ex["input"],
                limit=cfg["limit"],
                max_chars=28_000,
                exclude_id=ex["id"],
                same_type_first=cfg["same_type_first"],
            )
            raw = call_completion(prompt_builder(examples_text, ex["input"], cfg["style"]))
            pred = norm(raw)
            gold = label_of(ex)
            correct += int(pred == gold)
            detail.append({"id": ex["id"], "gold": gold, "pred": pred, "raw": raw})
            print(f"task{task_id} validate {cfg['name']} {len(detail)}/{count} pred={pred} gold={gold}", flush=True)
        results.append({"config": cfg, "correct": correct, "count": count, "accuracy": correct / count, "detail": detail})
    return {"task": task_id, "results": results}


def load_zip_rows(zip_name: str) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(OUTPUT_DIR / zip_name) as zf:
        for task_id in range(1, 9):
            rows[task_id] = [
                json.loads(line)
                for line in zf.read(f"openseek-{task_id}-v1.jsonl").decode("utf-8").splitlines()
                if line.strip()
            ]
    return rows


def generate_task(task_id: int, base_zip: str, validation_path: Path, max_changes: int) -> dict[str, Any]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    task_key = f"task{task_id}"
    best = max(validation[task_key]["results"], key=lambda row: row["accuracy"])
    cfg = best["config"]
    task = load_task(task_id)
    base_pred = {row["test_sample_id"]: str(row["prediction"]) for row in load_zip_rows(base_zip)[task_id]}
    norm = normalize_task2 if task_id == 2 else normalize_task6
    prompt_builder = task2_prompt if task_id == 2 else task6_prompt
    changes = []
    for idx, sample in enumerate(task["test_samples"], 1):
        examples_text = retrieve_examples(
            task["examples"],
            sample["input"],
            limit=cfg["limit"],
            max_chars=28_000,
            same_type_first=cfg["same_type_first"],
        )
        prompts = [
            prompt_builder(examples_text, sample["input"], cfg["style"]),
            prompt_builder(examples_text, sample["input"], "rubric" if task_id == 2 else "genre"),
            prompt_builder(examples_text, sample["input"], "examples"),
        ]
        raws = [call_completion(prompt) for prompt in prompts]
        votes = [norm(raw) for raw in raws]
        counter = Counter(votes)
        majority, majority_count = counter.most_common(1)[0]
        current = base_pred[sample["id"]]
        if majority and majority != current and majority_count >= 2:
            changes.append(
                {
                    "id": sample["id"],
                    "input": sample["input"],
                    "current": current,
                    "majority": majority,
                    "majority_count": majority_count,
                    "votes": votes,
                    "raw": raws,
                }
            )
        print(f"task{task_id} generate {idx}/{len(task['test_samples'])} current={current} votes={votes}", flush=True)
        if max_changes and len(changes) >= max_changes:
            # Continue would produce a full replacement strategy; for high precision candidate mining we stop at enough candidates.
            break
    return {"task": task_id, "base_zip": base_zip, "best_validation": best, "changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "generate"], required=True)
    parser.add_argument("--tasks", default="2,6")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--base-zip", default="submission_0513v4.zip")
    parser.add_argument("--max-changes", type=int, default=30)
    parser.add_argument("--validation-path", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    task_ids = [int(x) for x in args.tasks.split(",") if x.strip()]
    out_dir = OUTPUT_DIR / "0513_systematic"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "validate":
        result = {f"task{task_id}": validate_task(task_id, args.count) for task_id in task_ids}
        path = Path(args.output) if args.output else out_dir / "task2_6_validation.json"
    else:
        validation_path = Path(args.validation_path) if args.validation_path else out_dir / "task2_6_validation.json"
        result = {
            f"task{task_id}": generate_task(task_id, args.base_zip, validation_path, args.max_changes)
            for task_id in task_ids
        }
        path = Path(args.output) if args.output else out_dir / "task2_6_candidates.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
