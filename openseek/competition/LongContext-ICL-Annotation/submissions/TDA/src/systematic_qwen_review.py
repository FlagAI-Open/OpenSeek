import argparse
import difflib
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
    5: DATA_DIR / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    7: DATA_DIR / "openseek-7_jeopardy_answer_generation_all.json",
}


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads(TASK_FILES[task_id].read_text(encoding="utf-8"))


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


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9#@]+", text.lower()))


def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def example_label(ex: dict[str, Any]) -> str:
    out = ex["output"]
    return str(out[0] if isinstance(out, list) else out)


def call_completion(prompt: str, max_tokens: int = 32) -> str:
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
        except Exception as exc:  # noqa: BLE001 - keep long review resilient.
            last_error = exc
            time.sleep(2 + attempt)
    return f"ERROR: {last_error}"


def normalize_task5(text: str) -> str:
    lower = text.strip().lower()
    if "not sad" in lower or lower in {"not", "no", "n"}:
        return "Not sad"
    if "sad" in lower or lower in {"yes", "y"}:
        return "Sad"
    return ""


def normalize_task7(text: str) -> str:
    answer = text.strip().lower()
    answer = re.sub(r"^(final answer|answer|output|prediction)\s*:\s*", "", answer)
    answer = re.sub(r"^(what is|who is)\s+", "", answer)
    return answer.strip(" \t\n\r\"'`.,;:!?")


def retrieve_examples(
    examples: list[dict[str, Any]],
    query: str,
    limit: int,
    max_chars: int,
    exclude_id: str | None = None,
    balanced: bool = False,
) -> str:
    query_words = words(query)
    scored_by_label: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, ex in enumerate(examples):
        if exclude_id and ex["id"] == exclude_id:
            continue
        ex_words = words(ex["input"])
        score = len(query_words & ex_words) / math.sqrt(max(len(ex_words), 1))
        label = example_label(ex)
        scored.append((score, -idx, ex))
        scored_by_label[label].append((score, -idx, ex))

    selected: list[dict[str, Any]] = []
    if balanced:
        per_label = max(1, limit // max(len(scored_by_label), 1))
        for label in sorted(scored_by_label):
            selected.extend([ex for _, __, ex in sorted(scored_by_label[label], reverse=True)[:per_label]])
    selected_ids = {ex["id"] for ex in selected}
    for _, __, ex in sorted(scored, reverse=True):
        if len(selected) >= limit:
            break
        if ex["id"] not in selected_ids:
            selected.append(ex)
            selected_ids.add(ex["id"])

    chunks: list[str] = []
    total = 0
    for ex in selected:
        label = example_label(ex)
        if label in {"Sad", "Not sad"}:
            chunk = f"Tweet: {ex['input']}\nLabel: {label}\n\n"
        else:
            chunk = f"Input:\n{ex['input']}\nOutput:\n{label}\n\n"
        if chunks and total + len(chunk) > max_chars:
            continue
        chunks.append(chunk)
        total += len(chunk)
    return "".join(chunks)


def task5_prompt(examples_text: str, tweet: str, style: str) -> str:
    if style == "strict":
        rule = "Return exactly one label: Sad or Not sad."
    elif style == "perspective":
        rule = (
            "If the author expresses sadness, grief, depression, loneliness, hopelessness, or personal distress, return Sad. "
            "If it is joking, news, anger, complaint, or a hashtag without personal sadness, return Not sad. "
            "Return exactly Sad or Not sad."
        )
    else:
        rule = "Follow the official examples and return exactly Sad or Not sad."
    return (
        "/no_think\n"
        "Task: sadness detection for tweets.\n"
        f"{rule}\n\n"
        "Official examples:\n"
        f"{examples_text}"
        f"Tweet: {tweet}\n"
        "Label:"
    )


def task7_prompt(examples_text: str, sample_input: str, style: str) -> str:
    if style == "short":
        rule = "Return only the short final answer in lowercase. No explanation."
    else:
        rule = "Use the official Jeopardy examples to match answer style. Return only the final answer in lowercase."
    return (
        "/no_think\n"
        "Task: answer a Jeopardy clue.\n"
        f"{rule}\n\n"
        "Official examples:\n"
        f"{examples_text}"
        "Target:\n"
        f"{sample_input}\n"
        "Answer:"
    )


def validation_indices(examples: list[dict[str, Any]], count: int) -> list[int]:
    if count >= len(examples):
        return list(range(len(examples)))
    step = max(1, len(examples) // count)
    indices = list(range(0, len(examples), step))[:count]
    return indices


def validate_task5(count: int = 80) -> dict[str, Any]:
    task = load_task(5)
    examples = task["examples"]
    configs = [
        {"name": "balanced_strict", "limit": 80, "balanced": True, "style": "strict"},
        {"name": "balanced_perspective", "limit": 80, "balanced": True, "style": "perspective"},
        {"name": "nearest_examples", "limit": 100, "balanced": False, "style": "examples"},
    ]
    rows = []
    for cfg in configs:
        correct = 0
        detail = []
        for idx in validation_indices(examples, count):
            ex = examples[idx]
            examples_text = retrieve_examples(
                examples,
                ex["input"],
                limit=cfg["limit"],
                max_chars=28_000,
                exclude_id=ex["id"],
                balanced=cfg["balanced"],
            )
            raw = call_completion(task5_prompt(examples_text, ex["input"], cfg["style"]), max_tokens=16)
            pred = normalize_task5(raw)
            gold = example_label(ex)
            correct += int(pred == gold)
            detail.append({"id": ex["id"], "gold": gold, "pred": pred, "raw": raw})
            print(f"task5 validate {cfg['name']} {len(detail)}/{count} {pred} gold={gold}", flush=True)
        rows.append({"config": cfg, "accuracy": correct / count, "correct": correct, "count": count, "detail": detail})
    return {"task": 5, "results": rows}


def current_predictions(task_id: int, zip_name: str) -> dict[str, str]:
    rows = load_zip_rows(zip_name)[task_id]
    return {row["test_sample_id"]: str(row["prediction"]) for row in rows}


def review_task5_tests(zip_name: str, cfg: dict[str, Any], limit: int = 120) -> dict[str, Any]:
    task = load_task(5)
    cur = current_predictions(5, zip_name)
    candidates = []
    sad_terms = re.compile(r"\b(depress(?:ed|ion)?|cry(?:ing)?|tears?|hopeless|lonely|miserable|heartbroken|grief|sad)\b", re.I)
    normalized_examples = [
        {
            "id": ex["id"],
            "input": ex["input"],
            "label": example_label(ex),
            "norm": norm_space(re.sub(r"[^a-z0-9#@ ]+", " ", ex["input"].lower())),
            "words": words(ex["input"]),
        }
        for ex in task["examples"]
    ]
    for sample in task["test_samples"]:
        current = cur[sample["id"]]
        text = sample["input"]
        sample_words = words(text)
        sample_norm = norm_space(re.sub(r"[^a-z0-9#@ ]+", " ", text.lower()))
        best_match: dict[str, Any] | None = None
        for ex in normalized_examples:
            union = sample_words | ex["words"]
            jaccard = len(sample_words & ex["words"]) / len(union) if union else 0.0
            if jaccard < 0.35 and len(sample_words & ex["words"]) < 4:
                continue
            seq = difflib.SequenceMatcher(None, sample_norm, ex["norm"]).ratio()
            score = max(seq, jaccard)
            if best_match is None or score > best_match["score"]:
                best_match = {
                    "id": ex["id"],
                    "input": ex["input"],
                    "label": ex["label"],
                    "score": score,
                    "seq": seq,
                    "jaccard": jaccard,
                }
        reason = []
        # Candidate pool: current Not sad but strong sadness terms, or model-disputed near duplicates.
        if current == "Not sad" and sad_terms.search(text):
            reason.append("sad_terms")
        elif current == "Sad" and any(token in text.lower() for token in ["lol", "haha", "programme", "blessed"]):
            reason.append("anti_sad_terms")
        elif best_match and best_match["label"] != current and (
            best_match["score"] >= 0.72 or (best_match["seq"] >= 0.66 and best_match["jaccard"] >= 0.45)
        ):
            sample = dict(sample)
            sample["_nearest"] = best_match
            sample["_reason"] = ["nearest_conflict"]
            candidates.append(sample)
            continue
        if reason:
            sample = dict(sample)
            sample["_nearest"] = best_match
            sample["_reason"] = reason
            candidates.append(sample)
    candidates = candidates[:limit]
    detail = []
    for sample in candidates:
        examples_text = retrieve_examples(
            task["examples"],
            sample["input"],
            limit=cfg["limit"],
            max_chars=28_000,
            balanced=cfg["balanced"],
        )
        prompts = [
            task5_prompt(examples_text, sample["input"], cfg["style"]),
            task5_prompt(examples_text, sample["input"], "perspective"),
            task5_prompt(examples_text, sample["input"], "strict"),
        ]
        raws = [call_completion(prompt, max_tokens=16) for prompt in prompts]
        votes = [normalize_task5(raw) for raw in raws]
        counts = Counter(votes)
        majority, majority_count = counts.most_common(1)[0]
        detail.append(
            {
                "id": sample["id"],
                "input": sample["input"],
                "current": cur[sample["id"]],
                "votes": votes,
                "raw": raws,
                "majority": majority,
                "majority_count": majority_count,
                "nearest": sample.get("_nearest"),
                "reason": sample.get("_reason", []),
            }
        )
        print(f"task5 review {len(detail)}/{len(candidates)} {sample['id']} current={cur[sample['id']]} votes={votes}", flush=True)
    return {"task": 5, "base_zip": zip_name, "config": cfg, "detail": detail}


def validate_task7(count: int = 40) -> dict[str, Any]:
    task = load_task(7)
    examples = task["examples"]
    configs = [
        {"name": "nearest_short", "limit": 60, "balanced": False, "style": "short"},
        {"name": "nearest_style", "limit": 80, "balanced": False, "style": "style"},
    ]
    rows = []
    for cfg in configs:
        correct = 0
        detail = []
        for idx in validation_indices(examples, count):
            ex = examples[idx]
            examples_text = retrieve_examples(
                examples,
                ex["input"],
                limit=cfg["limit"],
                max_chars=28_000,
                exclude_id=ex["id"],
                balanced=False,
            )
            raw = call_completion(task7_prompt(examples_text, ex["input"], cfg["style"]), max_tokens=48)
            pred = normalize_task7(raw)
            gold = normalize_task7(example_label(ex))
            correct += int(pred == gold)
            detail.append({"id": ex["id"], "gold": gold, "pred": pred, "raw": raw})
            print(f"task7 validate {cfg['name']} {len(detail)}/{count} pred={pred!r} gold={gold!r}", flush=True)
        rows.append({"config": cfg, "accuracy": correct / count, "correct": correct, "count": count, "detail": detail})
    return {"task": 7, "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "review-task5"], required=True)
    parser.add_argument("--base-zip", default="submission_0513v4.zip")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--output", default="")
    parser.add_argument("--validation-path", default="")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / "0513_systematic"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "validate":
        result = {"task5": validate_task5(args.count), "task7": validate_task7(max(20, args.count // 2))}
        path = Path(args.output) if args.output else out_dir / "validation_results.json"
    else:
        validation_path = Path(args.validation_path) if args.validation_path else out_dir / "validation_results.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            best = max(validation["task5"]["results"], key=lambda row: row["accuracy"])["config"]
        else:
            best = {"name": "balanced_perspective", "limit": 80, "balanced": True, "style": "perspective"}
        result = review_task5_tests(args.base_zip, best)
        path = Path(args.output) if args.output else out_dir / "task5_review_results.json"

    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
