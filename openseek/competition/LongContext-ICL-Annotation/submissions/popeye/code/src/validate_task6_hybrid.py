import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

from validate_task6_genre_classifier import (
    TASK6_FILE,
    build_sentence_genre_classifier,
    parse_task6_fields,
    score_task6_target_genre,
)

DEFAULT_COMPLETION_URL = "http://127.0.0.1:2026/v1/completions"
DEFAULT_CHAT_URL = DEFAULT_COMPLETION_URL.replace("/v1/completions", "/v1/chat/completions")
DEFAULT_REQUEST_TIMEOUT = 300
TASK_REQUEST_ATTEMPTS = {6: 2}
TASK_CHAT_SYSTEM = {
    6: "You solve natural language inference. Compare Sentence 1 and Sentence 2 carefully, then answer with ONLY 'Y' or 'N'.",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_threshold", type=float, default=0.2)
    parser.add_argument("--n_max_threshold", type=float, default=None)
    parser.add_argument("--examples_limit", type=int, default=100)
    parser.add_argument("--chat_temperature", type=float, default=0.6)
    parser.add_argument("--chat_top_p", type=float, default=0.95)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def build_chat_examples(examples: list[dict], examples_limit: int) -> str:
    blocks = []
    for example in examples[:examples_limit]:
        blocks.append(f"Input: {example['input']}\nAnswer: {example['output'][0]}\n")
    return "\n".join(blocks)


def build_chat_user_message(examples_str: str, text2annotate: str, task_description: str) -> str:
    return (
        f"Task: {task_description}\n\n"
        f"Examples:\n{examples_str}\n"
        f"Input: {text2annotate}\n"
        "Answer:"
    )


def extract_chat_answer(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"<[^>]+>", "", text).strip()
    matches = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text.upper())
    return matches[-1] if matches else None


def request_chat_prediction(
    examples_str: str,
    text2annotate: str,
    task_description: str,
    chat_temperature: float,
    chat_top_p: float,
) -> str | None:
    data = {
        "model": "/mnt/nvme_data/qhuser/PJC/flagos/models/Qwen3-4B",
        "messages": [
            {"role": "system", "content": TASK_CHAT_SYSTEM[6]},
            {"role": "user", "content": build_chat_user_message(examples_str, text2annotate, task_description)},
        ],
        "max_tokens": 800,
        "temperature": chat_temperature,
        "top_p": chat_top_p,
    }

    attempts = TASK_REQUEST_ATTEMPTS.get(6, 1)
    for attempt in range(attempts):
        try:
            resp = requests.post(DEFAULT_CHAT_URL, json=data, timeout=DEFAULT_REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            prediction = extract_chat_answer(content)
            if prediction is not None:
                return prediction
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    return None


def main():
    args = parse_args()

    task_dict = json.loads(TASK6_FILE.read_text(encoding="utf-8"))
    examples = list(task_dict["examples"])
    task_description = task_dict["Definition"][0]

    rng = random.Random(args.seed + 6)
    holdout = rng.sample(examples, min(args.sample_limit, len(examples)))
    holdout_ids = {example["id"] for example in holdout}
    train_examples = [example for example in examples if example["id"] not in holdout_ids]
    icl_examples = train_examples[: args.examples_limit]

    classifier = build_sentence_genre_classifier(train_examples)
    chat_examples = build_chat_examples(icl_examples, args.examples_limit)

    rows = []
    correct = 0
    source_counter = {"classifier_n": 0, "chat": 0}

    for sample in holdout:
        sentence1, sentence2, genre = parse_task6_fields(sample["input"])
        score1, score2 = score_task6_target_genre(classifier, sentence1, sentence2, genre)
        min_score = min(score1, score2)
        max_score = max(score1, score2)

        classifier_n_ok = min_score < args.n_threshold
        if classifier_n_ok and args.n_max_threshold is not None:
            classifier_n_ok = max_score <= args.n_max_threshold

        if classifier_n_ok:
            prediction = "N"
            source = "classifier_n"
        else:
            prediction = request_chat_prediction(
                chat_examples,
                sample["input"],
                task_description,
                chat_temperature=args.chat_temperature,
                chat_top_p=args.chat_top_p,
            )
            source = "chat"

        gold = sample["output"][0]
        ok = prediction == gold
        correct += int(ok)
        source_counter[source] += 1
        rows.append(
            {
                "id": sample["id"],
                "genre": genre,
                "gold": gold,
                "prediction": prediction,
                "source": source,
                "sentence1_genre_score": round(score1, 6),
                "sentence2_genre_score": round(score2, 6),
                "min_genre_score": round(min_score, 6),
                "max_genre_score": round(max_score, 6),
                "correct": ok,
            }
        )

    report = {
        "seed": args.seed,
        "sample_limit": len(holdout),
        "n_threshold": args.n_threshold,
        "n_max_threshold": args.n_max_threshold,
        "examples_limit": args.examples_limit,
        "chat_temperature": args.chat_temperature,
        "chat_top_p": args.chat_top_p,
        "avg_score": correct / len(holdout) if holdout else 0.0,
        "source_counter": source_counter,
        "rows": rows,
    }

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
