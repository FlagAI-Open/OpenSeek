import argparse
import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TASK6_PATTERN = re.compile(r"Sentence 1: (.*?) Sentence 2: (.*?) Genre: (.*)$", re.S)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
TASK6_FILE = PROJECT_DIR / "data" / "openseek-6_mnli_same_genre_classification.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def parse_task6_fields(text: str) -> tuple[str, str, str]:
    match = TASK6_PATTERN.match(text)
    if not match:
        raise ValueError(f"Unable to parse task6 input: {text[:120]}")
    sentence1, sentence2, genre = match.groups()
    return sentence1.strip(), sentence2.strip(), genre.strip().rstrip(".")


def build_sentence_genre_classifier(train_examples: list[dict]):
    texts = []
    labels = []
    for example in train_examples:
        sentence1, sentence2, genre = parse_task6_fields(example["input"])
        if example["output"][0] != "Y":
            continue
        texts.extend([sentence1, sentence2])
        labels.extend([genre, genre])

    classifier = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
        LogisticRegression(max_iter=2000),
    )
    classifier.fit(texts, labels)
    return classifier


def score_task6_target_genre(classifier, sentence1: str, sentence2: str, genre: str) -> tuple[float, float]:
    class_labels = list(classifier.classes_)
    genre_index = class_labels.index(genre)
    probs1 = classifier.predict_proba([sentence1])[0]
    probs2 = classifier.predict_proba([sentence2])[0]
    return float(probs1[genre_index]), float(probs2[genre_index])


def main():
    args = parse_args()

    task_dict = json.loads(TASK6_FILE.read_text(encoding="utf-8"))
    examples = list(task_dict["examples"])

    rng = random.Random(args.seed + 6)
    holdout = rng.sample(examples, min(args.sample_limit, len(examples)))
    holdout_ids = {example["id"] for example in holdout}
    train_examples = [example for example in examples if example["id"] not in holdout_ids]

    classifier = build_sentence_genre_classifier(train_examples)
    rows = []
    correct = 0

    for sample in holdout:
        sentence1, sentence2, genre = parse_task6_fields(sample["input"])
        score1, score2 = score_task6_target_genre(classifier, sentence1, sentence2, genre)
        prediction = "Y" if score1 >= args.threshold and score2 >= args.threshold else "N"
        gold = sample["output"][0]
        ok = prediction == gold
        correct += int(ok)
        rows.append(
            {
                "id": sample["id"],
                "genre": genre,
                "gold": gold,
                "prediction": prediction,
                "sentence1_genre_score": round(score1, 6),
                "sentence2_genre_score": round(score2, 6),
                "correct": ok,
            }
        )

    report = {
        "seed": args.seed,
        "sample_limit": len(holdout),
        "threshold": args.threshold,
        "avg_score": correct / len(holdout) if holdout else 0.0,
        "rows": rows,
    }

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
