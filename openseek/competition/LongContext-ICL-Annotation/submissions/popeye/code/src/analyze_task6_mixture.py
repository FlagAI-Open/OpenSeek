import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_reports", type=str, nargs="+", required=True)
    parser.add_argument("--n_thresholds", type=float, nargs="+", default=[0.12, 0.15, 0.18, 0.2, 0.22, 0.25])
    parser.add_argument("--n_max_thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--y_thresholds", type=float, nargs="*", default=[])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_rows(report_path: str) -> dict:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = payload["rows"]
    seed = payload["seed"]
    return {"seed": seed, "path": report_path, "rows": rows}


def apply_rule(row: dict, n_threshold: float, n_max_threshold: float | None, y_threshold: float | None) -> tuple[str, str]:
    min_score = row.get("min_genre_score")
    max_score = row.get("max_genre_score")
    if min_score is None or max_score is None:
        score1 = row["sentence1_genre_score"]
        score2 = row["sentence2_genre_score"]
        min_score = min(score1, score2)
        max_score = max(score1, score2)

    if min_score < n_threshold and (n_max_threshold is None or max_score <= n_max_threshold):
        return "N", "classifier_n"
    if y_threshold is not None and min_score >= y_threshold:
        return "Y", "classifier_y"
    return row["prediction"], "chat"


def score_rule(report: dict, n_threshold: float, n_max_threshold: float | None, y_threshold: float | None) -> dict:
    correct = 0
    source_counter = {"classifier_n": 0, "classifier_y": 0, "chat": 0}
    for row in report["rows"]:
        prediction, source = apply_rule(row, n_threshold=n_threshold, n_max_threshold=n_max_threshold, y_threshold=y_threshold)
        correct += int(prediction == row["gold"])
        source_counter[source] += 1
    sample_count = len(report["rows"])
    return {
        "seed": report["seed"],
        "avg_score": correct / sample_count if sample_count else 0.0,
        "source_counter": source_counter,
        "path": report["path"],
    }


def main():
    args = parse_args()
    reports = [load_rows(path) for path in args.chat_reports]
    y_thresholds = [None] + list(args.y_thresholds)

    candidates = []
    for n_threshold in args.n_thresholds:
        for n_max_threshold in [None] + list(args.n_max_thresholds):
            for y_threshold in y_thresholds:
                per_seed = [
                    score_rule(
                        report,
                        n_threshold=n_threshold,
                        n_max_threshold=n_max_threshold,
                        y_threshold=y_threshold,
                    )
                    for report in reports
                ]
                avg_score = sum(item["avg_score"] for item in per_seed) / len(per_seed)
                min_seed_score = min(item["avg_score"] for item in per_seed)
                candidates.append(
                    {
                        "n_threshold": n_threshold,
                        "n_max_threshold": n_max_threshold,
                        "y_threshold": y_threshold,
                        "avg_score": avg_score,
                        "min_seed_score": min_seed_score,
                        "per_seed": per_seed,
                    }
                )

    candidates.sort(key=lambda item: (item["avg_score"], item["min_seed_score"]), reverse=True)
    result = {"top_candidates": candidates[: args.top_k]}

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
