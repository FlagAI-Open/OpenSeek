import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_files",
        nargs="+",
        required=True,
        help="Task7 rerank holdout JSON files produced by validate_task7_candidate_rerank.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional JSON path to save the report.",
    )
    return parser.parse_args()


def candidate_count(row: dict) -> int:
    counts = row.get("candidate_counts") or {}
    return len(counts)


def summarize_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])

    unique_hist = Counter()
    oracle_miss_rows = []
    judge_miss_rows = []
    vote_help_rows = []

    for row in rows:
        num_unique = candidate_count(row)
        unique_hist[num_unique] += 1

        slim_row = {
            "id": row.get("id"),
            "category": row.get("category"),
            "clue": row.get("clue"),
            "gold": row.get("gold"),
            "judge_candidate": row.get("judge_candidate"),
            "vote_candidate": row.get("vote_candidate"),
            "candidate_counts": row.get("candidate_counts"),
        }

        if not row.get("oracle_hit"):
            oracle_miss_rows.append(slim_row | {"unique_candidates": num_unique})
        elif not row.get("judge_correct"):
            judge_miss_rows.append(slim_row | {"unique_candidates": num_unique})

        if row.get("judge_correct") and not row.get("vote_correct"):
            vote_help_rows.append(slim_row | {"unique_candidates": num_unique})

    total = len(rows)
    oracle_hit_count = sum(bool(row.get("oracle_hit")) for row in rows)
    judge_correct_count = sum(bool(row.get("judge_correct")) for row in rows)
    vote_correct_count = sum(bool(row.get("vote_correct")) for row in rows)
    oracle_hit_but_judge_wrong = sum(
        bool(row.get("oracle_hit")) and not bool(row.get("judge_correct")) for row in rows
    )
    ambiguous_pool_count = sum(candidate_count(row) >= 4 for row in rows)
    collapsed_pool_count = sum(candidate_count(row) == 1 for row in rows)

    return {
        "file": str(path),
        "seed": payload.get("seed"),
        "sample_limit": payload.get("sample_limit"),
        "profile": payload.get("profile"),
        "retrieval_mode": payload.get("retrieval_mode"),
        "n_candidates": payload.get("n_candidates"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "vote_accuracy": payload.get("vote_accuracy"),
        "judge_accuracy": payload.get("judge_accuracy"),
        "oracle_hit_rate": payload.get("oracle_hit_rate"),
        "row_count": total,
        "vote_correct_count": vote_correct_count,
        "judge_correct_count": judge_correct_count,
        "oracle_hit_count": oracle_hit_count,
        "oracle_miss_count": total - oracle_hit_count,
        "oracle_hit_but_judge_wrong": oracle_hit_but_judge_wrong,
        "judge_help_count": len(vote_help_rows),
        "ambiguous_pool_count": ambiguous_pool_count,
        "collapsed_pool_count": collapsed_pool_count,
        "unique_candidate_histogram": dict(sorted(unique_hist.items())),
        "examples": {
            "oracle_miss_top10": oracle_miss_rows[:10],
            "judge_miss_top10": judge_miss_rows[:10],
            "judge_help_top10": vote_help_rows[:10],
        },
    }


def build_report(result_files: list[str]) -> dict:
    reports = [summarize_file(Path(path)) for path in result_files]

    aggregate = {
        "file_count": len(reports),
        "avg_vote_accuracy": round(sum(report["vote_accuracy"] for report in reports) / len(reports), 6),
        "avg_judge_accuracy": round(sum(report["judge_accuracy"] for report in reports) / len(reports), 6),
        "avg_oracle_hit_rate": round(sum(report["oracle_hit_rate"] for report in reports) / len(reports), 6),
        "total_oracle_miss_count": sum(report["oracle_miss_count"] for report in reports),
        "total_oracle_hit_but_judge_wrong": sum(report["oracle_hit_but_judge_wrong"] for report in reports),
        "total_judge_help_count": sum(report["judge_help_count"] for report in reports),
    }

    return {
        "result_files": result_files,
        "reports": reports,
        "aggregate": aggregate,
    }


def main():
    args = parse_args()
    report = build_report(args.result_files)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        Path(args.output_path).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
