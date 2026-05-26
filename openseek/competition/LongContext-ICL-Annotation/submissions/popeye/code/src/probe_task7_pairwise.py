import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from validate_task7_candidate_rerank import run_judge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triggered_bundle", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def run_pairwise(row: dict, left: str, right: str) -> dict:
    ab_winner, ab_raw = run_judge(
        category=row.get("category") or "",
        clue=row.get("clue") or "",
        candidates=[left, right],
        judge_mode=row.get("judge_mode") or "completion",
    )
    ba_winner, ba_raw = run_judge(
        category=row.get("category") or "",
        clue=row.get("clue") or "",
        candidates=[right, left],
        judge_mode=row.get("judge_mode") or "completion",
    )
    if ab_winner is None:
        ab_winner = left
    if ba_winner is None:
        ba_winner = right

    left_norm = normalize(left)
    right_norm = normalize(right)
    ab_prefers_right = normalize(ab_winner) == right_norm
    ba_prefers_right = normalize(ba_winner) == right_norm
    right_win_count = int(ab_prefers_right) + int(ba_prefers_right)

    if right_win_count == 2:
        verdict = "secondary_strong_win"
    elif right_win_count == 1:
        verdict = "split"
    else:
        verdict = "primary_hold"

    return {
        "primary_candidate": left,
        "secondary_candidate": right,
        "ab": {
            "order": [left, right],
            "judge_raw": ab_raw,
            "winner": ab_winner,
            "winner_is_secondary": ab_prefers_right,
        },
        "ba": {
            "order": [right, left],
            "judge_raw": ba_raw,
            "winner": ba_winner,
            "winner_is_secondary": ba_prefers_right,
        },
        "secondary_win_count": right_win_count,
        "secondary_preference_rate": right_win_count / 2.0,
        "verdict": verdict,
    }


def evaluate_row(row: dict) -> dict | None:
    primary = row.get("primary_judge_candidate") or row.get("vote_candidate")
    secondary_candidates = row.get("secondary_only_candidates") or []
    if not primary or not secondary_candidates:
        return None

    comparisons = [run_pairwise(row, primary, secondary) for secondary in secondary_candidates]
    strong_wins = sum(1 for comp in comparisons if comp["verdict"] == "secondary_strong_win")
    split_wins = sum(1 for comp in comparisons if comp["verdict"] == "split")
    avg_secondary_preference = sum(comp["secondary_preference_rate"] for comp in comparisons) / len(comparisons)
    return {
        "seed": row.get("seed"),
        "id": row.get("id"),
        "secondary_family": row.get("secondary_family"),
        "constraint_subtypes": row.get("constraint_subtypes") or [],
        "gold": row.get("gold"),
        "primary_candidate": primary,
        "secondary_only_candidates": secondary_candidates,
        "comparison_count": len(comparisons),
        "secondary_strong_win_count": strong_wins,
        "secondary_split_win_count": split_wins,
        "avg_secondary_preference_rate": avg_secondary_preference,
        "comparisons": comparisons,
    }


def build_report(triggered_bundle: dict) -> dict:
    rows = []
    family_counts = Counter()
    family_strong_wins = Counter()
    total_comparisons = 0
    total_strong_wins = 0
    total_split_wins = 0
    avg_preferences = []
    for raw_row in triggered_bundle.get("rows", []):
        row = evaluate_row(raw_row)
        if row is None:
            continue
        rows.append(row)
        family = row.get("secondary_family") or "unknown"
        family_counts[family] += row["comparison_count"]
        family_strong_wins[family] += row["secondary_strong_win_count"]
        total_comparisons += row["comparison_count"]
        total_strong_wins += row["secondary_strong_win_count"]
        total_split_wins += row["secondary_split_win_count"]
        avg_preferences.append(row["avg_secondary_preference_rate"])

    family_summary = {
        family: {
            "comparison_count": family_counts[family],
            "secondary_strong_win_count": family_strong_wins[family],
            "secondary_strong_win_rate": family_strong_wins[family] / family_counts[family] if family_counts[family] else 0.0,
        }
        for family in sorted(family_counts)
    }

    return {
        "triggered_bundle": triggered_bundle.get("report_files"),
        "evaluated_row_count": len(rows),
        "total_comparison_count": total_comparisons,
        "secondary_strong_win_count": total_strong_wins,
        "secondary_split_win_count": total_split_wins,
        "secondary_strong_win_rate": total_strong_wins / total_comparisons if total_comparisons else 0.0,
        "avg_secondary_preference_rate": sum(avg_preferences) / len(avg_preferences) if avg_preferences else 0.0,
        "family_summary": family_summary,
        "rows": rows,
    }


def main():
    args = parse_args()
    triggered_bundle = load_json(args.triggered_bundle)
    report = build_report(triggered_bundle)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
