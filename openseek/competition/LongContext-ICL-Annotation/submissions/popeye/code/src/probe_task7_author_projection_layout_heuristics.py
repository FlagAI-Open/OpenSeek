import argparse
import json
from pathlib import Path

from validate_task7_candidate_rerank import run_judge


AUTHOR_BUCKET = "quoted_work_author_relation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_report", type=str, required=True)
    parser.add_argument("--side", type=str, choices=["on", "off"], default="on")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def dedupe_keep_order(candidates: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for candidate in candidates:
        key = normalize(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def build_secondary_only(primary_unique: list[str], judge_candidates: list[str]) -> list[str]:
    primary_norms = {normalize(candidate) for candidate in primary_unique if normalize(candidate)}
    secondary_only = []
    seen = set()
    for candidate in judge_candidates:
        key = normalize(candidate)
        if not key or key in primary_norms or key in seen:
            continue
        seen.add(key)
        secondary_only.append(candidate)
    return secondary_only


def reorder_candidates(row: dict, heuristic: str) -> list[str]:
    base_candidates = list(row["base_judge_candidates"])
    if heuristic == "baseline":
        return base_candidates

    if row["bucket"] != AUTHOR_BUCKET or not row["secondary_only"]:
        return base_candidates

    primary_unique = list(row["primary_unique"])
    secondary_only = list(row["secondary_only"])

    if heuristic == "author_secondary_first":
        return dedupe_keep_order(secondary_only + primary_unique)
    if heuristic == "author_primary_anchor_top1":
        anchor = primary_unique[:1]
        tail = primary_unique[1:]
        return dedupe_keep_order(anchor + secondary_only + tail)
    if heuristic == "author_primary_anchor_top2":
        anchor = primary_unique[:2]
        tail = primary_unique[2:]
        return dedupe_keep_order(anchor + secondary_only + tail)

    raise ValueError(f"Unsupported heuristic: {heuristic}")


def evaluate_row(replay_row: dict, side: str, heuristic: str) -> dict:
    variant = replay_row[side]
    base_judge_candidates = list(variant.get("judge", {}).get("candidates") or [])
    primary_unique = list(variant.get("primary", {}).get("unique_candidates") or [])
    secondary_only = build_secondary_only(primary_unique, base_judge_candidates)
    category = replay_row.get("category") or ""
    clue = replay_row.get("clue") or ""
    structured_row = {
        "bucket": replay_row.get("bucket"),
        "base_judge_candidates": base_judge_candidates,
        "primary_unique": primary_unique,
        "secondary_only": secondary_only,
    }
    reordered = reorder_candidates(structured_row, heuristic)
    winner, judge_raw = run_judge(
        category=category,
        clue=clue,
        candidates=reordered,
        judge_mode="completion",
    )
    if winner is None:
        winner = reordered[0] if reordered else None
    baseline = variant.get("prediction")
    return {
        "id": replay_row.get("test_sample_id"),
        "bucket": replay_row.get("bucket"),
        "baseline_prediction": baseline,
        "heuristic_prediction": winner,
        "changed_vs_baseline": normalize(winner) != normalize(baseline),
        "category": category,
        "clue": clue,
        "primary_unique": primary_unique,
        "secondary_only": secondary_only,
        "baseline_judge_candidates": base_judge_candidates,
        "heuristic_judge_candidates": reordered,
        "judge_raw": judge_raw,
    }


def evaluate_heuristic(report: dict, side: str, heuristic: str) -> dict:
    rows = [evaluate_row(replay_row, side, heuristic) for replay_row in report.get("rows", [])]
    changed_rows = [row for row in rows if row["changed_vs_baseline"]]
    target_changed = [row for row in changed_rows if row["bucket"] == AUTHOR_BUCKET]
    non_target_changed = [row for row in changed_rows if row["bucket"] != AUTHOR_BUCKET]
    return {
        "heuristic": heuristic,
        "row_count": len(rows),
        "changed_row_count": len(changed_rows),
        "target_changed_count": len(target_changed),
        "non_target_changed_count": len(non_target_changed),
        "rows": rows,
    }


def main():
    args = parse_args()
    report = load_json(args.replay_report)
    heuristics = [
        "baseline",
        "author_secondary_first",
        "author_primary_anchor_top1",
        "author_primary_anchor_top2",
    ]
    outputs = [evaluate_heuristic(report, args.side, heuristic) for heuristic in heuristics]
    result = {
        "replay_report": args.replay_report,
        "side": args.side,
        "seed": report.get("effective_config", {}).get("completion_seed"),
        "heuristics": outputs,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
