import argparse
import json
from collections import Counter
from pathlib import Path

from main import TASK_FILES
from method import get_task7_constraint_subtypes, normalize_task7_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_report", type=str, required=True)
    parser.add_argument("--task7_data_path", type=str, default=str(TASK_FILES[7]))
    parser.add_argument("--side", type=str, choices=["off", "on"], default="on")
    parser.add_argument(
        "--row_filter",
        type=str,
        choices=["all", "changed_only", "unchanged_only", "target_only", "non_target_only"],
        default="all",
    )
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(text: str | None) -> str:
    return normalize_task7_answer(text)


def matches_filter(row: dict, row_filter: str) -> bool:
    changed = bool(row.get("match_summary", {}).get("changed_in_replay"))
    target = row.get("bucket") == "quoted_work_author_relation"
    if row_filter == "all":
        return True
    if row_filter == "changed_only":
        return changed
    if row_filter == "unchanged_only":
        return not changed
    if row_filter == "target_only":
        return target
    if row_filter == "non_target_only":
        return not target
    raise ValueError(f"Unsupported row_filter: {row_filter}")


def build_secondary_only_candidates(primary_unique: list[str], judge_candidates: list[str]) -> list[str]:
    primary_norms = {normalize(candidate) for candidate in primary_unique if normalize(candidate)}
    secondary_only = []
    seen = set()
    for candidate in judge_candidates:
        normalized = normalize(candidate)
        if not normalized or normalized in primary_norms or normalized in seen:
            continue
        secondary_only.append(candidate)
        seen.add(normalized)
    return secondary_only


def build_gold_presence_bucket(gold: str, primary_unique: list[str], secondary_only: list[str]) -> str:
    gold_norm = normalize(gold)
    if not gold_norm:
        return "absent"
    in_primary = gold_norm in {normalize(candidate) for candidate in primary_unique}
    in_secondary = gold_norm in {normalize(candidate) for candidate in secondary_only}
    if in_primary and in_secondary:
        return "both"
    if in_primary:
        return "primary_only"
    if in_secondary:
        return "secondary_only"
    return "absent"


def build_row(sample_map: dict[str, dict], replay_report: dict, replay_row: dict, side: str) -> dict:
    sample_id = replay_row["test_sample_id"]
    sample = sample_map[sample_id]
    outputs = sample.get("output") or []
    gold = outputs[0] if outputs else ""
    variant = replay_row[side]
    judge_candidates = variant.get("judge", {}).get("candidates") or []
    primary_unique = variant.get("primary", {}).get("unique_candidates") or []
    secondary_only = build_secondary_only_candidates(primary_unique, judge_candidates)
    secondary_only_norms = [normalize(candidate) for candidate in secondary_only if normalize(candidate)]
    prediction = variant.get("prediction")
    prediction_norm = normalize(prediction)
    secondary_positions = [
        idx for idx, candidate in enumerate(judge_candidates) if normalize(candidate) in set(secondary_only_norms)
    ]
    category = replay_row.get("category") or ""
    return {
        "seed": replay_report.get("effective_config", {}).get("completion_seed"),
        "source_report": replay_report.get("runner", {}).get("command"),
        "id": sample_id,
        "category": category,
        "clue": replay_row.get("clue") or "",
        "secondary_family": variant.get("secondary_family") or "unknown",
        "constraint_subtypes": get_task7_constraint_subtypes(category),
        "gold": gold,
        "gold_presence_bucket": build_gold_presence_bucket(gold, primary_unique, secondary_only),
        "selection_risk_marker": variant.get("judge", {}).get("strategy"),
        "primary_pool_summary": variant.get("primary", {}).get("pool_summary") or {},
        "primary_judge_candidates": primary_unique,
        "judge_candidates_visible": judge_candidates,
        "judge_candidate_order": list(judge_candidates),
        "secondary_only_candidates": secondary_only,
        "secondary_only_candidate_norms": secondary_only_norms,
        "secondary_only_visible_count": len(secondary_positions),
        "secondary_only_positions": secondary_positions,
        "secondary_only_lead_position": secondary_positions[0] if secondary_positions else None,
        "secondary_only_in_top2": any(position < 2 for position in secondary_positions),
        "secondary_only_in_top3": any(position < 3 for position in secondary_positions),
        "vote_candidate": variant.get("primary", {}).get("vote_candidate"),
        "primary_judge_candidate": variant.get("primary", {}).get("vote_candidate"),
        "judge_candidate": prediction,
        "primary_judge_correct": normalize(variant.get("primary", {}).get("vote_candidate")) == normalize(gold),
        "judge_correct": prediction_norm == normalize(gold),
        "primary_oracle_hit": normalize(gold) in {normalize(candidate) for candidate in primary_unique},
        "oracle_hit": normalize(gold) in {normalize(candidate) for candidate in judge_candidates},
        "secondary_gate_triggered": bool(variant.get("secondary", {}).get("attempted")),
        "judge_secondary_visible": bool(secondary_only),
        "judge_selected_from_secondary": prediction_norm in set(secondary_only_norms),
        "oracle_gain_on_triggered": 0,
        "judge_gain_on_triggered": 0,
        "diagnosis_mode": False,
        "judge_mode": "completion",
        "row_bucket": replay_row.get("bucket"),
        "changed_in_replay": bool(replay_row.get("match_summary", {}).get("changed_in_replay")),
        "side": side,
    }


def build_bundle(replay_report: dict, task7_data: dict, side: str, row_filter: str) -> dict:
    sample_map = {sample["id"]: sample for sample in task7_data.get("test_samples", [])}
    rows = []
    family_counts = Counter()
    gold_bucket_counts = Counter()
    risk_counts = Counter()
    for replay_row in replay_report.get("rows", []):
        if not matches_filter(replay_row, row_filter):
            continue
        slim = build_row(sample_map, replay_report, replay_row, side)
        rows.append(slim)
        family_counts[slim["secondary_family"]] += 1
        gold_bucket_counts[slim["gold_presence_bucket"]] += 1
        risk_counts[slim["selection_risk_marker"] or "unknown"] += 1

    return {
        "report_files": [replay_report.get("inputs", {}).get("diff_audit_path"), replay_report.get("runner", {}).get("command")],
        "seed_count": 1,
        "triggered_row_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "gold_presence_bucket_counts": dict(sorted(gold_bucket_counts.items())),
        "selection_risk_marker_counts": dict(sorted(risk_counts.items())),
        "side": side,
        "row_filter": row_filter,
        "rows": rows,
    }


def main():
    args = parse_args()
    replay_report = load_json(args.replay_report)
    task7_data = load_json(args.task7_data_path)
    bundle = build_bundle(replay_report, task7_data, side=args.side, row_filter=args.row_filter)
    text = json.dumps(bundle, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
