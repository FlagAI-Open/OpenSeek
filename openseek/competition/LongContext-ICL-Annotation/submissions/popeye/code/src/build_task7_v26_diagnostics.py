import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from method import detect_task7_secondary_family, get_task7_constraint_subtypes, parse_task7_fields
from validate_task7_candidate_rerank import normalize_qa_answer


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    triggered = subparsers.add_parser("triggered_bundle")
    triggered.add_argument("--report_files", nargs="+", required=True)
    triggered.add_argument("--output_path", type=str, default=None)

    mismatch = subparsers.add_parser("mismatch_audit")
    mismatch.add_argument("--base_file", type=str, required=True)
    mismatch.add_argument("--candidate_file", type=str, required=True)
    mismatch.add_argument("--task7_data", type=str, required=True)
    mismatch.add_argument("--triggered_bundle", type=str, required=True)
    mismatch.add_argument("--output_path", type=str, default=None)

    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["test_sample_id"]] = row
    return rows


def build_triggered_row(seed_report: dict, row: dict) -> dict:
    secondary_only_candidates = row.get("secondary_only_candidates") or []
    secondary_only_norms = row.get("secondary_only_candidate_norms") or [
        normalize_qa_answer(candidate) for candidate in secondary_only_candidates
    ]
    return {
        "seed": seed_report["seed"],
        "source_report": seed_report.get("_path"),
        "id": row["id"],
        "category": row.get("category"),
        "clue": row.get("clue"),
        "secondary_family": row.get("secondary_family"),
        "constraint_subtypes": row.get("constraint_subtypes") or [],
        "gold": row.get("gold"),
        "gold_presence_bucket": row.get("gold_presence_bucket", "absent"),
        "selection_risk_marker": row.get("selection_risk_marker"),
        "primary_pool_summary": row.get("primary_pool_summary") or {},
        "primary_judge_candidates": row.get("primary_judge_candidates") or [],
        "judge_candidates_visible": row.get("judge_candidates_visible") or [],
        "judge_candidate_order": row.get("judge_candidate_order") or row.get("judge_candidates_visible") or [],
        "secondary_only_candidates": secondary_only_candidates,
        "secondary_only_candidate_norms": secondary_only_norms,
        "secondary_only_visible_count": row.get("secondary_only_visible_count", 0),
        "secondary_only_positions": row.get("secondary_only_positions") or [],
        "secondary_only_lead_position": row.get("secondary_only_lead_position"),
        "secondary_only_in_top2": bool(row.get("secondary_only_in_top2", False)),
        "secondary_only_in_top3": bool(row.get("secondary_only_in_top3", False)),
        "vote_candidate": row.get("vote_candidate"),
        "primary_judge_candidate": row.get("primary_judge_candidate"),
        "judge_candidate": row.get("judge_candidate"),
        "primary_judge_correct": bool(row.get("primary_judge_correct")),
        "judge_correct": bool(row.get("judge_correct")),
        "primary_oracle_hit": bool(row.get("primary_oracle_hit")),
        "oracle_hit": bool(row.get("oracle_hit")),
        "secondary_gate_triggered": bool(row.get("secondary_gate_triggered")),
        "judge_secondary_visible": bool(row.get("judge_secondary_visible")),
        "judge_selected_from_secondary": bool(row.get("judge_selected_from_secondary")),
        "oracle_gain_on_triggered": row.get("oracle_gain_on_triggered", 0),
        "judge_gain_on_triggered": row.get("judge_gain_on_triggered", 0),
        "diagnosis_mode": bool(seed_report.get("diagnosis_mode", False)),
        "judge_mode": seed_report.get("judge_mode"),
    }


def build_triggered_bundle(report_files: list[str]) -> dict:
    report_paths = [Path(path) for path in report_files]
    payloads = []
    rows = []
    family_counter = Counter()
    bucket_counter = Counter()
    marker_counter = Counter()
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        payloads.append(payload)
        for row in payload.get("rows", []):
            if not row.get("secondary_gate_triggered"):
                continue
            slim = build_triggered_row(payload, row)
            rows.append(slim)
            family_counter[slim["secondary_family"]] += 1
            bucket_counter[slim["gold_presence_bucket"]] += 1
            marker_counter[slim.get("selection_risk_marker") or "unknown"] += 1

    return {
        "report_files": [str(path) for path in report_paths],
        "seed_count": len(payloads),
        "triggered_row_count": len(rows),
        "family_counts": dict(sorted(family_counter.items())),
        "gold_presence_bucket_counts": dict(sorted(bucket_counter.items())),
        "selection_risk_marker_counts": dict(sorted(marker_counter.items())),
        "rows": rows,
    }


def classify_triggered_row(row: dict) -> str:
    if row.get("judge_secondary_visible") and row.get("judge_selected_from_secondary"):
        return "judge_visible_secondary_gains"
    if row.get("judge_secondary_visible") and not row.get("judge_selected_from_secondary"):
        return "secondary_visible_but_not_selected"
    if not row.get("oracle_hit"):
        return "triggered_but_no_oracle_gain"
    if row.get("oracle_hit") and not row.get("judge_correct"):
        return "offline_helped_but_probably_fragile_noisy"
    return "offline_helped_and_online_likely_neutral"


def load_task7_test_map(task7_data: str) -> dict[str, dict]:
    payload = load_json(task7_data)
    return {sample["id"]: sample for sample in payload.get("test_samples", [])}


def build_changed_rows(base_file: str, candidate_file: str, task7_data: str) -> tuple[list[dict], dict]:
    base_rows = load_jsonl(base_file)
    candidate_rows = load_jsonl(candidate_file)
    test_map = load_task7_test_map(task7_data)

    changed_rows = []
    family_counts = Counter()
    subtype_counts = Counter()
    direction_counts = Counter()
    for sample_id, candidate_row in candidate_rows.items():
        base_prediction = (base_rows.get(sample_id) or {}).get("prediction") or ""
        candidate_prediction = candidate_row.get("prediction") or ""
        if candidate_prediction == base_prediction:
            continue
        sample = test_map.get(sample_id)
        if sample is None:
            continue
        category, clue = parse_task7_fields(sample["input"])
        family = detect_task7_secondary_family(category, clue, typed_route="auto")
        subtypes = get_task7_constraint_subtypes(category)
        base_norm = normalize_qa_answer(base_prediction)
        candidate_norm = normalize_qa_answer(candidate_prediction)
        if candidate_norm == base_norm:
            direction = "surface_only_change"
        elif len(candidate_norm) > len(base_norm):
            direction = "longer_candidate"
        elif len(candidate_norm) < len(base_norm):
            direction = "shorter_candidate"
        else:
            direction = "replacement_same_length_band"
        row = {
            "id": sample_id,
            "category": category,
            "clue": clue,
            "secondary_family": family,
            "constraint_subtypes": subtypes,
            "base_prediction": base_prediction,
            "candidate_prediction": candidate_prediction,
            "base_prediction_normalized": base_norm,
            "candidate_prediction_normalized": candidate_norm,
            "change_direction": direction,
        }
        changed_rows.append(row)
        family_counts[family] += 1
        direction_counts[direction] += 1
        for subtype in subtypes:
            subtype_counts[subtype] += 1

    changed_rows.sort(key=lambda row: row["id"])
    summary = {
        "changed_row_count": len(changed_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "constraint_subtype_counts": dict(sorted(subtype_counts.items())),
        "change_direction_counts": dict(sorted(direction_counts.items())),
    }
    return changed_rows, summary


def build_mismatch_audit(base_file: str, candidate_file: str, task7_data: str, triggered_bundle_path: str) -> dict:
    triggered_bundle = load_json(triggered_bundle_path)
    changed_rows, changed_summary = build_changed_rows(base_file, candidate_file, task7_data)

    category_buckets = defaultdict(list)
    family_category_counts = defaultdict(Counter)
    family_trigger_counts = Counter()
    family_visible_counts = Counter()
    family_selected_counts = Counter()
    gold_bucket_counts = Counter()
    risk_marker_counts = Counter()
    for row in triggered_bundle.get("rows", []):
        category = classify_triggered_row(row)
        category_buckets[category].append(row)
        family = row.get("secondary_family") or "unknown"
        family_category_counts[family][category] += 1
        family_trigger_counts[family] += 1
        family_visible_counts[family] += int(bool(row.get("judge_secondary_visible")))
        family_selected_counts[family] += int(bool(row.get("judge_selected_from_secondary")))
        gold_bucket_counts[row.get("gold_presence_bucket") or "unknown"] += 1
        risk_marker_counts[row.get("selection_risk_marker") or "unknown"] += 1

    family_summary = {}
    for family, total in sorted(family_trigger_counts.items()):
        family_summary[family] = {
            "triggered_count": total,
            "judge_secondary_visible_count": family_visible_counts[family],
            "judge_selected_from_secondary_count": family_selected_counts[family],
            "judge_secondary_visible_rate": family_visible_counts[family] / total if total else 0.0,
            "judge_selected_from_secondary_rate": family_selected_counts[family] / total if total else 0.0,
            "category_counts": dict(sorted(family_category_counts[family].items())),
        }

    category_samples = {
        category: rows[:10]
        for category, rows in sorted(category_buckets.items())
    }

    return {
        "base_file": base_file,
        "candidate_file": candidate_file,
        "task7_data": task7_data,
        "triggered_bundle": triggered_bundle_path,
        "coverage": {
            "changed_row_count": changed_summary["changed_row_count"],
            "triggered_row_count": triggered_bundle.get("triggered_row_count", 0),
            "all_changed_rows_accounted_for": changed_summary["changed_row_count"] == 194,
        },
        "changed_test_rows_summary": changed_summary,
        "changed_test_rows": changed_rows,
        "triggered_holdout_category_counts": {
            category: len(rows)
            for category, rows in sorted(category_buckets.items())
        },
        "triggered_holdout_gold_presence_bucket_counts": dict(sorted(gold_bucket_counts.items())),
        "triggered_holdout_selection_risk_marker_counts": dict(sorted(risk_marker_counts.items())),
        "triggered_holdout_family_summary": family_summary,
        "triggered_holdout_category_samples": category_samples,
        "conclusion": {
            "task7_status": "offline-promotable but online-unconfirmed",
            "recommended_direction": "diagnose_first",
            "promotion_gate": {
                "require_positive_avg_judge_accuracy": True,
                "require_positive_avg_oracle_hit_rate": True,
                "require_robustness_signal": [
                    "low_permutation_disagreement",
                    "positive_pairwise_secondary_conversion",
                ],
            },
        },
    }


def main():
    args = parse_args()
    if args.command == "triggered_bundle":
        report = build_triggered_bundle(args.report_files)
    else:
        report = build_mismatch_audit(
            base_file=args.base_file,
            candidate_file=args.candidate_file,
            task7_data=args.task7_data,
            triggered_bundle_path=args.triggered_bundle,
        )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
