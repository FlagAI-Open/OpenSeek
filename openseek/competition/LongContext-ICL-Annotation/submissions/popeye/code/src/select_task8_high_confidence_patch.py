import argparse
import json
from pathlib import Path

from task8_semantics import score_task8_proxy_v2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, required=True)
    parser.add_argument("--candidate_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task8_data", type=str, default="data/openseek-8_kernel_generation.json")
    parser.add_argument("--max_replacements", type=int, default=6)
    parser.add_argument("--min_selected", type=int, default=2)
    parser.add_argument("--min_score_improvement", type=float, default=0.12)
    parser.add_argument("--max_warning_count", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    base_rows = load_jsonl(Path(args.base_file))
    candidate_rows = load_jsonl(Path(args.candidate_file))
    task8 = json.loads(Path(args.task8_data).read_text(encoding="utf-8"))
    test_map = {sample["id"]: sample for sample in task8["test_samples"]}

    base_by_id = {row["test_sample_id"]: row for row in base_rows}
    candidate_by_id = {row["test_sample_id"]: row for row in candidate_rows}

    evaluated_rows = []
    eligible_rows = []
    for sample_id, candidate_row in candidate_by_id.items():
        base_row = base_by_id.get(sample_id)
        if base_row is None:
            continue

        base_prediction = base_row.get("prediction")
        candidate_prediction = candidate_row.get("prediction")
        changed = candidate_prediction != base_prediction
        if not changed:
            continue

        sample_input = test_map[sample_id]["input"]
        base_score = score_task8_proxy_v2(sample_input, base_prediction)
        candidate_score = score_task8_proxy_v2(sample_input, candidate_prediction)
        improvement = round(candidate_score["score"] - base_score["score"], 4)

        reject_reasons = []
        if candidate_score["audit"]["blocker_count"] > 0:
            reject_reasons.append("has_blocker")
        if candidate_score["audit"]["warning_count"] > args.max_warning_count:
            reject_reasons.append("too_many_warnings")
        if improvement < args.min_score_improvement:
            reject_reasons.append("improvement_below_threshold")
        if candidate_score["hard_gate_score"] < base_score["hard_gate_score"]:
            reject_reasons.append("hard_gate_regression")
        if candidate_score["family_semantics_score"] < base_score["family_semantics_score"]:
            reject_reasons.append("family_regression")
        if candidate_score["structural_similarity_score"] + 0.02 < base_score["structural_similarity_score"]:
            reject_reasons.append("structural_regression")

        row_report = {
            "test_sample_id": sample_id,
            "input_family": candidate_score["family"]["input_family"],
            "base_code_family": base_score["family"]["code_family"],
            "candidate_code_family": candidate_score["family"]["code_family"],
            "changed": True,
            "eligible": not reject_reasons,
            "reasons": reject_reasons,
            "base_score": base_score["score"],
            "candidate_score": candidate_score["score"],
            "improvement": improvement,
            "base_blocker_count": base_score["audit"]["blocker_count"],
            "candidate_blocker_count": candidate_score["audit"]["blocker_count"],
            "base_warning_count": base_score["audit"]["warning_count"],
            "candidate_warning_count": candidate_score["audit"]["warning_count"],
            "base_reason": base_score["reason"],
            "candidate_reason": candidate_score["reason"],
            "base_audit": base_score["audit"],
            "candidate_audit": candidate_score["audit"],
            "base_family": base_score["family"],
            "candidate_family": candidate_score["family"],
            "base_structural": base_score["structural"],
            "candidate_structural": candidate_score["structural"],
        }
        evaluated_rows.append(row_report)
        if not reject_reasons:
            eligible_rows.append(row_report)

    eligible_rows.sort(
        key=lambda row: (
            row["improvement"],
            row["candidate_score"],
            row["candidate_family"]["score"],
            row["candidate_structural"]["score"],
        ),
        reverse=True,
    )
    selected_rows = eligible_rows[: args.max_replacements]
    selected_ids = {row["test_sample_id"] for row in selected_rows}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = None
    if len(selected_rows) >= args.min_selected:
        merged_rows = []
        for row in base_rows:
            sample_id = row["test_sample_id"]
            if sample_id in selected_ids:
                merged_rows.append(candidate_by_id[sample_id])
            else:
                merged_rows.append(row)
        output_file = output_dir / Path(args.base_file).name
        write_jsonl(output_file, merged_rows)

    summary = {
        "base_file": str(args.base_file),
        "candidate_file": str(args.candidate_file),
        "output_dir": str(output_dir),
        "output_file": str(output_file) if output_file else None,
        "min_selected": args.min_selected,
        "max_replacements": args.max_replacements,
        "min_score_improvement": args.min_score_improvement,
        "max_warning_count": args.max_warning_count,
        "changed_count": len(evaluated_rows),
        "eligible_count": len(eligible_rows),
        "selected_count": len(selected_rows),
        "candidate_generated": output_file is not None,
        "selected_ids": sorted(selected_ids),
        "rows": evaluated_rows,
    }

    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
