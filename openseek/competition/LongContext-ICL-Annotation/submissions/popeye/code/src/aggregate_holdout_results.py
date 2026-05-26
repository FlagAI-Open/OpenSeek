import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


LOCAL_SOLVER_TASKS = {1, 3, 4}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout_dir",
        type=str,
        default="outputs/holdout_eval",
        help="Directory containing holdout evaluation JSON files.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing task dataset JSON files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the aggregated summary as JSON.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def infer_variant_from_filename(stem: str) -> str:
    lowered = stem.lower()
    if "family_lexical" in lowered:
        return "family_lexical"
    if "structured" in lowered:
        return "structured"
    if "hybrid" in lowered:
        return "hybrid"
    if "lexical" in lowered:
        return "lexical"
    if "chatlong" in lowered:
        return "chatlong"
    if "chatvote" in lowered:
        return "chatvote"
    if "shortchat" in lowered:
        return "shortchat"
    if re.search(r"(?:^|_)chat(?:_|$)", lowered):
        return "chat"
    if "completiondefault" in lowered:
        return "completiondefault"
    if "completion" in lowered:
        return "completion"
    if "guided" in lowered:
        return "guided"
    if "retrieval" in lowered:
        return "retrieval"
    if "ctx30k" in lowered:
        return "ctx30k"
    if "ctx16k" in lowered:
        return "ctx16k"
    if "baseline" in lowered:
        return "baseline"
    return "unlabeled"


def load_task_catalog(data_dir: Path) -> dict[int, dict]:
    catalog = {}
    for path in sorted(data_dir.glob("openseek-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_match = re.search(r"openseek-(\d+)", path.name)
        if not task_match:
            continue
        task_id = int(task_match.group(1))
        catalog[task_id] = {
            "task_id": task_id,
            "task_name": payload.get("task_name"),
            "examples": len(payload.get("examples", [])),
            "test_samples": len(payload.get("test_samples", [])),
            "local_solver": task_id in LOCAL_SOLVER_TASKS,
        }
    return catalog


def collect_records(holdout_dir: Path) -> tuple[list[dict], dict[tuple[int, str], list[dict]]]:
    records = []
    pairwise = defaultdict(list)
    for path in sorted(holdout_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            continue

        grouped_by_task = defaultdict(list)
        for row in results:
            if not isinstance(row, dict) or "task_id" not in row or "avg_score" not in row:
                continue
            task_id = int(row["task_id"])
            variant = row.get("profile") or infer_variant_from_filename(path.stem)
            record = {
                "file": path.name,
                "task_id": task_id,
                "variant": variant,
                "avg_score": float(row["avg_score"]),
                "sample_count": int(row.get("sample_count", 0)),
                "example_pool_limit": row.get("example_pool_limit"),
                "mismatch_count": int(row.get("mismatch_count", 0)),
            }
            records.append(record)
            grouped_by_task[task_id].append(record)

        for task_id, task_records in grouped_by_task.items():
            baseline_records = [item for item in task_records if item["variant"] == "baseline"]
            if not baseline_records:
                continue
            baseline_score = baseline_records[0]["avg_score"]
            for record in task_records:
                if record["variant"] == "baseline":
                    continue
                pairwise[(task_id, record["variant"])].append(
                    {
                        "file": path.name,
                        "baseline_score": baseline_score,
                        "candidate_score": record["avg_score"],
                        "delta": record["avg_score"] - baseline_score,
                    }
                )
    return records, pairwise


def summarize_records(records: list[dict]) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task_id"], record["variant"])].append(record)

    summary = defaultdict(list)
    for (task_id, variant), items in sorted(grouped.items()):
        scores = [item["avg_score"] for item in items]
        summary[task_id].append(
            {
                "variant": variant,
                "num_runs": len(items),
                "mean_avg_score": round(mean(scores), 4),
                "std_avg_score": round(stdev(scores), 4),
                "min_avg_score": round(min(scores), 4),
                "max_avg_score": round(max(scores), 4),
                "sample_counts": sorted({item["sample_count"] for item in items}),
                "example_pool_limits": sorted(
                    {item["example_pool_limit"] for item in items if item["example_pool_limit"] is not None}
                ),
                "files": [item["file"] for item in items],
            }
        )
        summary[task_id].sort(key=lambda row: (-row["mean_avg_score"], row["variant"]))
    return dict(summary)


def summarize_pairwise(pairwise: dict[tuple[int, str], list[dict]]) -> dict[int, list[dict]]:
    summary = defaultdict(list)
    for (task_id, variant), items in sorted(pairwise.items()):
        deltas = [item["delta"] for item in items]
        summary[task_id].append(
            {
                "variant": variant,
                "num_comparisons": len(items),
                "mean_delta_vs_baseline": round(mean(deltas), 4),
                "std_delta_vs_baseline": round(stdev(deltas), 4),
                "min_delta_vs_baseline": round(min(deltas), 4),
                "max_delta_vs_baseline": round(max(deltas), 4),
                "files": [item["file"] for item in items],
            }
        )
        summary[task_id].sort(key=lambda row: (-row["mean_delta_vs_baseline"], row["variant"]))
    return dict(summary)


def main():
    args = parse_args()
    holdout_dir = Path(args.holdout_dir)
    data_dir = Path(args.data_dir)

    task_catalog = load_task_catalog(data_dir)
    records, pairwise = collect_records(holdout_dir)

    summary = {
        "task_catalog": task_catalog,
        "record_count": len(records),
        "task_variant_summary": summarize_records(records),
        "pairwise_delta_vs_baseline": summarize_pairwise(pairwise),
    }

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
