from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .flagscale_adapter import describe_runtime
from .official_data import load_official_tasks
from .official_pipeline import OfficialAnnotationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official OpenSeek Track 3 datasets")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", default="data/official", help="Directory containing openseek-*.json files")
    parser.add_argument("--output", default="outputs/official_submission.json", help="Submission JSON output")
    parser.add_argument("--diagnostics", default="outputs/official_diagnostics.json", help="Diagnostics JSON output")
    parser.add_argument("--official-jsonl-dir", default="outputs/official_jsonl", help="Per-task JSONL output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock provider for pipeline validation")
    parser.add_argument("--deterministic-postprocess", action="store_true", help="Use exact programmatic solvers for tasks 1, 3, and 4")
    parser.add_argument("--retrieval-baseline", action="store_true", help="Use nearest official example output instead of model inference")
    parser.add_argument("--task-ids", default="", help="Comma-separated task ids to run, for example openseek-2,openseek-5")
    parser.add_argument("--max-samples-per-task", type=int, default=0, help="Limit test samples per task for local smoke tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.mock:
        config.model.provider = "mock"
    runtime = describe_runtime(config)
    print(
        "Runtime: "
        f"framework={runtime['framework']}, model={runtime['model_name']}, "
        f"flagscale_available={runtime['flagscale_available']}"
    )
    if config.model.model_name != "Qwen3-4B":
        raise ValueError("Track 3 FAQ requires Qwen3-4B exactly. Other models are not allowed.")

    tasks = load_official_tasks(args.data_dir)
    if args.task_ids:
        wanted = {item.strip() for item in args.task_ids.split(",") if item.strip()}
        tasks = [task for task in tasks if task.task_id in wanted]
    if args.max_samples_per_task > 0:
        tasks = [replace(task, test_samples=task.test_samples[: args.max_samples_per_task]) for task in tasks]
    if not tasks:
        raise ValueError("No tasks selected. Check --data-dir and --task-ids.")
    runner = OfficialAnnotationPipeline(
        config,
        deterministic_postprocess=args.deterministic_postprocess,
        retrieval_baseline=args.retrieval_baseline,
    )
    submission: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for task in tasks:
        task_submission, task_diagnostics = runner.annotate_task(task)
        submission.append(task_submission)
        diagnostics.append(task_diagnostics)
        jsonl_dir = Path(args.official_jsonl_dir)
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        task_number = task.task_id.split("-")[-1]
        jsonl_path = jsonl_dir / f"openseek-{task_number}-v1.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as file:
            for record in task_submission["records"]:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    diag = Path(args.diagnostics)
    diag.parent.mkdir(parents=True, exist_ok=True)
    diag.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote submission to {output}")
    print(f"Wrote diagnostics to {diag}")
    print(f"Wrote per-task JSONL files to {args.official_jsonl_dir}")


if __name__ == "__main__":
    main()
