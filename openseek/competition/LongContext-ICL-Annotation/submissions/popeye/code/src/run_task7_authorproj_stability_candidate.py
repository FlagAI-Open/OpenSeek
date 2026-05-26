import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WORK_LOGS_DIR = OUTPUTS_DIR / "work_logs"

DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "task7_author_projection_stability_live_candidate"
DEFAULT_REPORT_STEM = "task7_authorproj_stability_live_candidate_run_2026-04-09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--profile", type=str, default="frontier_task6_task7")
    parser.add_argument("--completion_seed", type=str, default="2028")
    parser.add_argument("--completion_url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--request_timeout", type=int, default=None)
    parser.add_argument("--sample_limit", type=int, default=None)
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "OPENSEEK_PROFILE": args.profile,
            "OPENSEEK_TASK7_RERANK": "1",
            "OPENSEEK_TASK7_RERANK_RETRIEVAL_MODE": "none",
            "OPENSEEK_TASK7_RERANK_CANDIDATES": "8",
            "OPENSEEK_TASK7_RERANK_MAX_JUDGE": "8",
            "OPENSEEK_TASK7_RERANK_TEMPERATURE": "0.9",
            "OPENSEEK_TASK7_RERANK_TOP_P": "0.95",
            "OPENSEEK_TASK7_RERANK_HINTS": "0",
            "OPENSEEK_TASK7_RERANK_HINT_MODE": "off",
            "OPENSEEK_TASK7_RERANK_SECONDARY_PROFILE": "long_context",
            "OPENSEEK_TASK7_RERANK_SECONDARY_RETRIEVAL_MODE": "none",
            "OPENSEEK_TASK7_RERANK_SECONDARY_CANDIDATES": "4",
            "OPENSEEK_TASK7_RERANK_SECONDARY_TEMPERATURE": "0.9",
            "OPENSEEK_TASK7_RERANK_SECONDARY_TOP_P": "0.95",
            "OPENSEEK_TASK7_RERANK_SECONDARY_HINTS": "0",
            "OPENSEEK_TASK7_RERANK_SECONDARY_HINT_MODE": "off",
            "OPENSEEK_TASK7_RERANK_SECONDARY_TYPED_ROUTE": "off",
            "OPENSEEK_TASK7_RERANK_SECONDARY_MERGE_MODE": "append_unique",
            "OPENSEEK_TASK7_RERANK_SECONDARY_GATE": "unique_only",
            "OPENSEEK_TASK7_RERANK_SECONDARY_MIN_UNIQUE": "7",
            "OPENSEEK_TASK7_RERANK_SECONDARY_MIN_ENTROPY": "1.3",
            "OPENSEEK_TASK7_RERANK_APPEND_UNIQUE_SECONDARY_JUDGE_SLOTS": "3",
            "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_PROJECTION": "1",
            "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_ROUNDS": "3",
            "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_N": "6",
            "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_MAX_TOKENS": "32",
            "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_TEMPERATURE": "0.9",
            "OPENSEEK_TASK7_AUTHOR_DIRECT_FACT_TOP_P": "0.95",
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT": "author_primary_anchor_top1",
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_DECISION_MODE": "anchor1_stability_gate",
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS": "3",
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE": "strong_only",
            "OPENSEEK_COMPLETION_SEED": args.completion_seed,
        }
    )
    if args.completion_url:
        env["OPENSEEK_VLLM_URL"] = args.completion_url
    if args.model_name:
        env["OPENSEEK_MODEL_NAME"] = args.model_name
    if args.request_timeout is not None:
        env["OPENSEEK_REQUEST_TIMEOUT"] = str(args.request_timeout)
    return env


def resolve_latest_task7_output(output_dir: Path) -> Path | None:
    matches = []
    for path in output_dir.glob("openseek-7-v*.jsonl"):
        match = re.search(r"-v(\d+)\.jsonl$", path.name)
        if match is None:
            continue
        matches.append((int(match.group(1)), path))
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    WORK_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SRC_DIR / "main.py"),
        "--task_id",
        "7",
        "--log_path_prefix",
        str(output_dir),
        "--profile",
        args.profile,
    ]
    if args.sample_limit is not None:
        command.extend(["--sample_limit", str(args.sample_limit)])

    env = build_env(args)
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)

    output_path = output_dir / "openseek-7-v1.jsonl"
    latest_output_path = resolve_latest_task7_output(output_dir)
    if latest_output_path and latest_output_path != output_path:
        output_path.write_text(latest_output_path.read_text(encoding="utf-8"), encoding="utf-8")
    report = {
        "generated_on": "2026-04-09",
        "command": " ".join(shlex.quote(part) for part in command),
        "cwd": str(PROJECT_DIR),
        "output_path": str(output_path),
        "exists": output_path.exists(),
        "line_count": sum(1 for _ in output_path.open("r", encoding="utf-8")) if output_path.exists() else 0,
        "latest_generated_output_path": str(latest_output_path) if latest_output_path else None,
        "env": {
            "OPENSEEK_PROFILE": env["OPENSEEK_PROFILE"],
            "OPENSEEK_COMPLETION_SEED": env["OPENSEEK_COMPLETION_SEED"],
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT": env["OPENSEEK_TASK7_AUTHOR_PROJECTION_JUDGE_LAYOUT"],
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_DECISION_MODE": env["OPENSEEK_TASK7_AUTHOR_PROJECTION_DECISION_MODE"],
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS": env["OPENSEEK_TASK7_AUTHOR_PROJECTION_STABILITY_REPEATS"],
            "OPENSEEK_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE": env["OPENSEEK_TASK7_AUTHOR_PROJECTION_PAIRWISE_GATE"],
        },
    }
    json_path = WORK_LOGS_DIR / f"{DEFAULT_REPORT_STEM}.json"
    md_path = WORK_LOGS_DIR / f"{DEFAULT_REPORT_STEM}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Task7 Author Projection Stability Live Candidate Run",
        "",
        "- Generated on: `2026-04-09`",
        f"- Output path: `{report['output_path']}`",
        f"- Output exists: `{report['exists']}`",
        f"- Output line count: `{report['line_count']}`",
        f"- Command: `{report['command']}`",
        "",
        "## Key env",
        "",
    ]
    for key, value in report["env"].items():
        lines.append(f"- `{key}` = `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
