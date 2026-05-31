#!/usr/bin/env python3
"""
Task6-v2 原版 + 两个投票变体的一键推理。

入口说明：
  python scripts/run_task6_v2_vote_infer.py          # 推荐
  python scripts/run_task6_v2_vote_infer.sh         # 同上（.sh 为 Python 转发器，兼容 IDE 误 Run）
  bash scripts/run_task6_v2_vote_infer.bash         # 纯 Bash 版（与下方环境变量一致）

环境变量：DATA_FILE, LIMIT, OUT_BASE, RESUME, THINKING, RUN_V2, ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in ("1", "true", "yes", "on")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    root = _repo_root()
    os.chdir(root)

    data_file = _env_str("DATA_FILE", "openseek-6_mnli_same_genre_classification.json")
    limit = _env_int("LIMIT", 0)
    out_base = _env_str("OUT_BASE", "outputs/task6_vote_fusion")
    out_v2 = _env_str("OUT_V2", f"{out_base}/v2_baseline")
    out_pair = _env_str("OUT_PAIR_CROSS", f"{out_base}/pair_cross")
    out_joint = _env_str("OUT_TASKDEF_JOINT", f"{out_base}/taskdef_joint")
    retries = _env_int("RETRIES", 3)
    retry_wait = _env_str("RETRY_WAIT_SECONDS", "2.0")
    resume = _truthy("RESUME", "0")
    thinking = _env_str("THINKING", "off")
    cross_max = _env_int("CROSS_CONTEXT_MAX_CHARS", 900)
    taskdef_max = _env_int("TASKDEF_MAX_CHARS", 1200)
    batch_size = _env_int("BATCH_SIZE", 16)
    flip_gate = _env_str("FLIP_GATE", "not_fiction")
    skip_v3 = _truthy("SKIP_V3", "0")

    py = _env_str("PYTHON", sys.executable)

    def build(script: str, out_dir: str, extra: list[str] | None = None) -> list[str]:
        cmd = [py, str(root / "src" / script), "--data_file", data_file]
        cmd += [
            "--limit",
            str(limit),
            "--output_dir",
            out_dir,
            "--retries",
            str(retries),
            "--retry_wait_seconds",
            retry_wait,
            "--thinking",
            thinking,
            "--batch_size",
            str(batch_size),
            "--flip_gate",
            flip_gate,
        ]
        if resume:
            cmd.append("--resume")
        if skip_v3:
            cmd.append("--skip_v3")
        if extra:
            cmd.extend(extra)
        return cmd

    print("==========================================")
    print(f"[Task6 vote fusion infer] REPO_ROOT={root}")
    print(
        f"  DATA_FILE={data_file}  LIMIT={limit}  RESUME={resume}  THINKING={thinking} "
        f"BATCH_SIZE={batch_size}  FLIP_GATE={flip_gate}  SKIP_V3={skip_v3}"
    )
    print("==========================================")

    runs: list[tuple[str, list[str]]] = []
    if _truthy("RUN_V2", "1"):
        runs.append(("infer_task6_v2.py", build("infer_task6_v2.py", out_v2)))
    if _truthy("RUN_PAIR_CROSS", "1"):
        runs.append(
            (
                "infer_task6_v2_vote_pair_cross.py",
                build(
                    "infer_task6_v2_vote_pair_cross.py",
                    out_pair,
                    ["--cross_context_max_chars", str(cross_max)],
                ),
            )
        )
    if _truthy("RUN_TASKDEF_JOINT", "1"):
        runs.append(
            (
                "infer_task6_v2_vote_taskdef_joint.py",
                build(
                    "infer_task6_v2_vote_taskdef_joint.py",
                    out_joint,
                    ["--taskdef_max_chars", str(taskdef_max)],
                ),
            )
        )

    for i, (label, cmd) in enumerate(runs, start=1):
        print(f">>> [{i}/{len(runs)}] {label}")
        print(" ", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("==========================================")
    print("[全部完成]")
    print("==========================================")


if __name__ == "__main__":
    main()
