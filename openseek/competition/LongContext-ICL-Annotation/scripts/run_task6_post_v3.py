#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 v2 后处理结果中预测为 N 的样本运行 task6 后处理 v3（语篇上下文复核）。

用法：
  python scripts/run_task6_post_v3.py
  python scripts/run_task6_post_v3.py --input examples/xxx.jsonl --output examples/xxx-post-v3.jsonl
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    default_in = (
        "examples/openseek-6-examples-compare-task6-v2-sentence-and-postprocess-v2.jsonl"
    )
    user_args = sys.argv[1:]
    extra: list[str] = []
    if not any(a == "--input" or a.startswith("--input=") for a in user_args):
        extra = ["--input", default_in]
    cmd = [
        sys.executable,
        str(root / "src" / "postprocess_task6_v3.py"),
        *extra,
        *user_args,
    ]
    print("[run_task6_post_v3]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
