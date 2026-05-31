#!/usr/bin/env python3
"""离线对比 post-v3 多种翻转约束。默认读已有 -post-v3.jsonl。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    cmd = [sys.executable, str(root / "src" / "task6_eval_post_v3_constraints.py"), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
