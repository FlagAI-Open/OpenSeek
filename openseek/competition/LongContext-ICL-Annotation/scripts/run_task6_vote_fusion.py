#!/usr/bin/env python3
"""Task6 三路投票融合入口。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    cmd = [sys.executable, str(root / "src" / "fuse_task6_vote.py"), *sys.argv[1:]]
    print("[run_task6_vote_fusion]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
