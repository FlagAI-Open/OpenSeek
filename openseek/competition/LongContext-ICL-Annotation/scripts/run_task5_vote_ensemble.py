"""
Task5 投票四条变体顺序执行（与 run_task5_vote_ensemble.sh 等价）。

勿用 python 运行 .sh 文件；若习惯用 Python 启动，请运行本脚本：

    python scripts/run_task5_vote_ensemble.py
    python scripts/run_task5_vote_ensemble.py --resume
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent

VOTE_SCRIPTS = [
    "src/infer_examples_compare_task5_vote_strip_on_postemoji_on.py",
    "src/infer_examples_compare_task5_vote_strip_on_postemoji_off.py",
    "src/infer_examples_compare_task5_vote_strip_off_postemoji_on.py",
    "src/infer_examples_compare_task5_vote_strip_off_postemoji_off.py",
]


def main() -> None:
    os.chdir(ROOT)
    extra = sys.argv[1:]
    exe = sys.executable
    print(f"[run_task5_vote_ensemble] ROOT={ROOT} PYTHON={exe}")
    for rel in VOTE_SCRIPTS:
        cmd = [exe, rel, *extra]
        print(f"==========> {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    print("[run_task5_vote_ensemble] 全部完成。")


if __name__ == "__main__":
    main()
