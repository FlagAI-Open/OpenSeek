#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从仓库根目录运行 Task6-v3 示例推理（句子级 AND + 与 postprocess_task6_v2 相同的后处理）。

默认使用 batch_size=16 的线程池并行（与 src 内 argparse 默认一致）；
若命令行已包含 --batch_size，则不再自动插入。

用法：
  python scripts/run_task6_v3_examples_infer.py
  python scripts/run_task6_v3_examples_infer.py --examples_limit 32 --resume
  python scripts/run_task6_v3_examples_infer.py --batch_size 8

等价于在仓库根目录执行：
  python src/infer_examples_compare_task6_v3.py ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    user_args = sys.argv[1:]
    extra: list[str] = []
    if not any(a == "--batch_size" or a.startswith("--batch_size=") for a in user_args):
        extra = ["--batch_size", "16"]
    cmd = [
        sys.executable,
        str(root / "src" / "infer_examples_compare_task6_v3.py"),
        *extra,
        *user_args,
    ]
    print("[run_task6_v3_examples_infer]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
