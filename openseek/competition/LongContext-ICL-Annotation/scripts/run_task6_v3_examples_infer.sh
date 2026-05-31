#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容 IDE「对当前文件运行 Python」：扩展名为 .sh 但内容是 Python，会转调 run_task6_v3_examples_infer.py。
"""
from __future__ import annotations

import pathlib
import runpy
import sys


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    target = here / "run_task6_v3_examples_infer.py"
    if not target.is_file():
        print(f"[错误] 缺少 {target}", file=sys.stderr)
        sys.exit(1)
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
