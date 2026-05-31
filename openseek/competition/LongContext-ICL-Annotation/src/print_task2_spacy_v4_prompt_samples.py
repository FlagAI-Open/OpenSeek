"""打印 task2 spacy v4 完整 prompt（零样本、仅 spaCy 表统计）。

用法::

    python src/print_task2_spacy_v4_prompt_samples.py
    python src/print_task2_spacy_v4_prompt_samples.py --count 10
    python src/print_task2_spacy_v4_prompt_samples.py --indices 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from infer_examples_main1_task2_spacy_v4 import (  # noqa: E402
    REPO_ROOT,
    TASK_DATA_FILES,
    _prompt_openseek_2_spacy_v4,
    build_prompt,
    register_task_prompt,
)


def _parse_indices(spec: str | None) -> list[int] | None:
    if not spec or not str(spec).strip():
        return None
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(description="打印 task2 spacy v4 完整构造后的 prompt")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="打印全局 examples 前 N 条（默认 10）；指定 --indices 时忽略",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="逗号分隔下标，如 `0,2,5`；指定后忽略 --count",
    )
    args = parser.parse_args()

    register_task_prompt(2, _prompt_openseek_2_spacy_v4)

    task_file = REPO_ROOT / "data" / TASK_DATA_FILES[2]
    with task_file.open(encoding="utf-8") as f:
        task_dict = json.load(f)
    task_description = task_dict["Definition"][0]
    examples = task_dict["examples"]

    picked = _parse_indices(args.indices)
    if picked is None:
        n = max(0, args.count)
        picked = list(range(min(n, len(examples))))

    for idx in picked:
        if idx < 0 or idx >= len(examples):
            raise SystemExit(f"index {idx} out of range [0, {len(examples)})")
        ex = examples[idx]
        prompt = build_prompt(task_description, ex["input"], task_id=2)
        bar = "=" * 80
        print(bar)
        print(f"global_index={idx}  example_id={ex.get('id')}")
        print(bar)
        print(prompt)
        print()


if __name__ == "__main__":
    main()
