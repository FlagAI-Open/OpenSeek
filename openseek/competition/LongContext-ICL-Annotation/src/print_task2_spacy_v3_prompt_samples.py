"""打印 task2 spacy v3（零样本）两条示例 prompt：一条 verbs、一条 nouns。

用法::

    python src/print_task2_spacy_v3_prompt_samples.py

依赖：与 infer_examples_main1_task2_spacy_v3.py 相同（含 spaCy en_core_web_sm）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from infer_examples_main1_task2_spacy_v3 import (  # noqa: E402
    REPO_ROOT,
    TASK_DATA_FILES,
    _prompt_openseek_2_spacy_v3,
    build_prompt,
    register_task_prompt,
)


def main() -> None:
    register_task_prompt(2, _prompt_openseek_2_spacy_v3)

    task_file = REPO_ROOT / "data" / TASK_DATA_FILES[2]
    with task_file.open(encoding="utf-8") as f:
        task_dict = json.load(f)
    task_description = task_dict["Definition"][0]
    examples = task_dict["examples"]

    verb_ex = examples[0]
    noun_ex = next(e for e in examples if "nouns" in e["input"].lower())

    for label, ex in [("VERB instance", verb_ex), ("NOUN instance", noun_ex)]:
        p = build_prompt(task_description, ex["input"], task_id=2)
        bar = "=" * 80
        print(bar)
        print(f"{label}  example_id={ex.get('id')}")
        print(bar)
        print(p)
        print()


if __name__ == "__main__":
    main()
