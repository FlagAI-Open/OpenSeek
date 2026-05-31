"""
main1 task2：在 ``infer_examples_main1_task2_spacy_v3`` 的零样本 spaCy 结构上增加两套提示词优化（**prompt1** / **prompt2**），
分别写出：

- ``openseek-{task}-examples-main1-compare_spacy_v3_prompt1_zeroshot.jsonl``
- ``openseek-{task}-examples-main1-compare_spacy_v3_prompt2_zeroshot.jsonl``

用法::

    python src/infer_examples_main1_task2_spacy_v3_prompt_variants.py
    python src/infer_examples_main1_task2_spacy_v3_prompt_variants.py --variant prompt1
    python src/infer_examples_main1_task2_spacy_v3_prompt_variants.py --examples_limit 20

依赖与 ``infer_examples_main1_task2_spacy_v3`` 相同。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from infer_examples_main1_task2_spacy_v3 import (  # noqa: E402
    _align_task2_official_definition,
    _compose_task2_guidelines,
    _resolve_output_dir,
    _task_prompt_shell_zero_shot,
    register_task_prompt,
    run_task,
)

# ---------------------------------------------------------------------------
# 在 v3 的 task-specific + spaCy 表之前插入的附加指导（与 v3 规则一致、仅强化流程）
# ---------------------------------------------------------------------------

_EXTRA_V3_PROMPT1 = (
    "### Reasoning protocol (v3 prompt1)\n"
    "Before writing `<label>`, do this in plain text **outside** the tags:\n"
    "1. **Candidates:** from the spaCy table, list the token **texts** you might count for **this** branch (nouns vs verbs).\n"
    "2. **Filter:** remove determiners, pronouns, adjectives-as-not-nouns, and (for verb items) bare auxiliaries / dummy *do* / "
    "modals and light copulas counted per the Task-specific rules.\n"
    "3. **Gerund / participle (`VBG` / `VBN`):** decide using whether this instance asks for **verbs** or **nouns**; when still ambiguous, "
    "prefer the reading consistent with typical OpenSeek caption-style counts.\n\n"
)

_EXTRA_V3_PROMPT2 = (
    "### Output discipline (v3 prompt2)\n"
    "- The integer inside `<label>` must equal your **final** kept list after **all** exclusions—no rounding or ranges.\n"
    "- For **verb** items, coordinated predicates (*... and ...*) often contribute **separate** counted verbs when each heads its own event; "
    "still exclude pure auxiliaries attached only for tense/voice.\n"
    "- For **noun** items, coordinated heads (*cats and dogs*) usually contribute **separate** nouns when both are lexical heads.\n"
    "- Immediately before `<label>`, **re-count** your kept list once backward to catch off-by-one slips.\n\n"
)


def _prompt_openseek_2_spacy_v3_prompt1(task_description: str, text2annotate: str) -> str:
    guidelines = _EXTRA_V3_PROMPT1 + _compose_task2_guidelines(text2annotate)
    definition_for_prompt = _align_task2_official_definition(task_description, text2annotate)
    return _task_prompt_shell_zero_shot(guidelines, definition_for_prompt, text2annotate)


def _prompt_openseek_2_spacy_v3_prompt2(task_description: str, text2annotate: str) -> str:
    guidelines = _EXTRA_V3_PROMPT2 + _compose_task2_guidelines(text2annotate)
    definition_for_prompt = _align_task2_official_definition(task_description, text2annotate)
    return _task_prompt_shell_zero_shot(guidelines, definition_for_prompt, text2annotate)


_PROMPT_BUILDERS: dict[str, Callable[[str, str], str]] = {
    "prompt1": _prompt_openseek_2_spacy_v3_prompt1,
    "prompt2": _prompt_openseek_2_spacy_v3_prompt2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="main1 task2：v3 零样本 + spaCy 上的 prompt1 / prompt2 变体（infer_examples_main1_task2_spacy_v3_prompt_variants）。"
    )
    parser.add_argument("--task_start", type=int, default=2)
    parser.add_argument("--task_end", type=int, default=2)
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=0,
        help="每任务最多推理条数；<=0 为全部。",
    )
    parser.add_argument("--output_dir", type=str, default="examples_main1")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("all", "prompt1", "prompt2"),
        default="all",
        help="prompt 变体：all 依次跑 prompt1 与 prompt2；否则只跑指定变体。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_start = max(1, args.task_start)
    task_end = min(8, args.task_end)
    if task_start > task_end:
        raise ValueError(f"task_start({task_start}) > task_end({task_end})")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    keys = ["prompt1", "prompt2"] if args.variant == "all" else [args.variant]

    for key in keys:
        builder = _PROMPT_BUILDERS[key]
        register_task_prompt(2, builder)
        slug = f"spacy_v3_{key}_zeroshot"
        print(f"[task2] 已注册 {slug}（在 spacy v3 结构上的附加指导）。")

        summary: list[dict] = []
        for task_id in range(task_start, task_end + 1):
            summary.append(
                run_task(
                    task_id=task_id,
                    output_dir=output_dir,
                    examples_limit=args.examples_limit,
                    retries=args.retries,
                    retry_wait_seconds=args.retry_wait_seconds,
                    resume=args.resume,
                    output_compare_slug=slug,
                )
            )

        summary_file = output_dir / f"summary_main1_task2_{slug}.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
