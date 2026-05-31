"""
打印若干条 task2 + spaCy 与推理脚本一致的「完整用户 prompt」（含 ICL）。

默认与推理脚本相同：``select_examples(..., hybrid=True, ...)``。

可选 ``--literal_icl_first_k``：不做检索，固定拼接示例池中前 K 条（便于离线核对正文结构）。

用法::

    python src/print_task2_spacy_prompt_samples.py --count 2

    python src/print_task2_spacy_prompt_samples.py --indices 0,5 --literal_icl_first_k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from infer_examples_main1_task2_spacy import (
    REPO_ROOT,
    TASK_DATA_FILES,
    _prompt_openseek_2_spacy,
    build_prompt,
    register_task_prompt,
)
from method_hyb import select_examples


def _normalize_output(output_value: object) -> str:
    if isinstance(output_value, list) and output_value:
        return str(output_value[0]).strip()
    return str(output_value).strip()


def _literal_icl_first_k(pool: list[dict], k: int) -> str:
    """固定取池中前 K 条，格式与无 explanation 时的 hybrid 输出一致。"""
    k = max(0, k)
    lines: list[str] = []
    for ex in pool[:k]:
        inp = ex["input"]
        out = _normalize_output(ex["output"])
        lines.append(f"# {inp} <label> {out} </label>\n")
    return "".join(lines)


def _parse_indices(spec: str | None) -> list[int] | None:
    if not spec or not str(spec).strip():
        return None
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(description="打印 task2 v2+spaCy 完整 prompt 样本")
    parser.add_argument("--count", type=int, default=2, help="从前 N 条 examples 打印（默认 2）")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="逗号分隔的全局下标，如 `0,5`；指定时忽略 --count",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="与 infer 脚本一致，传给 select_examples（仅 hybrid 模式）",
    )
    parser.add_argument(
        "--literal_icl_first_k",
        type=int,
        default=0,
        help=">0 时不调用 hybrid 检索，改用示例池前 K 条拼接 ICL（默认 0 表示用 hybrid）",
    )
    args = parser.parse_args()

    register_task_prompt(2, _prompt_openseek_2_spacy)

    task_file = REPO_ROOT / "data" / TASK_DATA_FILES[2]
    with task_file.open(encoding="utf-8") as f:
        task_dict = json.load(f)

    task_description = task_dict["Definition"][0]
    icl_examples = task_dict["examples"][:100]
    all_examples = task_dict["examples"]

    picked: list[tuple[int, dict]]
    explicit = _parse_indices(args.indices)
    if explicit is not None:
        picked = []
        for i in explicit:
            if i < 0 or i >= len(all_examples):
                raise SystemExit(f"index {i} out of range [0, {len(all_examples)})")
            picked.append((i, all_examples[i]))
    else:
        n = max(1, args.count)
        picked = list(enumerate(all_examples[:n]))

    for global_idx, example in picked:
        input_text = example["input"]
        example_id = example.get("id", "")
        prompt = build_prompt(task_description, input_text, task_id=2)
        if args.literal_icl_first_k > 0:
            examples_str = _literal_icl_first_k(icl_examples, args.literal_icl_first_k)
        else:
            examples_str = select_examples(
                icl_examples,
                task_description,
                input_text,
                tokenizer_path=args.tokenizer_path,
                hybrid=True,
                top_k=3,
                use_explanation=True,
                use_bm25_semantic_rerank=True,
            )
        full_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n\n")
        bar = "=" * 80
        print(bar)
        print(f"global_index={global_idx}  example_id={example_id}")
        print(bar)
        print(full_prompt)
        print()


if __name__ == "__main__":
    main()
