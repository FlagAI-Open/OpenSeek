#!/usr/bin/env python3
"""
Task7 子集评测（供 autoresearch_prompt 循环调用）。

输出 JSONL 字段与 compare 一致，并额外提供 ``correct`` / ``ground_truth`` / ``prediction``
供 failure_analysis 使用。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from method_hyb import annotate_nvidia, count_answer, select_examples_hybrid
from prompts_autoresearch import _REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK7_FILE = "openseek-7_jeopardy_answer_generation_all.json"
TASK7_DESCRIPTION = (
    "You will be given a trivia clue, and the category it belongs to. "
    "You should answer with the best answer that belongs to the category "
    "and is described by the clue. For simplicity, answers should be in all lower cased letters."
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _is_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task7 autoresearch 子集评测")
    p.add_argument("--prompt-version", type=int, required=True)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--output-tag", type=str, default="")
    p.add_argument("--n-icl", type=int, default=3, help="hybrid 检索 few-shot 条数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = REPO_ROOT / "autoresearch" / "task7"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    out_path = args.output or out_dir / f"results_v{args.prompt_version}{tag}.jsonl"

    if 7 not in _REGISTRY or args.prompt_version not in _REGISTRY[7]:
        raise SystemExit(f"prompt version v{args.prompt_version} not in prompts_autoresearch")

    tpl = _REGISTRY[7][args.prompt_version]

    with (REPO_ROOT / "data" / TASK7_FILE).open(encoding="utf-8") as f:
        bundle = json.load(f)
    examples = bundle.get("examples", [])
    rng = random.Random(args.seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    subset = [examples[i] for i in indices[: args.max_samples]]

    cleaned = [{"input": ex["input"], "output": ex["output"]} for ex in examples]
    total = matched = none_n = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in subset:
            total += 1
            inp = ex["input"]
            gold = ex["output"][0] if isinstance(ex.get("output"), list) else str(ex.get("output", ""))
            selected = select_examples_hybrid(inp, cleaned, k=args.n_icl, task_id=7)
            ex_str = "\n\n".join(
                f"Input: {s['input']}\nOutput: {s['output'][0] if isinstance(s['output'], list) else s['output']}"
                for s in selected
            )
            prompt = tpl.render(TASK7_DESCRIPTION, ex_str, inp)
            raw = annotate_nvidia(prompt)
            pred = count_answer(raw) if raw else None
            pred_s = "" if pred is None else str(pred).strip()
            if not pred_s:
                none_n += 1
            ok = _is_match(pred_s, gold)
            if ok:
                matched += 1
            row = {
                "example_id": ex.get("id", ""),
                "input": inp,
                "expected_output": gold,
                "ground_truth": gold,
                "model_output": pred_s,
                "prediction": pred_s,
                "is_match": ok,
                "correct": ok,
                "prompt_version": args.prompt_version,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc = matched / total if total else 0.0
    nr = none_n / total if total else 0.0
    print(f"准确率: {acc:.4f} ({matched}/{total})")
    print(f"解析失败: {none_n}/{total} ({nr:.4f})")
    print(f"结果文件 → {out_path}")


if __name__ == "__main__":
    main()
