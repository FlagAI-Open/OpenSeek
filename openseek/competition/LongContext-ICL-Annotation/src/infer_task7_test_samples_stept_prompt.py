"""
Task7 test_samples 推理（stept_prompt）：与 ``infer_examples_compare_task7_stept_prompt`` 对齐——
同款 [System Instructions] + hybrid ICL；输出每行 ``{"test_sample_id": "...", "prediction": "..."}``。
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from infer_examples_compare_task7_stept_prompt import ensure_task7_stept_prompt_registered
from infer_examples_compare_task7_v3 import (
    TASK7_CANONICAL_DESCRIPTION,
    _extract_output,
    _infer_task7_prediction,
    _normalize_task7_answer,
    _resolve_output_dir,
    _task_json_path,
)
from method_hyb import build_prompt, select_examples_hybrid

_OUTPUT_JSONL_NAME = "openseek-7-test_samples-task7opt-stept-prompt-predictions.jsonl"
_SUMMARY_JSON_NAME = "summary_task7_test_samples_stept_prompt.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task7 test_samples：对齐 infer_examples_compare_task7_stept_prompt（stept_prompt + hybrid ICL）。"
            "无标注时仅写 prediction。"
        )
    )
    p.add_argument("--samples_limit", type=int, default=0, help="<=0 表示全部 test_samples")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_wait_seconds", type=float, default=2.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task7_shot_k", type=int, default=8)
    p.add_argument("--task7_retrieval_pool_size", type=int, default=200)
    p.add_argument("--thinking", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_model_output", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_empty_prediction", type=str, choices=["on", "off"], default="on")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def _load_done_ids(output_file: Path) -> set[str]:
    done: set[str] = set()
    if not output_file.exists():
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("test_sample_id", "") or row.get("example_id", "")).strip()
            if sid:
                done.add(sid)
    return done


def run_task7_test_samples_stept_prompt(
    output_dir: Path,
    *,
    samples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task7_shot_k: int = 8,
    task7_retrieval_pool_size: int = 200,
    print_empty_prediction: str = "on",
    batch_size: int = 8,
) -> dict:
    ensure_task7_stept_prompt_registered()
    task_id = 7

    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = TASK7_CANONICAL_DESCRIPTION
    test_samples = list(task_dict.get("test_samples", []))
    if samples_limit > 0:
        test_samples = test_samples[:samples_limit]

    output_file = output_dir / _OUTPUT_JSONL_NAME
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"
    if done_ids:
        print(f"[断点续跑 stept_prompt test_samples] 已完成 {len(done_ids)} 条")

    cleaned_examples: list[dict] = []
    for ex in task_dict["examples"]:
        cleaned_examples.append(
            {
                "id": str(ex.get("id", "")).strip(),
                "input": str(ex.get("input", "")),
                "output": [_normalize_task7_answer(_extract_output(ex.get("output", "")))],
            }
        )

    pending: list[dict] = []
    for sample in test_samples:
        sid = str(sample.get("id", "")).strip()
        if resume and sid in done_ids:
            continue
        pending.append(sample)

    with output_file.open(mode, encoding="utf-8") as wf:
        for start in tqdm(
            range(0, len(pending), max(1, batch_size)),
            desc=f"Task7 stept_prompt test_samples: {task_name}",
        ):
            batch = pending[start : start + max(1, batch_size)]
            prepared: list[dict] = []
            for sample in batch:
                sid = str(sample.get("id", "")).strip()
                input_text = str(sample.get("input", ""))
                examples_str = select_examples_hybrid(
                    all_examples=cleaned_examples,
                    task_description=task_description,
                    text2annotate=input_text,
                    top_k=max(1, task7_shot_k),
                    rerank_pool_size=max(20, task7_retrieval_pool_size),
                    use_explanation=False,
                    use_bm25_semantic_rerank=True,
                    exclude_example_id=sid,
                )
                base_prompt = build_prompt(task_description, input_text, task_id=task_id)
                input_prompt = base_prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")
                prepared.append(
                    {
                        "test_sample_id": sid,
                        "input_text": input_text,
                        "input_prompt": input_prompt,
                    }
                )

            workers = max(1, batch_size)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                predictions = list(
                    executor.map(
                        lambda item: _infer_task7_prediction(
                            item["input_prompt"],
                            retries=retries,
                            retry_wait_seconds=retry_wait_seconds,
                        ),
                        prepared,
                    )
                )

            for item, prediction in zip(prepared, predictions, strict=True):
                if print_empty_prediction == "on" and not prediction:
                    print(
                        f"[空预测 stept_prompt test_sample] id={item['test_sample_id']} "
                        f"preview={item['input_text'][:160].replace(chr(10), ' ')}"
                    )
                row = {
                    "test_sample_id": item["test_sample_id"],
                    "prediction": prediction,
                }
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(item["test_sample_id"])

    total = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as rf:
            total = sum(1 for line in rf if line.strip())
    print(f"[保存完成 stept_prompt test_samples] total={total}, file={output_file}")
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_stept_system_style_prompt_test_samples",
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_task7_stept_prompt_registered()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")

    print("[prompt] Task7 stept_prompt（对齐 infer_examples_compare_task7_stept_prompt）")
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}, {_SUMMARY_JSON_NAME}")

    summary = run_task7_test_samples_stept_prompt(
        output_dir=output_dir,
        samples_limit=args.samples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task7_shot_k=args.task7_shot_k,
        task7_retrieval_pool_size=args.task7_retrieval_pool_size,
        print_empty_prediction=args.print_empty_prediction,
        batch_size=max(1, args.batch_size),
    )
    summary_path = output_dir / _SUMMARY_JSON_NAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总] {summary_path}")


if __name__ == "__main__":
    main()
