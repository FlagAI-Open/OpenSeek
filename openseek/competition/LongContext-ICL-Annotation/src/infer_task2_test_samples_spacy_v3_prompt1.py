"""
Task2（openseek-count_nouns_verbs）test_samples 推理：**spaCy v3 + prompt1** 变体。

与 ``infer_examples_main1_task2_spacy_v3_prompt_variants.py --variant prompt1`` 对齐：
``register_task_prompt(2, _prompt_openseek_2_spacy_v3_prompt1)`` + ``build_prompt(..., task_id=2)``
+ ``annotate_nvidia``（与 v3 test 脚本相同，无 v5 系列兜底）。
默认 **bs=16** 使用 ``ThreadPoolExecutor`` 并行调用 API。

test_samples 无 gold；每行 ``{\"test_sample_id\": \"...\", \"prediction\": \"...\"}``，
``prediction`` 为模型原始返回串（strip 后），与 ``infer_task2_test_samples_spacy_v3.py`` 一致。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from method_hyb import annotate_nvidia as annotate

from infer_examples_main1_task2_spacy_v3 import (
    TASK_DATA_FILES,
    _resolve_output_dir,
    build_prompt,
    register_task_prompt,
)
from infer_examples_main1_task2_spacy_v3_prompt_variants import _prompt_openseek_2_spacy_v3_prompt1

_OUTPUT_JSONL_NAME = "openseek-2-test_samples-task2-spacy-v3-prompt1-zeroshot-predictions.jsonl"
_SUMMARY_JSON_NAME = "summary_task2_test_samples_spacy_v3_prompt1.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task2 test_samples 推理 spaCy v3 prompt1（对齐 infer_examples_main1_task2_spacy_v3_prompt_variants --variant prompt1）。"
        )
    )
    p.add_argument("--samples_limit", type=int, default=0, help="最多推理多少条；<=0 全部 test_samples")
    p.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_wait_seconds", type=float, default=2.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--thinking", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_model_output", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_empty_prediction", type=str, choices=["on", "off"], default="on")
    p.add_argument(
        "--bs",
        type=int,
        default=16,
        help="并行批大小（ThreadPoolExecutor max_workers）；<=0 时视为 1。",
    )
    return p.parse_args()


def _task_json_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / TASK_DATA_FILES[2]


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


def _infer_one_v3_prompt1(
    sample: dict[str, Any],
    task_description: str,
    *,
    task_id: int,
    retries: int,
    retry_wait_seconds: float,
    print_empty_prediction: str,
) -> dict[str, str]:
    sid = str(sample.get("id", "")).strip()
    input_text = str(sample.get("input", ""))
    input_prompt = build_prompt(task_description, input_text, task_id=task_id)

    prediction = ""
    for attempt in range(1, retries + 1):
        try:
            raw = annotate(input_prompt)
            prediction = "" if raw is None else str(raw).strip()
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(f"[推理失败] task2 test_sample id={sid} attempt={attempt}/{retries} err={e}")
            else:
                print(f"[重试] task2 test_sample id={sid} attempt={attempt}/{retries} err={e}")
                time.sleep(retry_wait_seconds)

    if print_empty_prediction == "on" and not prediction:
        preview = input_text[:160].replace("\n", " ")
        print(f"[空预测 task2 spacy v3 prompt1 test_sample] id={sid} preview={preview}")
    return {"test_sample_id": sid, "prediction": prediction}


def run_task2_test_samples_spacy_v3_prompt1(
    output_dir: Path,
    *,
    samples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    print_empty_prediction: str = "on",
    bs: int = 16,
) -> dict:
    task_id = 2
    task_file = _task_json_path()
    with task_file.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    test_samples = list(task_dict.get("test_samples", []))
    if samples_limit > 0:
        test_samples = test_samples[:samples_limit]

    output_file = output_dir / _OUTPUT_JSONL_NAME
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑 task2 spacy v3 prompt1 test_samples] 已完成 {len(done_ids)} 条")

    pending: list[dict[str, Any]] = []
    for sample in test_samples:
        sid = str(sample.get("id", "")).strip()
        if resume and sid in done_ids:
            continue
        pending.append(sample)

    batch_size = max(1, int(bs))
    print(f"[并行推理] bs={batch_size}")

    with output_file.open(mode, encoding="utf-8") as wf:
        for i in tqdm(
            range(0, len(pending), batch_size),
            desc=f"Task2 spacy v3 prompt1 test_samples bs={batch_size}: {task_name}",
        ):
            chunk = pending[i : i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                futures = [
                    executor.submit(
                        _infer_one_v3_prompt1,
                        sample,
                        task_description,
                        task_id=task_id,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                        print_empty_prediction=print_empty_prediction,
                    )
                    for sample in chunk
                ]
                rows = [f.result() for f in futures]
            for row in rows:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(str(row.get("test_sample_id", "")).strip())

    total = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as rf:
            total = sum(1 for line in rf if line.strip())
    print(f"[保存完成 task2 spacy v3 prompt1 test_samples] total={total}, file={output_file}")
    return {
        "task_id": task_id,
        "task_name": task_name,
        "prompt_version": "task2_spacy_v3_prompt1_zeroshot_test_samples",
        "batch_size": batch_size,
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()

    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"

    register_task_prompt(2, _prompt_openseek_2_spacy_v3_prompt1)

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[prompt] Task2 spaCy v3 + prompt1（同 infer_examples_main1_task2_spacy_v3_prompt_variants --variant prompt1）")
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}, {_SUMMARY_JSON_NAME}")

    summary = run_task2_test_samples_spacy_v3_prompt1(
        output_dir=output_dir,
        samples_limit=args.samples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        print_empty_prediction=args.print_empty_prediction,
        bs=args.bs,
    )
    summary_path = output_dir / _SUMMARY_JSON_NAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总] {summary_path}")


if __name__ == "__main__":
    main()
