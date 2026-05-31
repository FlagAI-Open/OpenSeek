"""
Task2（openseek-count_nouns_verbs）test_samples 推理：**spaCy v5 零样本**。

与 ``infer_examples_main1_task2_spacy_v5`` 对齐：``_build_prompt_task2_v5(..., variant=\"v5\")``
+ ``_predict_with_fallbacks``（重试、``<label>`` 提醒、全文解析兜底）。
默认 **bs=1**（串行）；``bs>1`` 时使用 ``ThreadPoolExecutor`` 对一批样本并行调用 API。

test_samples 无 gold；每行 ``{\"test_sample_id\": \"...\", \"prediction\": \"...\"}``。
其中 ``prediction`` 为管线归一化后的答案字符串（与 main1 compare JSONL 中 ``model_output`` 形态一致），
可能仅为数字；解析失败则为空串。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from infer_examples_main1_task2_spacy_v3 import TASK_DATA_FILES, _resolve_output_dir
from infer_examples_main1_task2_spacy_v5 import (
    Task2PromptVariant,
    _build_prompt_task2_v5,
    _predict_with_fallbacks,
)

_OUTPUT_JSONL_NAME = "openseek-2-test_samples-task2-spacy-v5-zeroshot-predictions.jsonl"
_SUMMARY_JSON_NAME = "summary_task2_test_samples_spacy_v5.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task2 test_samples 推理 spaCy v5 零样本（对齐 infer_examples_main1_task2_spacy_v5 --prompt_variant v5）。"
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
        default=1,
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


def _infer_one_v5(
    sample: dict[str, Any],
    task_description: str,
    *,
    variant: Task2PromptVariant,
    retries: int,
    retry_wait_seconds: float,
    print_empty_prediction: str,
) -> dict[str, str]:
    sid = str(sample.get("id", "")).strip()
    input_text = str(sample.get("input", ""))
    input_prompt, _spacy_lower = _build_prompt_task2_v5(task_description, input_text, variant=variant)
    prediction, _meta = _predict_with_fallbacks(
        input_prompt,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )
    if print_empty_prediction == "on" and not prediction:
        preview = input_text[:160].replace("\n", " ")
        print(f"[空预测 task2 spacy v5 test_sample] id={sid} preview={preview}")
    return {"test_sample_id": sid, "prediction": prediction}


def run_task2_test_samples_spacy_v5(
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
    variant = cast(Task2PromptVariant, "v5")
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
        print(f"[断点续跑 task2 spacy v5 test_samples] 已完成 {len(done_ids)} 条")

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
            desc=f"Task2 spacy v5 test_samples bs={batch_size}: {task_name}",
        ):
            chunk = pending[i : i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                futures = [
                    executor.submit(
                        _infer_one_v5,
                        sample,
                        task_description,
                        variant=variant,
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
    print(f"[保存完成 task2 spacy v5 test_samples] total={total}, file={output_file}")
    return {
        "task_id": task_id,
        "task_name": task_name,
        "prompt_version": "task2_spacy_v5_zeroshot_test_samples",
        "batch_size": batch_size,
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()

    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prompt] Task2 spaCy v5 零样本（同 infer_examples_main1_task2_spacy_v5 --prompt_variant v5）")
    print(f"[thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}, {_SUMMARY_JSON_NAME}")

    summary = run_task2_test_samples_spacy_v5(
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
