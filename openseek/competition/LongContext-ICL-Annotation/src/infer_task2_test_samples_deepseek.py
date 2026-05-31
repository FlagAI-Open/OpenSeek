"""
Task2（openseek-count_nouns_verbs）test_samples 推理：**DeepSeek API + spaCy v3 零样本**。

对齐 ``infer_task2_test_samples_spacy_v3.py`` 的 prompt（``infer_examples_main1_task2_spacy_v3``），
仅将 ``annotate_nvidia`` 换为 DeepSeek。输出每行 ``{"test_sample_id": "...", "prediction": "..."}``。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from deepseek_client import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, annotate_with_retry
from infer_examples_main1_task2_spacy_v3 import (
    TASK_DATA_FILES,
    _prompt_openseek_2_spacy_v3,
    _resolve_output_dir,
    build_prompt,
    register_task_prompt,
)

_OUTPUT_JSONL_NAME = "openseek-2-test_samples-task2-deepseek-spacy-v3-zeroshot-predictions.jsonl"
_SUMMARY_JSON_NAME = "summary_task2_test_samples_deepseek_spacy_v3.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task2 test_samples 推理：DeepSeek + spaCy v3 零样本（对齐 infer_task2_test_samples_spacy_v3）。"
        )
    )
    p.add_argument("--samples_limit", type=int, default=0, help="最多推理多少条；<=0 全部 test_samples")
    p.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_wait_seconds", type=float, default=2.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--deepseek_model",
        type=str,
        default=DEEPSEEK_MODEL,
        help=f"DeepSeek 模型名，默认 {DEEPSEEK_MODEL}。",
    )
    p.add_argument("--print_model_output", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_empty_prediction", type=str, choices=["on", "off"], default="on")
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


def run_task2_test_samples_deepseek(
    output_dir: Path,
    *,
    samples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    deepseek_model: str = DEEPSEEK_MODEL,
    print_model_output: bool = False,
    print_empty_prediction: bool = True,
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
        print(f"[断点续跑 task2 deepseek test_samples] 已完成 {len(done_ids)} 条")

    with output_file.open(mode, encoding="utf-8") as wf:
        for sample in tqdm(test_samples, desc=f"Task2 DeepSeek test_samples: {task_name}"):
            sid = str(sample.get("id", "")).strip()
            if resume and sid in done_ids:
                continue

            input_text = str(sample.get("input", ""))
            input_prompt = build_prompt(task_description, input_text, task_id=task_id)

            def _on_err(attempt: int, total: int, err: Exception) -> None:
                if attempt >= total:
                    print(f"[推理失败] task2 deepseek id={sid} attempt={attempt}/{total} err={err}")
                else:
                    print(f"[重试] task2 deepseek id={sid} attempt={attempt}/{total} err={err}")
                    time.sleep(retry_wait_seconds)

            prediction = annotate_with_retry(
                input_prompt,
                retries=retries,
                retry_wait_seconds=retry_wait_seconds,
                model=deepseek_model,
                on_error=_on_err,
            )

            if print_model_output and prediction:
                preview = prediction[:200].replace("\n", " ")
                print(f"[deepseek task2] id={sid} prediction_preview={preview}")

            if print_empty_prediction and not prediction:
                preview = input_text[:160].replace("\n", " ")
                print(f"[空预测 task2 deepseek] id={sid} preview={preview}")

            wf.write(
                json.dumps({"test_sample_id": sid, "prediction": prediction}, ensure_ascii=False) + "\n"
            )
            wf.flush()
            done_ids.add(sid)

    total = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as rf:
            total = sum(1 for line in rf if line.strip())
    print(f"[保存完成 task2 deepseek test_samples] total={total}, file={output_file}")
    return {
        "task_id": task_id,
        "task_name": task_name,
        "prompt_version": "task2_deepseek_spacy_v3_zeroshot_test_samples",
        "model": deepseek_model,
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    register_task_prompt(2, _prompt_openseek_2_spacy_v3)

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DeepSeek] model={args.deepseek_model} base_url={DEEPSEEK_BASE_URL}")
    print(f"[prompt] Task2 spaCy v3 零样本（同 infer_task2_test_samples_spacy_v3）")
    print(f"[输出目录] {output_dir} -> {_OUTPUT_JSONL_NAME}")

    summary = run_task2_test_samples_deepseek(
        output_dir=output_dir,
        samples_limit=args.samples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        deepseek_model=args.deepseek_model,
        print_model_output=args.print_model_output == "on",
        print_empty_prediction=args.print_empty_prediction == "on",
    )
    summary_path = output_dir / _SUMMARY_JSON_NAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总] {summary_path}")


if __name__ == "__main__":
    main()
