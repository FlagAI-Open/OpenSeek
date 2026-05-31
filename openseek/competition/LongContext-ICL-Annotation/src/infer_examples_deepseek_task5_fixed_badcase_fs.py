"""
Task5 **examples** 推理（DeepSeek API）：固定 8 条 bad-case few-shot + 可选 hybrid 检索。

与 ``infer_task5_test_samples_deepseek_fixed_badcase_fs.py`` 同 prompt 策略，
输出 compare 格式（含 ``expected_output`` / ``is_match``），便于与 recall 路线对比。

示例::

    cd src
    python infer_examples_deepseek_task5_fixed_badcase_fs.py --examples_limit 500
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

from deepseek_client import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, deepseek_chat
from infer_examples_compare_task5_fixed_badcase_fs import (
    FIXED_TASK5_FS_SHOTS,
    REPO_ROOT,
    TASK5_CANONICAL_DESCRIPTION,
    TASK5_FILE,
    _build_fixed_fewshot_block,
    _compute_metrics_from_jsonl,
    _convert_task5_emoji,
    _extract_output,
    _load_done_ids,
    _normalize_text,
    _resolve_output_dir,
    _safe_console_text,
    _strip_hashtag_symbol,
)
from method_hyb import count_answer, select_examples_hybrid
from method_task5 import build_prompt


def _normalize_label(s: Any) -> str:
    t = str(s or "").strip()
    if t in ("Sad", "Not sad"):
        return t
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task5 examples：DeepSeek + 固定 bad-case few-shot（8 条）+ 可选 hybrid 检索。",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=500,
        help="最多推理多少条 examples；<=0 表示全部（默认 500）。",
    )
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument(
        "--task5_emoji_mode",
        type=str,
        choices=["off", "alias", "zh"],
        default="off",
        help="emoji 转换模式：off/alias/zh。",
    )
    parser.add_argument(
        "--task5_strip_hashtag",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否去掉 hashtag 的 #。",
    )
    parser.add_argument(
        "--task5_shot_k",
        type=int,
        default=4,
        help="hybrid 检索追加的示例数；0 表示仅使用固定 8 条 few-shot。",
    )
    parser.add_argument(
        "--task5_retrieval_pool_size",
        type=int,
        default=200,
        help="候选检索池大小（按相似度排序后截断），默认 200。",
    )
    parser.add_argument(
        "--deepseek_model",
        type=str,
        default=DEEPSEEK_MODEL,
        help=f"DeepSeek 模型名，默认 {DEEPSEEK_MODEL}。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否打印模型原始输出预览。",
    )
    parser.add_argument(
        "--print_empty_prediction",
        type=str,
        choices=["on", "off"],
        default="on",
        help="当解析结果为空时是否打印样本信息。",
    )
    parser.add_argument(
        "--infer_parallelism",
        type=int,
        default=8,
        help="并发推理线程数，默认 8。",
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=8,
        help="每批 prepare+infer 的样本数，默认 8。",
    )
    return parser.parse_args()


def _task_json_path() -> Path:
    return REPO_ROOT / "data" / TASK5_FILE


def _prepare_task5_example_item(
    example: dict,
    processed_all_examples: list[dict],
    task_description: str,
    fixed_examples_prefix: str,
    task5_emoji_mode: str,
    strip_hashtag_enabled: bool,
    task5_shot_k: int,
    task5_retrieval_pool_size: int,
) -> dict:
    example_id = str(example.get("id", "")).strip()
    input_text = str(example.get("input", ""))
    if strip_hashtag_enabled:
        input_text = _strip_hashtag_symbol(input_text)
    prompt_input_text = _convert_task5_emoji(str(input_text), task5_emoji_mode)
    expected = _extract_output(example.get("output", ""))

    if task5_shot_k > 0:
        retrieved = select_examples_hybrid(
            all_examples=processed_all_examples,
            task_description=task_description,
            text2annotate=prompt_input_text,
            top_k=max(1, task5_shot_k),
            rerank_pool_size=max(20, task5_retrieval_pool_size),
            use_explanation=False,
            use_bm25_semantic_rerank=True,
            exclude_example_id=example_id,
        )
        examples_str = fixed_examples_prefix + retrieved
    else:
        examples_str = fixed_examples_prefix

    return {
        "example_id": example_id,
        "input_text": input_text,
        "prompt_input_text": prompt_input_text,
        "expected": expected,
        "examples_str": examples_str,
    }


def _infer_task5_example_item(
    item: dict,
    task_description: str,
    *,
    deepseek_model: str,
    retries: int,
    retry_wait_seconds: float,
    print_model_output: bool,
) -> dict:
    example_id = item["example_id"]
    prompt_input_text = item["prompt_input_text"]
    examples_str = item["examples_str"]

    prompt = build_prompt(task_description, prompt_input_text, task_id=5)
    input_prompt = prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n\n")

    prediction_raw = ""
    prediction = ""
    for attempt in range(1, retries + 1):
        try:
            prediction_raw = deepseek_chat(input_prompt, model=deepseek_model) or ""
            if print_model_output and prediction_raw:
                preview = _safe_console_text(prediction_raw[:300].replace("\n", " "))
                print(f"[deepseek] example_id={example_id} raw_preview={preview}")
            parsed = count_answer(prediction_raw) if prediction_raw else None
            prediction = _normalize_label(parsed) if parsed else ""
            if not prediction and prediction_raw:
                prediction = _normalize_label(prediction_raw)
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                print(
                    f"[推理失败] task=5 deepseek example_id={example_id} "
                    f"attempt={attempt}/{retries} error={e}"
                )
            else:
                print(
                    f"[重试] task=5 deepseek example_id={example_id} "
                    f"attempt={attempt}/{retries} error={e}"
                )
                time.sleep(retry_wait_seconds)

    expected = item["expected"]
    is_match = _normalize_text(prediction) == _normalize_text(expected)
    return {
        "example_id": example_id,
        "input_text": item["input_text"],
        "expected": expected,
        "prediction_raw": prediction_raw,
        "prediction": prediction,
        "is_match": is_match,
    }


def run_task5_examples_deepseek_fixed_fs(
    output_dir: Path,
    *,
    examples_limit: int = 500,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task5_emoji_mode: str = "off",
    task5_strip_hashtag: str = "on",
    task5_shot_k: int = 4,
    task5_retrieval_pool_size: int = 200,
    deepseek_model: str = DEEPSEEK_MODEL,
    print_model_output: bool = False,
    print_empty_prediction: bool = True,
    infer_parallelism: int = 8,
    retrieval_batch_size: int = 8,
) -> dict:
    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_id = 5
    task_name = task_dict["task_name"]
    task_description = TASK5_CANONICAL_DESCRIPTION
    all_examples = list(task_dict["examples"])
    if examples_limit > 0:
        all_examples = all_examples[:examples_limit]

    strip_hashtag_enabled = task5_strip_hashtag == "on"
    hashtag_mode = "striphash-on" if strip_hashtag_enabled else "striphash-off"
    n_fixed = len(FIXED_TASK5_FS_SHOTS)
    output_file = output_dir / (
        f"openseek-{task_id}-examples-deepseek-fixedbadcase{n_fixed}fs-ret{task5_shot_k}-"
        f"emoji-{task5_emoji_mode}-{hashtag_mode}.jsonl"
    )
    done_ids = _load_done_ids(output_file) if resume else set()
    file_mode = "a" if resume else "w"

    if done_ids:
        print(f"[断点续跑] task=5 deepseek examples, 已完成 {len(done_ids)} 条")

    fixed_examples_prefix = _build_fixed_fewshot_block(strip_hashtag_enabled, task5_emoji_mode)

    processed_all_examples: list[dict] = []
    for ex in task_dict["examples"]:
        ex_input = str(ex.get("input", "")).strip()
        if strip_hashtag_enabled:
            ex_input = _strip_hashtag_symbol(ex_input)
        ex_input = _convert_task5_emoji(ex_input, task5_emoji_mode)
        processed_all_examples.append(
            {
                "id": ex.get("id", ""),
                "input": ex_input,
                "output": [_extract_output(ex.get("output", ""))],
            }
        )

    pending_examples = []
    for example in all_examples:
        example_id = str(example.get("id", "")).strip()
        if resume and example_id in done_ids:
            continue
        pending_examples.append(example)

    parallelism = max(1, int(infer_parallelism))
    batch_size = max(1, int(retrieval_batch_size))
    empty_count = 0

    with output_file.open(file_mode, encoding="utf-8") as wf:
        for i in tqdm(
            range(0, len(pending_examples), batch_size),
            desc=f"Task5 DeepSeek examples fixed-fs: {task_name}",
        ):
            chunk = pending_examples[i : i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(batch_size, len(chunk))
            ) as prep_executor:
                prepare_futures = [
                    prep_executor.submit(
                        _prepare_task5_example_item,
                        example=ex,
                        processed_all_examples=processed_all_examples,
                        task_description=task_description,
                        fixed_examples_prefix=fixed_examples_prefix,
                        task5_emoji_mode=task5_emoji_mode,
                        strip_hashtag_enabled=strip_hashtag_enabled,
                        task5_shot_k=task5_shot_k,
                        task5_retrieval_pool_size=task5_retrieval_pool_size,
                    )
                    for ex in chunk
                ]
                prepared_items = [f.result() for f in prepare_futures]

            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
                infer_futures = [
                    executor.submit(
                        _infer_task5_example_item,
                        item=item,
                        task_description=task_description,
                        deepseek_model=deepseek_model,
                        retries=retries,
                        retry_wait_seconds=retry_wait_seconds,
                        print_model_output=print_model_output,
                    )
                    for item in prepared_items
                ]
                infer_results = [f.result() for f in infer_futures]

            for result in infer_results:
                example_id = result["example_id"]
                prediction = result["prediction"]
                if print_empty_prediction and not prediction:
                    input_preview = _safe_console_text(
                        result["input_text"][:160].replace("\n", " ")
                    )
                    print(
                        f"[空预测] example_id={example_id} expected={result['expected']} "
                        f"input_preview={input_preview}"
                    )
                    empty_count += 1

                row = {
                    "example_id": example_id,
                    "input": result["input_text"],
                    "expected_output": result["expected"],
                    "model_output": prediction,
                    "prediction": prediction,
                    "prediction_raw": result["prediction_raw"],
                    "is_match": result["is_match"],
                }
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(example_id)

    total, match_count, accuracy = _compute_metrics_from_jsonl(output_file)
    print(
        f"[保存完成] task=5 deepseek examples, total={total}, matched={match_count}, "
        f"accuracy={accuracy:.2%}, empty_prediction={empty_count}, file={output_file}"
    )
    return {
        "task_id": 5,
        "task_name": task_name,
        "mode": "deepseek_fixed_badcase_fewshot_examples",
        "model": deepseek_model,
        "fixed_fs_count": n_fixed,
        "retrieval_shot_k": task5_shot_k,
        "examples_limit": examples_limit,
        "total": total,
        "matched": match_count,
        "accuracy": accuracy,
        "empty_prediction": empty_count,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("ICL_DISABLE_SEMANTIC", "0")
    os.environ.setdefault("ICL_DISABLE_RERANK", "0")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DeepSeek] model={args.deepseek_model} base_url={DEEPSEEK_BASE_URL}")
    print(f"[输出目录] {output_dir}")
    print(f"[examples_limit] {args.examples_limit}")

    summary_item = run_task5_examples_deepseek_fixed_fs(
        output_dir=output_dir,
        examples_limit=args.examples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task5_emoji_mode=args.task5_emoji_mode,
        task5_strip_hashtag=args.task5_strip_hashtag,
        task5_shot_k=args.task5_shot_k,
        task5_retrieval_pool_size=args.task5_retrieval_pool_size,
        deepseek_model=args.deepseek_model,
        print_model_output=args.print_model_output == "on",
        print_empty_prediction=args.print_empty_prediction == "on",
        infer_parallelism=args.infer_parallelism,
        retrieval_batch_size=args.retrieval_batch_size,
    )
    summary_file = output_dir / "summary_task5_deepseek_examples_fixed_badcase_fs.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
