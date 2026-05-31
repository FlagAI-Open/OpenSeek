"""
Task7 test_samples 推理 V6：与 ``infer_examples_compare_task7_v6`` 对齐——

V2 Jeopardy 模板（register_task_prompt）、hybrid ICL、外循环 **LLM 审核** + reject 时注入

**审核 rationale** 的 ReAct 尾缀（不以 gold 为停机条件）。

正式 ``test_samples`` 通常无标签：``expected`` 为空，``gold_is_match`` 在 trace 中恒为 false，仅当有

开发用 ``output`` 字段时有对照意义。

默认输出每行 ``{"test_sample_id": "...", "prediction": "..."}``；可选写入 ``outer_trace``。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from infer_examples_compare_task7_v6 import (
    TASK7_CANONICAL_DESCRIPTION,
    _extract_output,
    _infer_task7_auto_agent_outer_loop,
    _normalize_task7_answer,
    _outer_trace_for_jsonl,
    _resolve_output_dir,
    _task_json_path,
    ensure_task7_v2_prompt_registered,
)
from method_hyb import build_prompt, select_examples_hybrid

_OUTPUT_JSONL_NAME = "openseek-7-test_samples-task7opt-v6-predictions.jsonl"
_SUMMARY_JSON_NAME = "summary_task7_test_samples_v6.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task7 test_samples V6（对齐 infer_examples_compare_task7_v6）："
            "V2 + hybrid ICL + LLM 审核外循环 + ReAct。"
        )
    )
    p.add_argument("--samples_limit", type=int, default=0, help="最多推理多少条；<=0 全部 test_samples")
    p.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_wait_seconds", type=float, default=2.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task7_shot_k", type=int, default=8)
    p.add_argument("--task7_retrieval_pool_size", type=int, default=200)
    p.add_argument(
        "--outer_rounds",
        type=int,
        default=3,
        help="外循环最多轮数（含首轮）；每轮后审核模型 accept/reject，默认 3。",
    )
    p.add_argument("--thinking", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_model_output", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_empty_prediction", type=str, choices=["on", "off"], default="on")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--quiet",
        action="store_true",
        help="关闭详细过程日志（求解/审核/ReAct 增量）；默认打印。",
    )
    p.add_argument(
        "--log_raw_max_chars",
        type=int,
        default=24_000,
        help="打印原始输出时的最大字符数；<=0 不截断。默认 24000。",
    )
    p.add_argument(
        "--save_outer_trace",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否在 JSONL 中写入 outer_trace / final_llm_judge_accepted。",
    )
    p.add_argument(
        "--jsonl_include_raw",
        action="store_true",
        help="与 save_outer_trace=on 联用：trace 中保留 raw_model_output / raw_judge_output。",
    )
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


def run_task7_test_samples_v6(
    output_dir: Path,
    *,
    samples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task7_shot_k: int = 8,
    task7_retrieval_pool_size: int = 200,
    outer_rounds: int = 3,
    verbose_process_log: bool = True,
    log_raw_max_chars: int = 24_000,
    jsonl_include_raw: bool = False,
    save_outer_trace: bool = False,
    print_empty_prediction: str = "on",
    batch_size: int = 8,
) -> dict:
    ensure_task7_v2_prompt_registered()

    task_id = 7
    task_description = TASK7_CANONICAL_DESCRIPTION

    with _task_json_path().open("r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    test_samples = list(task_dict.get("test_samples", []))
    if samples_limit > 0:
        test_samples = test_samples[:samples_limit]

    output_file = output_dir / _OUTPUT_JSONL_NAME
    done_ids = _load_done_ids(output_file) if resume else set()
    mode = "a" if resume else "w"
    if done_ids:
        print(f"[断点续跑 V6 test_samples] 已完成 {len(done_ids)} 条")

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

    outer_rounds_eff = max(1, outer_rounds)
    log_lock = threading.Lock() if verbose_process_log else None
    retain_raw = verbose_process_log or jsonl_include_raw

    with output_file.open(mode, encoding="utf-8") as wf:
        for start in tqdm(
            range(0, len(pending), max(1, batch_size)),
            desc=f"Task7 V6 test_samples: {task_name}",
        ):
            batch = pending[start : start + max(1, batch_size)]
            prepared: list[dict] = []
            for sample in batch:
                sid = str(sample.get("id", "")).strip()
                input_text = str(sample.get("input", ""))
                expected_raw = _extract_output(sample.get("output", ""))
                expected = _normalize_task7_answer(expected_raw) if expected_raw else ""

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
                        "expected": expected,
                        "input_prompt": input_prompt,
                    }
                )

            workers = max(1, batch_size)

            def _work(item: dict) -> tuple[str, bool, list[dict]]:
                pred, gold_ok, trace = _infer_task7_auto_agent_outer_loop(
                    item["input_prompt"],
                    item["expected"],
                    task_description,
                    item["input_text"],
                    retries,
                    retry_wait_seconds,
                    outer_rounds_eff,
                    example_id=str(item["test_sample_id"]),
                    verbose_log=verbose_process_log,
                    log_lock=log_lock,
                    log_raw_max_chars=log_raw_max_chars,
                    retain_raw_in_trace=retain_raw,
                )
                return pred, gold_ok, trace

            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_work, prepared))

            for item, (prediction, _gold_ok, outer_trace) in zip(prepared, results, strict=True):
                if print_empty_prediction == "on" and not prediction:
                    print(
                        f"[空预测 V6 test_sample] id={item['test_sample_id']} "
                        f"preview={item['input_text'][:160].replace(chr(10), ' ')}"
                    )
                row: dict = {
                    "test_sample_id": item["test_sample_id"],
                    "prediction": prediction,
                }
                if save_outer_trace:
                    final_llm_ok = (
                        bool(outer_trace[-1].get("llm_judge_accepted")) if outer_trace else False
                    )
                    row["final_llm_judge_accepted"] = final_llm_ok
                    row["outer_rounds_used"] = len(outer_trace)
                    row["outer_trace"] = _outer_trace_for_jsonl(
                        outer_trace, include_raw=jsonl_include_raw
                    )
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(item["test_sample_id"])

    total = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as rf:
            total = sum(1 for line in rf if line.strip())
    print(f"[保存完成 V6 test_samples] total={total}, file={output_file}")
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_v6_test_samples_llm_judge_react",
        "outer_rounds": outer_rounds_eff,
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    ensure_task7_v2_prompt_registered()

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")

    print("[prompt] Task7 Jeopardy 模板同 compare V6（V2 + hybrid ICL + LLM 审核外循环）")
    if args.quiet:
        print("[日志] --quiet：关闭求解/审核/ReAct 详细打印")
    summary = run_task7_test_samples_v6(
        output_dir=output_dir,
        samples_limit=args.samples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task7_shot_k=args.task7_shot_k,
        task7_retrieval_pool_size=args.task7_retrieval_pool_size,
        outer_rounds=max(1, args.outer_rounds),
        verbose_process_log=not args.quiet,
        log_raw_max_chars=args.log_raw_max_chars,
        jsonl_include_raw=args.jsonl_include_raw,
        save_outer_trace=args.save_outer_trace == "on",
        print_empty_prediction=args.print_empty_prediction,
        batch_size=max(1, args.batch_size),
    )
    summary_path = output_dir / _SUMMARY_JSON_NAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总] {summary_path}")


if __name__ == "__main__":
    main()
