"""
Task7 test_samples 推理 V4：与 ``infer_examples_compare_task7_v4`` 同步的
Category/Clue 检索 query、Jeopardy V4 提示与可选 Verifier/ReAct。

输出格式与 ``infer_task7_test_samples.py`` 一致：每行
``{"test_sample_id": "...", "prediction": "..."}``，便于提交对比。
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from method_hyb import build_prompt, select_examples_hybrid

from infer_examples_compare_task7_v3 import (
    TASK7_CANONICAL_DESCRIPTION,
    _extract_output,
    _normalize_task7_answer,
    _resolve_output_dir,
    _task_json_path,
)
from infer_examples_compare_task7_v4 import (
    _infer_task7_v4_one,
    _parse_category_clue,
    _task7_retrieval_query_v4,
    ensure_task7_v4_prompt_registered,
)

_OUTPUT_JSONL_NAME = "openseek-7-test_samples-task7opt-v4-predictions.jsonl"
_SUMMARY_JSON_NAME = "summary_task7_test_samples_v4.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task7 test_samples 推理 V4（对齐 infer_examples_compare_task7_v4："
            "hybrid 检索 + V4 提示 + 可选 ReAct）。"
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
        "--ic_query_v4",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否启用 Category/Clue 分解后的检索 query（默认 on）",
    )
    p.add_argument(
        "--react",
        type=str,
        choices=["on", "off"],
        default="on",
        help="是否启用 Verifier + 条件性 Refine（默认 on）",
    )
    p.add_argument("--react_max_refines", type=int, default=1)
    p.add_argument(
        "--echo_threshold",
        type=float,
        default=0.72,
        help="预测与 Clue 词重叠超过该值时强制 Refine（默认 0.72）",
    )
    p.add_argument("--thinking_solver", type=str, choices=["on", "off"], default="off")
    p.add_argument("--thinking_verifier", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_model_output", type=str, choices=["on", "off"], default="off")
    p.add_argument("--print_empty_prediction", type=str, choices=["on", "off"], default="on")
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="并行条数（每条约 2~3 次 API；默认 8）",
    )
    p.add_argument(
        "--save_react_trace",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否在 JSONL 中写入 task7_v4_react 调试字段",
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


def run_task7_test_samples_v4(
    output_dir: Path,
    *,
    samples_limit: int = 0,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    resume: bool = False,
    task7_shot_k: int = 8,
    task7_retrieval_pool_size: int = 200,
    ic_query_v4: bool = True,
    use_react: bool = True,
    react_max_refines: int = 1,
    echo_threshold: float = 0.72,
    thinking_solver: str = "off",
    thinking_verifier: str = "off",
    print_empty_prediction: str = "on",
    batch_size: int = 8,
    save_react_trace: bool = False,
    log_model_output: str = "0",
) -> dict:
    ensure_task7_v4_prompt_registered()
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
        print(f"[断点续跑 V4 test_samples] 已完成 {len(done_ids)} 条")

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
            desc=f"Task7 V4 test_samples: {task_name}",
        ):
            batch = pending[start : start + max(1, batch_size)]
            prepared: list[dict] = []
            for sample in batch:
                sid = str(sample.get("id", "")).strip()
                input_text = str(sample.get("input", ""))
                cat, clue = _parse_category_clue(input_text)
                rq = _task7_retrieval_query_v4(task_description, input_text, enabled=ic_query_v4)
                examples_str = select_examples_hybrid(
                    all_examples=cleaned_examples,
                    task_description=task_description,
                    text2annotate=input_text,
                    top_k=max(1, task7_shot_k),
                    rerank_pool_size=max(20, task7_retrieval_pool_size),
                    use_explanation=False,
                    use_bm25_semantic_rerank=True,
                    exclude_example_id=sid,
                    retrieval_query_override=rq,
                )
                base_prompt = build_prompt(task_description, input_text, task_id=task_id)
                solver_prompt = base_prompt.replace("[[EXAMPLES]]\n\n", f"{examples_str}\n")
                prepared.append(
                    {
                        "test_sample_id": sid,
                        "input_text": input_text,
                        "solver_prompt": solver_prompt,
                        "refine_prompt_base": solver_prompt,
                        "category": cat,
                        "clue": clue,
                    }
                )

            workers = max(1, batch_size)

            def _work(item: dict) -> tuple[str, dict]:
                return _infer_task7_v4_one(
                    item["solver_prompt"],
                    item["refine_prompt_base"],
                    item["category"],
                    item["clue"],
                    use_react=use_react,
                    echo_threshold_force_fail=echo_threshold,
                    retries=retries,
                    retry_wait_seconds=retry_wait_seconds,
                    thinking_solver=thinking_solver,
                    thinking_verifier=thinking_verifier,
                    log_model=log_model_output,
                    max_refines=react_max_refines,
                )

            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_work, prepared))

            for item, (prediction, react_meta) in zip(prepared, results, strict=True):
                if print_empty_prediction == "on" and not prediction:
                    print(
                        f"[空预测 V4 test_sample] id={item['test_sample_id']} "
                        f"preview={item['input_text'][:160].replace(chr(10), ' ')}"
                    )
                row: dict = {
                    "test_sample_id": item["test_sample_id"],
                    "prediction": prediction,
                }
                if save_react_trace:
                    row["task7_v4_react"] = react_meta
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done_ids.add(item["test_sample_id"])

    total = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as rf:
            total = sum(1 for line in rf if line.strip())
    print(f"[保存完成 V4 test_samples] total={total}, file={output_file}")
    return {
        "task_id": 7,
        "task_name": task_name,
        "prompt_version": "task7_v4_test_samples",
        "total": total,
        "file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_task7_v4_prompt_registered()
    os.environ.setdefault("ANNOTATE_GLOBAL_API_LOCK", "0")
    os.environ.setdefault("ANNOTATE_FALLBACK_PLAIN", "1")
    os.environ.setdefault("ANNOTATE_MAX_PLAIN_CHARS", "120")
    os.environ.setdefault("DASHSCOPE_ENABLE_THINKING", "1" if args.thinking_solver == "on" else "0")

    log_model_out = "1" if args.print_model_output == "on" else "0"
    summary = run_task7_test_samples_v4(
        output_dir=output_dir,
        samples_limit=args.samples_limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task7_shot_k=args.task7_shot_k,
        task7_retrieval_pool_size=args.task7_retrieval_pool_size,
        ic_query_v4=args.ic_query_v4 == "on",
        use_react=args.react == "on",
        react_max_refines=max(0, args.react_max_refines),
        echo_threshold=float(args.echo_threshold),
        thinking_solver=args.thinking_solver,
        thinking_verifier=args.thinking_verifier,
        print_empty_prediction=args.print_empty_prediction,
        batch_size=max(1, args.batch_size),
        save_react_trace=args.save_react_trace == "on",
        log_model_output=log_model_out,
    )
    summary_path = output_dir / _SUMMARY_JSON_NAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False, indent=2)
    print(f"[汇总] {summary_path}")


if __name__ == "__main__":
    main()
