"""
Task 8 ``test_samples`` 推理 — v3 独立入口。

逻辑与 ``infer_examples_compare_task8_v3.py`` 中 ``run_task8_test_samples_v3`` 一致：
输出契约、thinking 剥离、AST 门控、增强 ReAct。

输出：
- ``{output_dir}/openseek-8-test_samples-predictions-v3.jsonl``
- ``{output_dir}/summary_task8_test_samples_v3.json``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from infer_examples_compare_task8_v3 import run_task8_test_samples_v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task8 test_samples 推理（v3 流水线，与 infer_examples_compare_task8_v3 对齐）。"
    )
    parser.add_argument("--output_dir", type=str, default="examples", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条 API 失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument("--task8_shot_k", type=int, default=6, help="每条样本检索示例数，默认 6。")
    parser.add_argument(
        "--task8_retrieval_pool_size",
        type=int,
        default=200,
        help="候选检索池大小（按相似度排序后截断），默认 200。",
    )
    parser.add_argument(
        "--task8_icl_retrieval",
        type=str,
        choices=["bm25_lexical", "semantic", "jaccard_only", "bm25_llm"],
        default="bm25_lexical",
        help="ICL 检索策略，与 infer_examples_compare_task8_v3 一致（默认 bm25_lexical）。",
    )
    parser.add_argument(
        "--task8_bm25_llm_pool",
        type=int,
        default=20,
        help="bm25_llm：BM25 初筛池大小。",
    )
    parser.add_argument(
        "--task8_bm25_llm_final_k",
        type=int,
        default=5,
        help="bm25_llm：LLM 精选后条数上限（与 task8_shot_k 取较小值）。",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否开启模型 thinking（映射到 DASHSCOPE_ENABLE_THINKING）。",
    )
    parser.add_argument(
        "--print_model_output",
        type=str,
        choices=["on", "off"],
        default="off",
        help="是否打印模型原始输出预览（通过 ANNOTATE_LOG_EVERY_RESPONSE）。",
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=8,
        help="并行推理批大小，默认 8。",
    )
    parser.add_argument(
        "--accuracy_criterion",
        type=str,
        choices=[
            "exec_outcome_match",
            "exec_dual_pass",
            "text",
            "text_and_exec",
            "text_and_exec_outcome",
        ],
        default="exec_outcome_match",
        help="预测侧记录用判据（test_samples 无 gold 时 is_match 恒为 False，字段仍写入 jsonl）。",
    )
    parser.add_argument(
        "--max_react_rounds",
        type=int,
        default=5,
        help="执行器 / AST / 前向探针失败后 ReAct 最大轮数（默认 5）。",
    )
    parser.add_argument(
        "--single_test_sample_id",
        type=str,
        default="",
        help="若非空：仅推理该 test_sample id。",
    )
    parser.add_argument(
        "--single_index",
        type=int,
        default=-1,
        help="若 >=0：仅推理待跑队列中的第 N 条（0-based）。",
    )
    return parser.parse_args()


def _resolve_output_dir(output_dir: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    p = Path(output_dir)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def main() -> None:
    args = parse_args()
    os.environ["DASHSCOPE_ENABLE_THINKING"] = "1" if args.thinking == "on" else "0"
    os.environ["ANNOTATE_LOG_EVERY_RESPONSE"] = "1" if args.print_model_output == "on" else "0"
    print(f"[v3 test_samples thinking] DASHSCOPE_ENABLE_THINKING={os.environ['DASHSCOPE_ENABLE_THINKING']}")
    print(f"[v3 print_model_output] ANNOTATE_LOG_EVERY_RESPONSE={os.environ['ANNOTATE_LOG_EVERY_RESPONSE']}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v3 test_samples 输出目录] {output_dir}")

    single_id = str(args.single_test_sample_id or "").strip()
    if single_id and args.single_index >= 0:
        raise SystemExit("请只指定 --single_test_sample_id 或 --single_index 之一。")

    summary_item = run_task8_test_samples_v3(
        output_dir=output_dir,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        task8_shot_k=args.task8_shot_k,
        task8_retrieval_pool_size=args.task8_retrieval_pool_size,
        retrieval_batch_size=args.retrieval_batch_size,
        accuracy_criterion=args.accuracy_criterion,
        task8_icl_retrieval=args.task8_icl_retrieval,
        task8_bm25_llm_pool=args.task8_bm25_llm_pool,
        task8_bm25_llm_final_k=args.task8_bm25_llm_final_k,
        max_react_rounds=max(1, args.max_react_rounds),
        single_example_id=single_id,
        single_index=int(args.single_index),
    )

    summary_file = output_dir / "summary_task8_test_samples_v3.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump([summary_item], f, ensure_ascii=False, indent=2)
    print(f"[v3 test_samples 汇总] {summary_file}")


if __name__ == "__main__":
    main()
