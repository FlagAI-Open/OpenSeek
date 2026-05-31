"""
Task6 test_samples 推理（DeepSeek API）：任务定义检索 + 联合句对多数表决（taskdef_joint 变体）。

对齐 ``infer_task6_v2_vote_taskdef_joint.py`` 全流程（基础推理 → 变体后处理 → 标准 v2/v3 → 提交导出），
在导入相关模块前将 ``annotate_nvidia`` 全局替换为 DeepSeek。
"""

from __future__ import annotations

# 必须在 task6 流水线模块导入之前完成替换
from deepseek_client import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, install_deepseek_as_annotate_nvidia

install_deepseek_as_annotate_nvidia()

import argparse
import json
import os
from pathlib import Path

import infer_task6_v2 as t6
import infer_task6_v2_vote_taskdef_joint as joint
from deepseek_client import DEEPSEEK_MODEL as _DEFAULT_MODEL
from task6_infer_pipeline import (
    DEFAULT_BATCH_SIZE,
    postprocess_v2_path,
    postprocess_v3_path,
    run_standard_v2_postprocess,
    run_standard_v3_postprocess,
)

joint.BASE_JSONL_NAME = "openseek-6-v2-taskdef-joint-deepseek.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task6 test_samples：DeepSeek + taskdef_joint 投票流水线。",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=joint.DEFAULT_TASK6_FILE,
        help="`data/` 下的 task6 数据文件名。",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多推理多少条；<=0 表示全部。")
    parser.add_argument("--output_dir", type=str, default="outputs", help="结果输出目录。")
    parser.add_argument("--retries", type=int, default=3, help="单条失败重试次数。")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试等待秒数。")
    parser.add_argument("--resume", action="store_true", help="开启断点续跑。")
    parser.add_argument(
        "--deepseek_model",
        type=str,
        default=_DEFAULT_MODEL,
        help=f"DeepSeek 模型名，默认 {_DEFAULT_MODEL}。",
    )
    parser.add_argument(
        "--taskdef_max_chars",
        type=int,
        default=1200,
        help="注入 prompt 的任务 Definition 最大字符数（超出截断）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="基础推理并行批大小。",
    )
    parser.add_argument(
        "--flip_gate",
        type=str,
        choices=["none", "not_fiction", "block_nn_domain_n", "combined"],
        default="not_fiction",
        help="标准 v3 语篇后处理门控。",
    )
    parser.add_argument("--skip_v3", action="store_true", help="跳过标准 v3 后处理。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["DEEPSEEK_MODEL_OVERRIDE"] = args.deepseek_model

    print(f"[DeepSeek] model={args.deepseek_model} base_url={DEEPSEEK_BASE_URL}")
    print(f"[流水线] taskdef_joint（同 infer_task6_v2_vote_taskdef_joint）")
    print(f"[输出基线] {joint.BASE_JSONL_NAME}")

    output_dir = t6._resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    data_path = joint._task_json_path(args.data_file)
    with data_path.open("r", encoding="utf-8") as f:
        task_dict = json.load(f)
    task_definition = joint._trunc(joint._get_task_definition(task_dict), args.taskdef_max_chars)

    base_summary, base_file = joint._run_base_inference(
        data_file=args.data_file,
        output_dir=output_dir,
        limit=args.limit,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        resume=args.resume,
        taskdef_max_chars=args.taskdef_max_chars,
        batch_size=args.batch_size,
    )
    base_summary["model"] = args.deepseek_model
    base_summary["backend"] = "deepseek"

    variant_post_file = output_dir / f"{base_file.stem}-postprocess-variant{base_file.suffix}"
    variant_summary = joint._run_postprocess(
        input_path=base_file,
        output_path=variant_post_file,
        task_definition=task_definition,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )
    variant_summary["model"] = args.deepseek_model

    v2_file = postprocess_v2_path(variant_post_file)
    v2_summary = run_standard_v2_postprocess(
        variant_post_file,
        v2_file,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )
    v2_summary["model"] = args.deepseek_model
    print(f"[标准v2后处理] post_acc={v2_summary['post_acc']:.2%} file={v2_summary['file']}")

    summaries = [base_summary, variant_summary, v2_summary]
    final_file = Path(v2_summary["file"])

    if not args.skip_v3:
        v3_file = postprocess_v3_path(v2_file)
        v3_summary = run_standard_v3_postprocess(
            v2_file,
            v3_file,
            retries=args.retries,
            retry_wait_seconds=args.retry_wait_seconds,
            flip_gate=args.flip_gate,
        )
        v3_summary["model"] = args.deepseek_model
        print(
            f"[标准v3后处理] post_v3_acc={v3_summary['post_v3_accuracy']:.2%} "
            f"file={v3_summary['file']}"
        )
        final_file = Path(v3_summary["file"])
        summaries.append(v3_summary)

    submit_file = output_dir / f"{base_file.stem}-submit{base_file.suffix}"
    submit_summary = t6._export_submission(input_path=final_file, output_path=submit_file)
    summaries.append(submit_summary)

    summary_file = output_dir / "summary_task6_deepseek_taskdef_joint.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"[汇总完成] {summary_file}")


if __name__ == "__main__":
    main()
