"""Task6 推理流水线共用：标准 v2/v3 后处理与 JSONL 批处理路径约定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from postprocess_task6_v2 import process_file as process_v2_postprocess_file
from postprocess_task6_v3 import process_file as process_v3_postprocess_file

DEFAULT_BATCH_SIZE = 16


def postprocess_v2_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-postprocess-v2{input_path.suffix}")


def postprocess_v3_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-postprocess-v3{input_path.suffix}")


def run_standard_v2_postprocess(
    input_path: Path,
    output_path: Path | None,
    *,
    retries: int,
    retry_wait_seconds: float,
) -> dict[str, Any]:
    out = output_path if output_path is not None else postprocess_v2_path(input_path)
    (
        total,
        base_matched,
        post_matched,
        triggered_domain_nn,
        triggered_context_yn,
        triggered_context_ny,
        changed,
        changed_to_y,
        changed_from_context_yn,
        changed_from_context_ny,
        base_acc,
        post_acc,
    ) = process_v2_postprocess_file(
        input_path=input_path,
        output_path=out,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )
    return {
        "stage": "postprocess_v2",
        "total": total,
        "base_matched": base_matched,
        "post_matched": post_matched,
        "base_acc": base_acc,
        "post_acc": post_acc,
        "triggered_domain_nn": triggered_domain_nn,
        "triggered_context_yn": triggered_context_yn,
        "triggered_context_ny": triggered_context_ny,
        "changed": changed,
        "changed_to_y": changed_to_y,
        "changed_from_context_yn": changed_from_context_yn,
        "changed_from_context_ny": changed_from_context_ny,
        "file": str(out),
    }


def run_standard_v3_postprocess(
    input_path: Path,
    output_path: Path | None,
    *,
    retries: int,
    retry_wait_seconds: float,
    flip_gate: str = "not_fiction",
) -> dict[str, Any]:
    out = output_path if output_path is not None else postprocess_v3_path(input_path)
    stats = process_v3_postprocess_file(
        input_path=input_path,
        output_path=out,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
        flip_gate=flip_gate,
    )
    stats["stage"] = "postprocess_v3"
    stats["file"] = str(out)
    return stats


def run_v2_v3_postprocess_chain(
    input_path: Path,
    *,
    retries: int,
    retry_wait_seconds: float,
    flip_gate: str = "not_fiction",
    v2_output_path: Path | None = None,
    v3_output_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """input 需含 sentence1_pred/sentence2_pred/model_output（AND 或变体基线结果）。"""
    v2_out = v2_output_path if v2_output_path is not None else postprocess_v2_path(input_path)
    v2_summary = run_standard_v2_postprocess(
        input_path,
        v2_out,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )
    v3_out = v3_output_path if v3_output_path is not None else postprocess_v3_path(v2_out)
    v3_summary = run_standard_v3_postprocess(
        Path(v2_summary["file"]),
        v3_out,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
        flip_gate=flip_gate,
    )
    return v2_summary, v3_summary, Path(v3_summary["file"])
