"""
对已有 Task5 推理结果 JSONL（如 outputs_recall_v5 的 openseek-5-v1.jsonl）做 emoji 先验后处理。

每行需含 ``test_sample_id`` 与 ``prediction``；原文从 data 中 task5 JSON 的 ``test_samples`` 按 id 对齐。

依赖: pip install emoji

示例:
    python src/task5_postprocess_recall_jsonl.py \\
        --in outputs/outputs_recall_v5/openseek-5-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from infer_task5_test_samples import (  # noqa: PLC0415 同仓库脚本
    _finalize_task5_prediction,
    _strip_hashtag_symbol,
    _task_json_path,
)
from task5_eval_emoji_postprocess_compare import (  # noqa: PLC0415
    _build_emoji_prior_from_examples,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对 openseek-5 recall JSONL 写入 emoji 后处理结果。")
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=REPO_ROOT / "outputs" / "outputs_recall_v5" / "openseek-5-v1.jsonl",
        help="输入 JSONL：test_sample_id + prediction（默认 openseek-5-v1.jsonl）。",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="输出路径；默认在输入旁生成 *-postemoji.jsonl。",
    )
    p.add_argument(
        "--task5-json",
        type=Path,
        default=_task_json_path(),
        help="含 test_samples / examples 的 openseek-5 JSON。",
    )
    p.add_argument(
        "--strip-hashtag",
        type=str,
        choices=["on", "off"],
        default="off",
        help="对齐 input 时是否去掉 #（与推理 pipeline 一致时可设 on）。默认 off（用库里原始 input）。",
    )
    p.add_argument(
        "--prob-threshold",
        type=float,
        default=90.0,
        help="emoji 先验置信阈值（百分比）。",
    )
    p.add_argument(
        "--prob-ge",
        action="store_true",
        help="阈值改为 ≥（默认严格 >）。",
    )
    p.add_argument(
        "--min-distinct-emojis",
        type=int,
        default=2,
        help="去重 emoji 种类下限（默认 2）。",
    )
    return p.parse_args()


def _build_test_id_inputs(bundle: dict[str, Any], strip_hashtag: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in bundle.get("test_samples") or []:
        sid = str(item.get("id", "")).strip()
        if not sid:
            continue
        t = str(item.get("input", "") or "")
        if strip_hashtag:
            t = _strip_hashtag_symbol(t)
        out[sid] = t
    return out


def main() -> None:
    args = parse_args()
    inp = args.in_path.resolve()
    if not inp.is_file():
        raise SystemExit(f"输入不存在: {inp}")

    task_path = args.task5_json.resolve()
    if not task_path.is_file():
        raise SystemExit(f"找不到 task JSON: {task_path}")

    with task_path.open(encoding="utf-8") as f:
        bundle = json.load(f)

    id_to_input = _build_test_id_inputs(bundle, strip_hashtag=args.strip_hashtag == "on")
    prior = _build_emoji_prior_from_examples(bundle)
    strict_prob_gt = not args.prob_ge

    out_path = args.out_path.resolve() if args.out_path else inp.with_name(
        f"{inp.stem}-postemoji{inp.suffix}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_missing = n_lines = n_applied = 0
    with inp.open(encoding="utf-8-sig") as fin, out_path.open("w", encoding="utf-8") as fout:
        for lineno, raw in enumerate(fin, start=1):
            line = raw.replace("\x00", "").lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{inp} 第 {lineno} 行 JSON 无效: {exc}") from exc

            sid = str(row.get("test_sample_id", row.get("sample_id", ""))).strip()
            base_pred = row.get("prediction", row.get("model_output", ""))
            if not sid:
                raise SystemExit(f"{inp} 第 {lineno} 行缺少 test_sample_id/sample_id")

            text = id_to_input.get(sid)
            if text is None:
                n_missing += 1
                text = ""

            final, applied = _finalize_task5_prediction(
                str(base_pred) if base_pred is not None else "",
                text,
                postemoji_enabled=True,
                prior=prior,
                prob_threshold=args.prob_threshold,
                strict_prob_gt=strict_prob_gt,
                min_distinct_emojis=args.min_distinct_emojis,
            )

            out_row: dict[str, Any] = dict(row)
            out_row["prediction_base"] = base_pred
            out_row["prediction"] = final
            out_row["emoji_postprocess_applied"] = applied
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_lines += 1
            if applied:
                n_applied += 1

    print(f"[完成] 写出 {out_path}")
    print(f"  行数: {n_lines}, emoji 表决覆盖: {n_applied}, test_samples 缺 id（input 放空）: {n_missing}")
    if n_missing:
        print("  提示: 若有缺 id，请核对 task5 JSON test_samples 是否与输入 id 对齐。")


if __name__ == "__main__":
    main()
