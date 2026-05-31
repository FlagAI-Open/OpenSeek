#!/usr/bin/env python3
"""
Task6 三路结果多数票融合（Y/N）。

默认输入（均在 outputs/task6_vote_fusion/ 下）：
  - v2_baseline/...-postprocess-v3.jsonl
  - taskdef_joint/...-postprocess-v3.jsonl
  - pair_cross/...-postprocess-v3.jsonl

用法：
  python src/fuse_task6_vote.py
  python src/fuse_task6_vote.py --tie 0 --out outputs/task6_vote_fusion/fused.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_BASE = REPO_ROOT / "outputs" / "task6_vote_fusion"

DEFAULT_INPUTS = [
    DEFAULT_OUT_BASE
    / "v2_baseline/openseek-6-v1-postprocess-v2-postprocess-v3.jsonl",
    DEFAULT_OUT_BASE
    / "taskdef_joint/openseek-6-v2-taskdef-joint-postprocess-variant-postprocess-v2-postprocess-v3.jsonl",
    DEFAULT_OUT_BASE
    / "pair_cross/openseek-6-v2-pair-cross-postprocess-variant-postprocess-v2-postprocess-v3.jsonl",
]

SOURCE_LABELS = ("v2_baseline", "taskdef_joint", "pair_cross")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6 三路 Y/N 多数票融合")
    parser.add_argument(
        "paths",
        nargs="*",
        help="输入 JSONL（可传 3 个）；不传则用默认三路 post-v3 结果",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_BASE / "openseek-6-vote-fusion.jsonl",
        help="融合结果 JSONL",
    )
    parser.add_argument(
        "--submit",
        type=Path,
        default=None,
        help="提交 JSONL；默认与 --out 同目录 openseek-6-vote-fusion-submit.jsonl",
    )
    parser.add_argument(
        "--tie",
        type=int,
        default=0,
        help="平局时采用第几个输入文件的预测（0-based，默认 0=v2_baseline）",
    )
    return parser.parse_args()


def _norm_yn(value: object) -> str:
    s = str(value or "").strip().upper()
    if s in {"Y", "N"}:
        return s
    return ""


def _load_jsonl(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {exc}") from exc
            eid = str(row.get("test_sample_id", "") or row.get("example_id", "")).strip()
            if eid:
                by_id[eid] = row
    return by_id


def _majority_vote(preds: list[str], tie_idx: int) -> tuple[str, bool]:
    """返回 (融合标签, 是否平局后使用 tie 规则)。"""
    valid = [p for p in preds if p in {"Y", "N"}]
    if not valid:
        pick = preds[tie_idx] if tie_idx < len(preds) else ""
        return (_norm_yn(pick) or "N"), True

    y = sum(1 for p in valid if p == "Y")
    n = len(valid) - y
    if y > n:
        return "Y", False
    if n > y:
        return "N", False

    pick = valid[tie_idx] if tie_idx < len(valid) else valid[0]
    return pick, True


def fuse(
    input_paths: list[Path],
    output_path: Path,
    submit_path: Path,
    tie_idx: int,
) -> dict:
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"输入不存在: {p}")

    maps = [_load_jsonl(p) for p in input_paths]
    inter: set[str] | None = None
    for m in maps:
        ids = set(m.keys())
        inter = ids if inter is None else inter & ids
    assert inter is not None
    ids = sorted(inter)

    labels = SOURCE_LABELS[: len(input_paths)]
    if len(labels) < len(input_paths):
        labels = [p.parent.name for p in input_paths]

    fused_rows: list[dict] = []
    ties = 0
    unanimous = 0
    per_source_match_fused = [0] * len(maps)

    for eid in ids:
        rows = [m[eid] for m in maps]
        preds = [_norm_yn(r.get("model_output", "") or r.get("prediction", "")) for r in rows]
        fused, used_tie = _majority_vote(preds, tie_idx)
        if used_tie:
            ties += 1
        if len(set(preds)) == 1:
            unanimous += 1

        base = rows[0]
        expected = _norm_yn(base.get("expected_output", ""))
        is_match = fused == expected if expected else False
        if is_match:
            for i, p in enumerate(preds):
                if p == expected:
                    per_source_match_fused[i] += 1

        fused_rows.append(
            {
                "test_sample_id": eid,
                "example_id": eid,
                "input": base.get("input", ""),
                "genre": base.get("genre", ""),
                "expected_output": expected,
                "model_output": fused,
                "prediction": fused,
                "is_match": is_match,
                "vote_sources": labels,
                "vote_predictions": preds,
                "vote_tie_used": used_tie,
                "vote_tie_index": tie_idx,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as wf:
        for row in fused_rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    with submit_path.open("w", encoding="utf-8", newline="\n") as wf:
        for row in fused_rows:
            wf.write(
                json.dumps(
                    {
                        "test_sample_id": row["test_sample_id"],
                        "prediction": row["model_output"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    matched = sum(1 for r in fused_rows if r["is_match"])
    has_label = sum(1 for r in fused_rows if r["expected_output"])
    return {
        "intersection": len(ids),
        "unanimous": unanimous,
        "ties_resolved": ties,
        "matched": matched,
        "has_expected": has_label,
        "per_source_acc_vs_expected": [
            (per_source_match_fused[i], has_label) for i in range(len(maps))
        ],
        "output": str(output_path),
        "submit": str(submit_path),
        "inputs": [str(p) for p in input_paths],
    }


def main() -> None:
    args = parse_args()
    input_paths = [Path(p).resolve() for p in (args.paths or DEFAULT_INPUTS)]
    output_path = args.out.resolve()
    submit_path = (
        args.submit.resolve()
        if args.submit
        else output_path.with_name("openseek-6-vote-fusion-submit.jsonl")
    )

    stats = fuse(input_paths, output_path, submit_path, tie_idx=args.tie)

    print(f"[融合] 交集样本数: {stats['intersection']}")
    print(f"[融合] 三路一致: {stats['unanimous']}  平局用 tie 规则: {stats['ties_resolved']}")
    print(f"[输出] {stats['output']}")
    print(f"[提交] {stats['submit']}")

    if stats["has_expected"]:
        n = stats["has_expected"]
        print(
            f"[准确率] 融合: {stats['matched']}/{n} = "
            f"{stats['matched']/n:.2%}"
        )
        for i, (ok, tot) in enumerate(stats["per_source_acc_vs_expected"]):
            name = SOURCE_LABELS[i] if i < len(SOURCE_LABELS) else f"src{i}"
            print(f"  单路[{name}]: {ok}/{tot} = {ok/tot:.2%}")
    else:
        print("[准确率] 无 expected_output，仅生成融合预测与提交文件")
        for i, p in enumerate(input_paths):
            name = SOURCE_LABELS[i] if i < len(SOURCE_LABELS) else p.parent.name
            print(f"  输入[{i}] {name}: {p.name}")


if __name__ == "__main__":
    main()
