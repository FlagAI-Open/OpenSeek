#!/usr/bin/env python3
"""
在已有 post-v3 JSONL 上离线回放多种「语篇翻转」约束，对比准确率（无需调 API）。

输入需含：model_output_post_v2, discourse_context_pred, sentence1_pred, sentence2_pred,
domain_same_pred, sentence2_context_pred, expected_output, genre 等字段。

用法：
  python src/task6_eval_post_v3_constraints.py
  python src/task6_eval_post_v3_constraints.py --input examples/...-post-v3.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="post-v3 约束策略离线对比")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "examples/openseek-6-examples-compare-task6-v2-sentence-and-postprocess-v2-post-v3.jsonl",
        help="含 discourse_context_pred 的 post-v3 结果",
    )
    return parser.parse_args()


def _label(value: Any) -> str:
    return str(value or "").strip().upper()


def _is_nn(row: dict) -> bool:
    return _label(row.get("sentence1_pred")) == "N" and _label(row.get("sentence2_pred")) == "N"


def _is_yn(row: dict) -> bool:
    return _label(row.get("sentence1_pred")) == "Y" and _label(row.get("sentence2_pred")) == "N"


def _metrics(rows: list[dict], pred_fn: Callable[[dict], str]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        exp = _label(row.get("expected_output"))
        pred = _label(pred_fn(row))
        if exp == "Y" and pred == "Y":
            tp += 1
        elif exp == "N" and pred == "N":
            tn += 1
        elif exp == "Y" and pred == "N":
            fn += 1
        else:
            fp += 1
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": n,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "spec": spec,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _predict_v2(row: dict) -> str:
    return _label(row.get("model_output_post_v2"))


def _predict_v3_full(row: dict) -> str:
    v2 = _label(row.get("model_output_post_v2"))
    if v2 != "N":
        return v2
    if _label(row.get("discourse_context_pred")) == "Y":
        return "Y"
    return "N"


def _allow_flip(row: dict, gate: Callable[[dict], bool]) -> bool:
    if _label(row.get("model_output_post_v2")) != "N":
        return False
    if _label(row.get("discourse_context_pred")) != "Y":
        return False
    return gate(row)


def _predict_with_gate(row: dict, gate: Callable[[dict], bool]) -> str:
    v2 = _label(row.get("model_output_post_v2"))
    if v2 != "N":
        return v2
    if _allow_flip(row, gate):
        return "Y"
    return "N"


# --- 约束门控 ---


def _gate_v2_signal(row: dict) -> bool:
    """v2 任一辅助信号为 Y：同领域 或 Y+N/N+Y 上下文判 Y。"""
    if _label(row.get("domain_same_pred")) == "Y":
        return True
    if _label(row.get("sentence2_context_pred")) == "Y":
        return True
    if _label(row.get("sentence1_context_pred")) == "Y":
        return True
    return False


def _gate_not_nn_domain_n(row: dict) -> bool:
    """禁止：N+N 且 v2 同领域已为 N 仍靠语篇单独翻 Y。"""
    if _is_nn(row) and _label(row.get("domain_same_pred")) == "N":
        return False
    return True


def _gate_any_sentence_y(row: dict) -> bool:
    """至少一句单句体裁判为 Y。"""
    return _label(row.get("sentence1_pred")) == "Y" or _label(row.get("sentence2_pred")) == "Y"


def _gate_not_fiction(row: dict) -> bool:
    return _label(row.get("genre")) != "FICTION"


def _gate_v2_or_not_nn(row: dict) -> bool:
    return _gate_v2_signal(row) or not _is_nn(row)


def _gate_v2_or_sentence_y(row: dict) -> bool:
    return _gate_v2_signal(row) or _gate_any_sentence_y(row)


def _gate_combined(row: dict) -> bool:
    """推荐组合：非 (N+N∧同领域N) 且 (v2信号 或 任一句Y) 且 非 fiction-only 风险 — 简化为 v2信号∨非NN∨任句Y，且非(fiction∧NN∧domain N)。"""
    if _label(row.get("genre")) == "FICTION" and _is_nn(row) and _label(row.get("domain_same_pred")) == "N":
        return False
    return _gate_v2_signal(row) or not _is_nn(row) or _gate_any_sentence_y(row)


STRATEGIES: list[tuple[str, str, Callable[[dict], str]]] = [
    ("v2_baseline", "v2 结果（不启用 v3）", _predict_v2),
    ("v3_full", "v3 全量：discourse=Y 即翻 Y", _predict_v3_full),
    (
        "c1_v2_signal",
        "约束1：discourse=Y 且 (同领域Y 或 S2上下文Y)",
        lambda r: _predict_with_gate(r, _gate_v2_signal),
    ),
    (
        "c2_block_nn_domain_n",
        "约束2：禁止 N+N 且 domain_same=N 单独翻 Y",
        lambda r: _predict_with_gate(r, _gate_not_nn_domain_n),
    ),
    (
        "c3_not_fiction",
        "约束3：discourse=Y 且 genre≠fiction",
        lambda r: _predict_with_gate(r, _gate_not_fiction),
    ),
    (
        "c4_any_sentence_y",
        "约束4：discourse=Y 且 (s1=Y 或 s2=Y)",
        lambda r: _predict_with_gate(r, _gate_any_sentence_y),
    ),
    (
        "c5_v2_or_not_nn",
        "约束5：discourse=Y 且 (v2信号Y 或 非N+N)",
        lambda r: _predict_with_gate(r, _gate_v2_or_not_nn),
    ),
    (
        "c6_v2_or_sentence_y",
        "约束6：discourse=Y 且 (v2信号Y 或 任一句Y)",
        lambda r: _predict_with_gate(r, _gate_v2_or_sentence_y),
    ),
    (
        "c7_combined",
        "约束7：组合（禁 fiction∧N+N∧domain N；否则 v2信号∨非NN∨任句Y）",
        lambda r: _predict_with_gate(r, _gate_combined),
    ),
]


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入不存在: {input_path}")

    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"[输入] {input_path}  n={len(rows)}\n")
    print(
        f"{'策略':<22} {'准确率':>8} {'Δvs v2':>8} {'P(Y)':>8} {'R(Y)':>8} "
        f"{'F1':>8} {'特异度':>8} {'FP':>5} {'FN':>5} {'翻转':>5}"
    )
    print("-" * 100)

    v2_acc = _metrics(rows, _predict_v2)["acc"]
    v3_full_flip = sum(
        1
        for r in rows
        if _label(r.get("model_output_post_v2")) == "N"
        and _label(r.get("discourse_context_pred")) == "Y"
    )

    best_name = ""
    best_acc = -1.0
    results: list[dict] = []

    for key, desc, pred_fn in STRATEGIES:
        m = _metrics(rows, pred_fn)
        flips = sum(
            1
            for r in rows
            if _label(pred_fn(r)) == "Y" and _label(r.get("model_output_post_v2")) == "N"
        )
        delta = m["acc"] - v2_acc
        print(
            f"{key:<22} {m['acc']*100:7.2f}% {delta*100:+7.2f}pp {m['prec']*100:7.2f}% "
            f"{m['rec']*100:7.2f}% {m['f1']*100:7.2f}% {m['spec']*100:7.2f}% "
            f"{m['fp']:5d} {m['fn']:5d} {flips:5d}"
        )
        results.append({"key": key, "desc": desc, **m, "flips": flips, "delta_vs_v2": delta})
        if m["acc"] > best_acc:
            best_acc = m["acc"]
            best_name = key

    print("-" * 100)
    print(f"v3_full 理论翻转上限（discourse=Y 条数）: {v3_full_flip}")
    print(f"\n[最高准确率] {best_name}  acc={best_acc*100:.2f}%")

    # 相对 v3_full 的 tradeoff
    full = next(r for r in results if r["key"] == "v3_full")
    print("\n[相对 v3_full 的 FP/FN 变化]（负 FP、正 FN 表示更保守）")
    for r in results:
        if r["key"] in ("v2_baseline", "v3_full"):
            continue
        print(
            f"  {r['key']:<22} Δacc={r['delta_vs_v2']*100:+.2f}pp  "
            f"ΔFP={r['fp']-full['fp']:+4d}  ΔFN={r['fn']-full['fn']:+4d}  flips={r['flips']}"
        )


if __name__ == "__main__":
    main()
