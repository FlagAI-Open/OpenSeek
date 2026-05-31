"""
预测 JSONL 可为两种常见格式之一：
- taskdata 扩展：含 ``model_output_base_normalized``（及可选 postprocess 字段）；
- 精简对比：仅有 ``model_output``（与 ``expected_output`` 对比），无 base 分裂字段。
并按 examples 学到的 emoji→Sad/Not sad 先验校准预测，打印处理前后准确率。

触发条件（可配）：
- 推文里 **emoji 字形种类数（去重）> 1**；
- 且至少存在一个出现在先验表中、且 **max(pct_sad_occ, pct_not_sad_occ) > 阈值** 的 emoji。

若多条 emoji 符合条件，按其各自主导标签 **多数表决**；若表决平局则沿用基线预测。

依赖: pip install emoji

示例:
    python src/task5_eval_emoji_postprocess_compare.py
    python src/task5_eval_emoji_postprocess_compare.py \\
        --pred outputs/其他输出.jsonl --prob-ge
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK5_JSON = REPO_ROOT / "data" / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"


def _extract_emojis(text: str) -> list[str]:
    try:
        from emoji import emoji_list
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "需要安装 emoji：python -m pip install emoji\n" + repr(exc)
        ) from exc
    return [item["emoji"] for item in emoji_list(text)]


def _normalize_label(s: Any) -> str | None:
    if s is None:
        return None
    t = str(s).strip()
    if t in ("Sad", "Not sad"):
        return t
    return None


def _baseline_prediction_raw(row: dict[str, Any]) -> Any:
    """不同对比 JSONL 里基线字段名不一致，按优先级取第一个非空。"""
    for key in (
        "model_output_base_normalized",
        "model_output_base",
        "model_output",
    ):
        if key not in row:
            continue
        v = row[key]
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _build_emoji_prior_from_examples(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """与 stat_openseek5_examples_emoji_labels 一致的 Sad/Not sad 共现计数。"""
    examples: list[dict[str, Any]] = bundle.get("examples") or []
    sad_occ: dict[str, int] = defaultdict(int)
    not_sad_occ: dict[str, int] = defaultdict(int)

    for ex in examples:
        raw = ex.get("output")
        if not isinstance(raw, list) or not raw:
            continue
        label = _normalize_label(raw[0])
        if label is None:
            continue

        text = ex.get("input") or ""
        found = _extract_emojis(text)
        if not found:
            continue

        m = sad_occ if label == "Sad" else not_sad_occ
        for ch in found:
            m[ch] += 1

    prior: dict[str, dict[str, Any]] = {}
    for emoji in set(sad_occ.keys()) | set(not_sad_occ.keys()):
        so, ns = sad_occ[emoji], not_sad_occ[emoji]
        t = so + ns
        pso = round(100.0 * so / t, 4) if t else 0.0
        pns = round(100.0 * ns / t, 4) if t else 0.0
        prior[emoji] = {
            "sad_occ": so,
            "not_sad_occ": ns,
            "total_occ": t,
            "pct_sad_occ": pso,
            "pct_not_sad_occ": pns,
        }
    return prior


def _dominant_label_from_prior(row: dict[str, Any]) -> str:
    return "Sad" if row["pct_sad_occ"] >= row["pct_not_sad_occ"] else "Not sad"


def _emoji_vote_postprocess(
    text: str,
    prior: dict[str, dict[str, Any]],
    *,
    prob_threshold: float,
    strict_prob_gt: bool,
    min_distinct_emojis: int,
) -> tuple[str | None, list[tuple[str, float, str]]]:
    """返回 (若触发则新模式预测否则 None, 参与投票的 emoji 列表 (emoji, max_pct, vote))"""
    seq = _extract_emojis(text)
    distinct = list(dict.fromkeys(seq))
    if len(distinct) <= min_distinct_emojis - 1:
        return None, []

    votes: list[tuple[str, float, str]] = []
    for em in distinct:
        row = prior.get(em)
        if row is None:
            continue
        mx = max(row["pct_sad_occ"], row["pct_not_sad_occ"])
        if strict_prob_gt:
            ok = mx > prob_threshold
        else:
            ok = mx >= prob_threshold
        if not ok:
            continue
        lab = _dominant_label_from_prior(row)
        votes.append((em, mx, lab))

    if not votes:
        return None, []

    c = Counter(v[2] for v in votes)
    top2 = c.most_common(2)
    if len(top2) == 1 or top2[0][1] > top2[1][1]:
        majority = top2[0][0]
    else:
        return None, votes  # 平局则不覆盖

    return majority, votes


def main() -> None:
    p = argparse.ArgumentParser(description="Task5 emoji 后处理准确率对比")
    p.add_argument(
        "--pred",
        type=Path,
        default=REPO_ROOT
        / "examples"
        / "openseek-5-examples-compare-emoji-off-striphash-on.jsonl",
        help=(
            "对比 JSONL：需 expected_output；基线为 model_output_base_normalized，"
            "若无则 model_output_base，再无则 model_output。"
        ),
    )
    p.add_argument(
        "--examples-json",
        type=Path,
        default=DEFAULT_TASK5_JSON,
        help="用于构建 emoji 先验的 task5 JSON（默认 data 下 openseek-5）。",
    )
    p.add_argument(
        "--prob-threshold",
        type=float,
        default=80.0,
        help="先验中高置信阈值（百分比，默认 90）。默认判定为 **严格大于**；若需≥请传 --prob-ge。",
    )
    p.add_argument(
        "--prob-ge",
        action="store_true",
        help="将概率条件改为 ≥ 阈值（默认是 > 阈值）。",
    )
    p.add_argument(
        "--min-distinct-emojis",
        type=int,
        default=2,
        help="推文去重 emoji 个数下限（默认 2，即「>1」种）。改为 3 则更严。",
    )
    args = p.parse_args()

    pred_path = args.pred.resolve()
    if not pred_path.is_file():
        raise SystemExit(f"找不到预测文件: {pred_path}")

    ex_path = args.examples_json.resolve()
    if not ex_path.is_file():
        raise SystemExit(f"找不到 examples JSON: {ex_path}")

    with ex_path.open(encoding="utf-8") as f:
        bundle = json.load(f)

    prior = _build_emoji_prior_from_examples(bundle)
    strict = not args.prob_ge

    rows: list[dict[str, Any]] = []
    # utf-8-sig：去掉部分编辑器写入的 BOM，避免 json.loads 在首个字符报错
    with pred_path.open(encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.replace("\x00", "").lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{pred_path} 第 {lineno} 行 JSON 无效: {exc}\n预览: {line[:240]!r}"
                ) from exc

    if not rows:
        raise SystemExit(f"未从 {pred_path} 解析到任何有效 JSON 行（是否文件为空或被误清空）。")

    ok_base = 0
    ok_new = 0
    n_override = 0
    n_tie_no_override = 0

    for row in rows:
        gold = _normalize_label(row.get("expected_output"))
        base = _normalize_label(_baseline_prediction_raw(row))
        if gold is None or base is None:
            raise SystemExit(
                "样本缺金标或无法解析基线预测。\n"
                f"  keys={list(row.keys())}\n"
                "  需要 expected_output ∈ {Sad, Not sad}；基线优先取 "
                "model_output_base_normalized → model_output_base → model_output。"
            )

        inp = row.get("input") or ""
        maj, voted = _emoji_vote_postprocess(
            inp,
            prior,
            prob_threshold=args.prob_threshold,
            strict_prob_gt=strict,
            min_distinct_emojis=args.min_distinct_emojis,
        )

        if maj is None and voted:
            n_tie_no_override += 1

        new_pred = maj if maj is not None else base
        if maj is not None:
            n_override += 1

        ok_base += int(base == gold)
        ok_new += int(new_pred == gold)

    n = len(rows)
    acc_base = 100.0 * ok_base / n if n else 0.0
    acc_new = 100.0 * ok_new / n if n else 0.0

    prob_desc = "> " if strict else ">= "
    print("=" * 64)
    print("Task5 emoji 后处理 — 准确率对比")
    print("=" * 64)
    print(f"预测文件: {pred_path}")
    print(f"先验来源: {ex_path}（examples 含 emoji 推文的 Sad/Not sad 出现比例）")
    print(f"条件: 去重 emoji 种类数 ≥ {args.min_distinct_emojis}（默认即「>1」种）")
    print(f"      且存在先验 emoji 满足 max(pct_sad, pct_not) {prob_desc}{args.prob_threshold}%")
    print(f"策略: 多数表决；平局则不改基线预测")
    print()
    print(f"样本数: {n}")
    print(f"基线准确率: {ok_base}/{n} = {acc_base:.2f}%")
    print(f"后处理准确率: {ok_new}/{n} = {acc_new:.2f}%")
    print(f"Δ 准确率: {acc_new - acc_base:+.2f} 百分点")
    print()
    print(f"被 emoji 规则覆盖的条数（非平局且成功表决）: {n_override}")
    print(f"满足条件但表决平局、保留基线: {n_tie_no_override}")


if __name__ == "__main__":
    main()