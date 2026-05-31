"""
统计 openseek-5 任务 JSON 中 `examples` 里出现的所有 emoji，
并按金标 Sad / Not sad 统计每个 emoji 的出现次数与占比。
结果写入 JSONL（首行 meta，随后每行一条 emoji），便于与 test_sample 预测结果后处理对接。

依赖: pip install emoji

用法（在仓库根目录）:
    python src/stat_openseek5_examples_emoji_labels.py
    python src/stat_openseek5_examples_emoji_labels.py --out outputs/my_emoji_stats.jsonl
    python src/stat_openseek5_examples_emoji_labels.py --quiet   # 仅写文件，不打印 TSV
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO_ROOT / "data" / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
DEFAULT_JSONL_OUT = REPO_ROOT / "outputs" / "openseek5_examples_emoji_label_stats.jsonl"


def _extract_emojis(text: str) -> list[str]:
    try:
        from emoji import emoji_list
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "未找到第三方库 emoji，请先安装：\n"
            "  python -m pip install emoji\n"
            "原版错误: {}".format(exc)
        ) from exc
    return [item["emoji"] for item in emoji_list(text)]


def _normalize_label(example: dict[str, Any]) -> str | None:
    raw = example.get("output")
    if isinstance(raw, list) and raw:
        s = str(raw[0]).strip()
        if s in ("Sad", "Not sad"):
            return s
    return None


def _emoji_rows(
    emojis: list[str],
    sad_occ: dict[str, int],
    not_sad_occ: dict[str, int],
    sad_ex: dict[str, int],
    not_sad_ex: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for em in emojis:
        so, no_ = sad_occ[em], not_sad_occ[em]
        t = so + no_
        pso = round(100.0 * so / t, 4) if t else 0.0
        pno = round(100.0 * no_ / t, 4) if t else 0.0

        se, ne = sad_ex[em], not_sad_ex[em]
        te = se + ne
        pse = round(100.0 * se / te, 4) if te else 0.0
        pne = round(100.0 * ne / te, 4) if te else 0.0

        rows.append(
            {
                "kind": "emoji",
                "emoji": em,
                "sad_occ": so,
                "not_sad_occ": no_,
                "total_occ": t,
                "pct_sad_occ": pso,
                "pct_not_sad_occ": pno,
                "sad_ex": se,
                "not_sad_ex": ne,
                "examples_with_emoji": te,
                "pct_sad_ex": pse,
                "pct_not_sad_ex": pne,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 examples 中 emoji 及 Sad / Not sad 分布。")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_JSON,
        help="openseek-5 JSON 路径（默认仓库 data 下文件名）。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_JSONL_OUT,
        help="输出的 JSONL 路径（默认 outputs/openseek5_examples_emoji_label_stats.jsonl）。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不写 TSV 到 stdout，仅写 JSONL 并打印一行摘要。",
    )
    args = parser.parse_args()

    path = args.data.resolve()
    if not path.is_file():
        raise SystemExit(f"找不到数据文件: {path}")

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(encoding="utf-8") as f:
        bundle = json.load(f)

    task_id = bundle.get("task_id", "openseek-5")
    examples: list[dict[str, Any]] = bundle.get("examples") or []
    sad_occ: dict[str, int] = defaultdict(int)
    not_sad_occ: dict[str, int] = defaultdict(int)
    sad_ex: dict[str, int] = defaultdict(int)
    not_sad_ex: dict[str, int] = defaultdict(int)

    n_sad_examples = n_not_examples = 0
    for ex in examples:
        label = _normalize_label(ex)
        if label is None:
            continue
        if label == "Sad":
            n_sad_examples += 1
        else:
            n_not_examples += 1

        text = ex.get("input") or ""
        found = _extract_emojis(text)
        if not found:
            continue

        uniq = set(found)
        occ_map = sad_occ if label == "Sad" else not_sad_occ
        ex_map = sad_ex if label == "Sad" else not_sad_ex
        for ch in found:
            occ_map[ch] += 1
        for ch in uniq:
            ex_map[ch] += 1

    emojis = sorted(
        set(sad_occ.keys()) | set(not_sad_occ.keys()),
        key=lambda e: -(sad_occ[e] + not_sad_occ[e]),
    )

    total_occ_sum = sum(sad_occ[e] + not_sad_occ[e] for e in emojis)
    labeled_examples = n_sad_examples + n_not_examples

    meta: dict[str, Any] = {
        "kind": "meta",
        "schema_version": 1,
        "task_id": task_id,
        "source_examples_json": str(path),
        "n_labeled_examples": labeled_examples,
        "n_sad_examples": n_sad_examples,
        "n_not_sad_examples": n_not_examples,
        "n_distinct_emojis": len(emojis),
        "total_emoji_occurrences_in_examples": total_occ_sum,
        "description_zh": (
            "基于 examples 金标统计；emoji 字段可与 test_sample.input 抽取结果 join。"
            "pct_* 为该 emoji 自身条件下的占比（0–100）。"
        ),
    }

    emoji_records = _emoji_rows(emojis, sad_occ, not_sad_occ, sad_ex, not_sad_ex)

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for rec in emoji_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[jsonl] 已写入 {out_path}（1 行 meta + {len(emoji_records)} 行 emoji）")

    if args.quiet:
        return

    print("=" * 72)
    print("openseek-5 — examples 中 emoji 与 Sad / Not sad 分布（TSV）")
    print("=" * 72)
    print(f"数据文件: {path}")
    print(f"有金标的 examples: Sad={n_sad_examples}, Not sad={n_not_examples}, 合计={labeled_examples}")
    print(f"独立 emoji 种数: {len(emojis)}")
    print(f"emoji 出现次数总和: {total_occ_sum}")
    print()

    colnames = (
        "emoji",
        "sad_occ",
        "not_sad_occ",
        "total_occ",
        "pct_sad_occ",
        "pct_not_sad_occ",
        "sad_ex",
        "not_sad_ex",
        "examples_with_emoji",
        "pct_sad_ex",
        "pct_not_sad_ex",
    )
    print("\t".join(colnames))

    for rec in emoji_records:
        em = rec["emoji"]
        print(
            "\t".join(
                [
                    em,
                    str(rec["sad_occ"]),
                    str(rec["not_sad_occ"]),
                    str(rec["total_occ"]),
                    str(rec["pct_sad_occ"]),
                    str(rec["pct_not_sad_occ"]),
                    str(rec["sad_ex"]),
                    str(rec["not_sad_ex"]),
                    str(rec["examples_with_emoji"]),
                    str(rec["pct_sad_ex"]),
                    str(rec["pct_not_sad_ex"]),
                ]
            )
        )

    print()
    print("说明：")
    print("  详细字段见 JSONL；后处理时请跳过 kind=meta 或只读取 kind=emoji。")
    print("  sad_occ / not_occ — 该 emoji 在金标样本 input 中出现次数；")
    print("  sad_ex / not_ex   — 至少含过一次该 emoji 的样本条数；")
    print("  pct_* — 在该 emoji 子集内的比例（0–100）。")


if __name__ == "__main__":
    main()
