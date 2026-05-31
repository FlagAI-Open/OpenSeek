#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {exc}") from exc

            sample_id = obj.get("test_sample_id")
            prediction = obj.get("prediction")
            if sample_id is None or prediction is None:
                raise ValueError(
                    f"{path} 第 {line_no} 行缺少字段，必须包含 test_sample_id 和 prediction"
                )

            sample_id = str(sample_id)
            if sample_id in data:
                raise ValueError(f"{path} 存在重复 test_sample_id: {sample_id}")
            data[sample_id] = str(prediction)

    return data


def compare_predictions(
    v1: Dict[str, str], v2: Dict[str, str]
) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    ids1 = set(v1.keys())
    ids2 = set(v2.keys())

    only_v1 = sorted(ids1 - ids2)
    only_v2 = sorted(ids2 - ids1)
    common_ids = sorted(ids1 & ids2)

    changed: List[Tuple[str, str, str]] = []
    for sample_id in common_ids:
        p1 = v1[sample_id]
        p2 = v2[sample_id]
        if p1 != p2:
            changed.append((sample_id, p1, p2))

    return changed, only_v1, only_v2


def print_summary(
    v1_name: str,
    v2_name: str,
    v1: Dict[str, str],
    v2: Dict[str, str],
    changed: List[Tuple[str, str, str]],
    only_v1: List[str],
    only_v2: List[str],
    show_examples: int,
) -> None:
    ids1 = set(v1.keys())
    ids2 = set(v2.keys())
    common_count = len(ids1 & ids2)
    same_count = common_count - len(changed)
    diff_rate = (len(changed) / common_count * 100.0) if common_count else 0.0

    print("=== 对比结果 ===")
    print(f"{v1_name} 样本数: {len(v1)}")
    print(f"{v2_name} 样本数: {len(v2)}")
    print(f"共同样本数: {common_count}")
    print(f"预测一致数: {same_count}")
    print(f"预测不一致数: {len(changed)}")
    print(f"不一致比例: {diff_rate:.2f}%")
    print(f"仅在 {v1_name} 中出现: {len(only_v1)}")
    print(f"仅在 {v2_name} 中出现: {len(only_v2)}")

    if changed:
        pair_counter = Counter((old, new) for _, old, new in changed)
        print("\n=== 变化模式 Top 10 (v1 -> v2) ===")
        for (old, new), cnt in pair_counter.most_common(10):
            print(f"{old} -> {new}: {cnt}")

    if show_examples > 0 and changed:
        print(f"\n=== 不一致示例 (前 {min(show_examples, len(changed))} 条) ===")
        for sample_id, p1, p2 in changed[:show_examples]:
            print(f"{sample_id}: {v1_name}={p1}, {v2_name}={p2}")

    if show_examples > 0 and only_v1:
        print(f"\n=== 仅在 {v1_name} 中的示例 (前 {min(show_examples, len(only_v1))} 条) ===")
        for sample_id in only_v1[:show_examples]:
            print(sample_id)

    if show_examples > 0 and only_v2:
        print(f"\n=== 仅在 {v2_name} 中的示例 (前 {min(show_examples, len(only_v2))} 条) ===")
        for sample_id in only_v2[:show_examples]:
            print(sample_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比两个 JSONL 结果文件中的 prediction 差异。"
    )
    parser.add_argument(
        "--v1",
        type=Path,
        default=Path("outputs/openseek-3-v1.jsonl"),
        help="旧版本结果文件路径",
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=Path("outputs/openseek-3-v4.jsonl"),
        help="新版本结果文件路径",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=20,
        help="展示多少条差异示例（默认 20，设为 0 表示不展示）",
    )

    args = parser.parse_args()

    v1 = load_jsonl(args.v1)
    v2 = load_jsonl(args.v2)
    changed, only_v1, only_v2 = compare_predictions(v1, v2)
    print_summary(
        args.v1.name,
        args.v2.name,
        v1,
        v2,
        changed,
        only_v1,
        only_v2,
        show_examples=max(args.show_examples, 0),
    )


if __name__ == "__main__":
    main()
