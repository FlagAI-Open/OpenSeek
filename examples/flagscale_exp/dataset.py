"""
OpenSeek -> FlagScale 数据集转换脚本。

本脚本将 HuggingFace/OpenSeek 数据集转换为 FlagScale 支持的 JSONL 分片。
后续可配合 FlagScale 提供的 `tools/preprocess_data.py` 生成 Megatron/Energon
格式的数据。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "缺少 datasets 依赖，请先运行 `pip install datasets`"
    ) from exc


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 OpenSeek 数据集导出为 FlagScale 可用的 JSONL 分片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BAAI/OpenSeek-Pretrain-100B",
        help="HuggingFace 数据集名称或本地路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="需要导出的数据集切分",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="文本字段名称",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认为 ~/.cache/openseek_flagscale/jsonl",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=200_000,
        help="每个 JSONL 分片包含的样本数",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多导出的样本数（None 表示全部导出）",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="使用 HuggingFace streaming 模式",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="datasets 缓存目录",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "gzip"],
        default="none",
        help="是否对 JSONL 分片使用 gzip 压缩",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="加载远程数据集时允许执行自定义代码",
    )
    return parser


def resolve_output_dir(path: Optional[str]) -> Path:
    if path:
        out_dir = Path(path).expanduser()
    else:
        out_dir = Path.home() / ".cache" / "openseek_flagscale" / "jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def open_output_file(path: Path, compression: str):
    if compression == "gzip":
        import gzip

        return gzip.open(path, mode="wt", encoding="utf-8")
    return path.open(mode="w", encoding="utf-8")


def load_split_dataset(args) -> Iterable:
    dataset = load_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
    )
    return dataset


def convert_dataset(args) -> None:
    dataset = load_split_dataset(args)
    output_dir = resolve_output_dir(args.output_dir)

    print("=" * 60)
    print("OpenSeek -> FlagScale JSONL 导出")
    print("=" * 60)
    print(f"数据集：{args.dataset} ({args.split})")
    print(f"输出目录：{output_dir}")
    print(f"每分片样本数：{args.samples_per_shard}")
    print(f"最大样本数：{args.max_samples or '全部'}")
    print(f"压缩方式：{args.compression}")
    print("=" * 60)

    shard_index = 0
    num_written_total = 0
    num_written_current = 0
    file_handle = None

    try:
        for example in dataset:
            if args.max_samples is not None and num_written_total >= args.max_samples:
                break

            text = example.get(args.text_column, "")
            if not text:
                continue

            if file_handle is None or num_written_current >= args.samples_per_shard:
                if file_handle is not None:
                    file_handle.close()
                shard_name = f"shard_{shard_index:05d}.jsonl"
                if args.compression == "gzip":
                    shard_name += ".gz"
                file_path = output_dir / shard_name
                print(f"[write] {file_path}")
                file_handle = open_output_file(file_path, args.compression)
                shard_index += 1
                num_written_current = 0

            record = {"text": text}
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written_total += 1
            num_written_current += 1
    finally:
        if file_handle is not None:
            file_handle.close()

    print(f"\n导出完成，共写入 {num_written_total} 条样本，生成 {shard_index} 个分片。")


def main():
    parser = parse_args()
    args = parser.parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()


