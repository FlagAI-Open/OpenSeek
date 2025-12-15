#!/usr/bin/env python3
"""
查看 parquet 数据集的前几条数据示例。

Usage:
    python -m examples.nanochat_exp.inspect_parquet [--num-samples N] [--data-dir DIR]
"""

import os
import argparse
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    try:
        import pandas as pd
        USE_PANDAS = True
    except ImportError:
        print("错误: 需要安装 pyarrow 或 pandas")
        print("请运行以下命令之一:")
        print("  pip install pyarrow")
        print("  或")
        print("  pip install pandas")
        sys.exit(1)
    else:
        USE_PANDAS = True
else:
    USE_PANDAS = False

from .dataset import list_parquet_files, PARQUET_DIR


def inspect_parquet(data_dir: str = None, num_samples: int = 5, max_chars: int = 500):
    """
    查看 parquet 数据集的前几条数据。
    
    Args:
        data_dir: Parquet 文件目录
        num_samples: 显示多少条记录
        max_chars: 每条记录显示的最大字符数
    """
    if data_dir is None:
        data_dir = PARQUET_DIR
    
    parquet_paths = list_parquet_files(data_dir)
    
    if not parquet_paths:
        print(f"错误: 在 {data_dir} 中未找到 parquet 文件")
        return
    
    print("=" * 80)
    print("Parquet 数据集检查")
    print("=" * 80)
    print(f"数据目录: {data_dir}")
    print(f"找到 {len(parquet_paths)} 个 parquet 文件")
    print()
    
    # 读取第一个文件
    first_file = parquet_paths[0]
    print(f"检查文件: {os.path.basename(first_file)}")
    print()
    
    try:
        if USE_PANDAS:
            # 使用 pandas 读取
            import pandas as pd
            df = pd.read_parquet(first_file)
            print(f"文件信息:")
            print(f"  列名: {list(df.columns)}")
            print(f"  总行数: {len(df):,}")
            print()
            
            # 检查列是否存在
            if 'text' not in df.columns:
                print(f"错误: 文件没有 'text' 列")
                print(f"可用列: {list(df.columns)}")
                return
            
            texts = df['text'].tolist()
        else:
            # 使用 pyarrow 读取
            pf = pq.ParquetFile(first_file)
            print(f"文件信息:")
            print(f"  Row groups: {pf.num_row_groups}")
            print(f"  列名: {pf.schema.names}")
            print(f"  总行数: {pf.metadata.num_rows}")
            print()
            
            # 读取第一个 row group
            rg = pf.read_row_group(0)
            
            # 检查列是否存在
            if 'text' not in rg.column_names:
                print(f"错误: 文件没有 'text' 列")
                print(f"可用列: {rg.column_names}")
                return
            
            texts = rg.column('text').to_pylist()
        
        print(f"第一个 row group 包含 {len(texts)} 条记录")
        print()
        print("=" * 80)
        print(f"前 {min(num_samples, len(texts))} 条数据示例:")
        print("=" * 80)
        
        for i, text in enumerate(texts[:num_samples], 1):
            print(f"\n记录 {i}:")
            print(f"  长度: {len(text):,} 字符")
            
            # 显示文本内容
            if text:
                if isinstance(text, str):
                    preview = text[:max_chars]
                    print(f"  内容预览（前{max_chars}字符）:")
                    print(f"  {preview}")
                    if len(text) > max_chars:
                        print(f"  ... (还有 {len(text) - max_chars:,} 字符)")
                else:
                    print(f"  类型: {type(text)}")
                    print(f"  值: {text}")
            else:
                print("  (空文本)")
            
            print("-" * 80)
        
        # 统计信息
        print("\n统计信息:")
        if texts:
            text_lengths = [len(t) if isinstance(t, str) else 0 for t in texts]
            valid_lengths = [l for l in text_lengths if l > 0]
            
            if valid_lengths:
                print(f"  有效记录数: {len(valid_lengths)}/{len(texts)}")
                print(f"  最短文本: {min(valid_lengths):,} 字符")
                print(f"  最长文本: {max(valid_lengths):,} 字符")
                print(f"  平均长度: {sum(valid_lengths) / len(valid_lengths):,.0f} 字符")
                
                # 中位数
                sorted_lengths = sorted(valid_lengths)
                median_idx = len(sorted_lengths) // 2
                median = sorted_lengths[median_idx]
                print(f"  中位数长度: {median:,} 字符")
        
    except Exception as e:
        print(f"错误: 读取 parquet 文件失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="查看 parquet 数据集的前几条数据"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Parquet 文件目录（默认: 从环境变量或默认位置）"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="显示多少条记录（默认: 5）"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="每条记录显示的最大字符数（默认: 500）"
    )
    
    args = parser.parse_args()
    
    inspect_parquet(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        max_chars=args.max_chars
    )


if __name__ == "__main__":
    main()
