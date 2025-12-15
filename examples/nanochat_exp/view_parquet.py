#!/usr/bin/env python3
"""
独立脚本：查看 parquet 数据集的前几条数据示例。

不依赖其他模块，可以直接运行。

Usage:
    python examples/nanochat_exp/view_parquet.py [--num-samples N] [--file FILE]
"""

import os
import sys
import argparse

def view_parquet(parquet_file: str, num_samples: int = 3, max_chars: int = 800):
    """查看 parquet 文件的前几条数据"""
    
    if not os.path.exists(parquet_file):
        print(f"错误: 文件不存在: {parquet_file}")
        return
    
    print("=" * 80)
    print("Parquet 数据集查看")
    print("=" * 80)
    print(f"文件: {parquet_file}")
    print(f"文件大小: {os.path.getsize(parquet_file) / 1024 / 1024:.2f} MB")
    print()
    
    # 尝试使用 pyarrow
    try:
        import pyarrow.parquet as pq
        
        pf = pq.ParquetFile(parquet_file)
        print(f"文件信息:")
        print(f"  Row groups: {pf.num_row_groups}")
        print(f"  列名: {pf.schema.names}")
        print(f"  总行数: {pf.metadata.num_rows:,}")
        print()
        
        # 读取第一个 row group
        rg = pf.read_row_group(0)
        
        if 'text' not in rg.column_names:
            print(f"错误: 文件没有 'text' 列")
            print(f"可用列: {rg.column_names}")
            return
        
        texts = rg.column('text').to_pylist()
        
        print(f"第一个 row group 包含 {len(texts):,} 条记录")
        print()
        print("=" * 80)
        print(f"前 {min(num_samples, len(texts))} 条数据示例:")
        print("=" * 80)
        
        for i, text in enumerate(texts[:num_samples], 1):
            print(f"\n记录 {i}:")
            print(f"  长度: {len(text):,} 字符")
            
            if text:
                if isinstance(text, str):
                    preview = text[:max_chars]
                    print(f"  内容预览（前{max_chars}字符）:")
                    print(f"  {preview}")
                    if len(text) > max_chars:
                        print(f"  ... (还有 {len(text) - max_chars:,} 字符)")
                else:
                    print(f"  类型: {type(text)}")
                    print(f"  值: {str(text)[:max_chars]}")
            else:
                print("  (空文本)")
            
            print("-" * 80)
        
        # 统计信息
        if texts:
            text_lengths = [len(t) if isinstance(t, str) else 0 for t in texts]
            valid_lengths = [l for l in text_lengths if l > 0]
            
            if valid_lengths:
                print(f"\n统计信息（第一个 row group）:")
                print(f"  有效记录数: {len(valid_lengths):,}/{len(texts):,}")
                print(f"  最短文本: {min(valid_lengths):,} 字符")
                print(f"  最长文本: {max(valid_lengths):,} 字符")
                print(f"  平均长度: {sum(valid_lengths) / len(valid_lengths):,.0f} 字符")
                
                sorted_lengths = sorted(valid_lengths)
                median_idx = len(sorted_lengths) // 2
                median = sorted_lengths[median_idx]
                print(f"  中位数长度: {median:,} 字符")
        
    except ImportError:
        # 尝试使用 pandas
        try:
            import pandas as pd
            
            df = pd.read_parquet(parquet_file)
            print(f"文件信息:")
            print(f"  列名: {list(df.columns)}")
            print(f"  总行数: {len(df):,}")
            print()
            
            if 'text' not in df.columns:
                print(f"错误: 文件没有 'text' 列")
                print(f"可用列: {list(df.columns)}")
                return
            
            texts = df['text'].tolist()
            
            print("=" * 80)
            print(f"前 {min(num_samples, len(texts))} 条数据示例:")
            print("=" * 80)
            
            for i, text in enumerate(texts[:num_samples], 1):
                print(f"\n记录 {i}:")
                print(f"  长度: {len(text):,} 字符")
                
                if text:
                    if isinstance(text, str):
                        preview = text[:max_chars]
                        print(f"  内容预览（前{max_chars}字符）:")
                        print(f"  {preview}")
                        if len(text) > max_chars:
                            print(f"  ... (还有 {len(text) - max_chars:,} 字符)")
                    else:
                        print(f"  类型: {type(text)}")
                        print(f"  值: {str(text)[:max_chars]}")
                else:
                    print("  (空文本)")
                
                print("-" * 80)
            
            # 统计信息
            if texts:
                text_lengths = [len(t) if isinstance(t, str) else 0 for t in texts]
                valid_lengths = [l for l in text_lengths if l > 0]
                
                if valid_lengths:
                    print(f"\n统计信息:")
                    print(f"  有效记录数: {len(valid_lengths):,}/{len(texts):,}")
                    print(f"  最短文本: {min(valid_lengths):,} 字符")
                    print(f"  最长文本: {max(valid_lengths):,} 字符")
                    print(f"  平均长度: {sum(valid_lengths) / len(valid_lengths):,.0f} 字符")
                    
                    sorted_lengths = sorted(valid_lengths)
                    median_idx = len(sorted_lengths) // 2
                    median = sorted_lengths[median_idx]
                    print(f"  中位数长度: {median:,} 字符")
        
        except ImportError:
            print("错误: 需要安装 pyarrow 或 pandas")
            print("请运行以下命令之一:")
            print("  pip install pyarrow")
            print("  或")
            print("  pip install pandas")
            sys.exit(1)
    
    except Exception as e:
        print(f"错误: 读取 parquet 文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="查看 parquet 数据集的前几条数据"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Parquet 文件路径（默认: ~/.cache/openseek_nanochat/parquet_shards/shard_00000.parquet）"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="显示多少条记录（默认: 3）"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="每条记录显示的最大字符数（默认: 800）"
    )
    
    args = parser.parse_args()
    
    if args.file:
        parquet_file = args.file
    else:
        # 默认路径
        parquet_file = os.path.expanduser(
            "~/.cache/openseek_nanochat/parquet_shards/shard_00000.parquet"
        )
    
    view_parquet(parquet_file, args.num_samples, args.max_chars)


if __name__ == "__main__":
    main()
