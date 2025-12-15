"""
OpenSeek dataset adapter for nanochat.

This module adapts OpenSeek datasets to work with nanochat's training pipeline.
It supports loading data from HuggingFace datasets (OpenSeek-Pretrain-Data-Examples) and
converting them to the parquet format expected by nanochat.
"""

import os
import argparse
import time
import subprocess
import shutil
import json
import glob
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset, Dataset, concatenate_datasets
except ImportError:
    raise ImportError(
        "Please install required dependencies: pip install pyarrow datasets"
    )

# -----------------------------------------------------------------------------
# Configuration

def get_data_dir():
    """Get the base directory for OpenSeek nanochat data."""
    base_dir = os.environ.get("OPENSEEK_NANOCHAT_DATA_DIR")
    if base_dir is None:
        # Default to OpenSeek-Pretrain-Data-Examples in the project root
        project_root = Path(__file__).parent.parent.parent.parent
        base_dir = project_root / "OpenSeek-Pretrain-Data-Examples"
    else:
        base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)

DATA_DIR = get_data_dir()
PARQUET_DIR = os.path.join(DATA_DIR, "parquet_shards")
os.makedirs(PARQUET_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset downloading using huggingface-cli

def download_dataset_with_hf_cli(
    dataset_name: str = "BAAI/OpenSeek-Pretrain-Data-Examples",
    local_dir: Optional[str] = None,
    hf_endpoint: str = "https://hf-mirror.com",
    token: Optional[str] = None,
):
    """
    使用 huggingface-cli 下载数据集（通过 hf-mirror.com 镜像）。
    
    Args:
        dataset_name: HuggingFace 数据集标识符
        local_dir: 本地保存目录（默认使用 DATA_DIR）
        hf_endpoint: HuggingFace 镜像端点（默认使用 hf-mirror.com）
        token: HuggingFace token（用于需要认证的数据集）
    
    Returns:
        下载后的本地目录路径
    """
    if local_dir is None:
        local_dir = DATA_DIR
    
    # 检查 huggingface-cli 是否可用
    hf_cli_path = shutil.which("huggingface-cli")
    if hf_cli_path is None:
        raise RuntimeError(
            "huggingface-cli 未找到。\n"
            "请先安装: pip install -U huggingface_hub\n"
            "推荐使用虚拟环境以避免权限问题。"
        )
    
    print(f"使用 huggingface-cli 下载数据集: {dataset_name}")
    print(f"镜像端点: {hf_endpoint}")
    print(f"本地目录: {local_dir}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["HF_ENDPOINT"] = hf_endpoint
    
    # 构建命令
    cmd = [
        hf_cli_path,
        "download",
        "--repo-type", "dataset",
        "--resume-download",
        dataset_name,
        "--local-dir", local_dir,
        "--local-dir-use-symlinks", "False",  # 禁用软链接，所见即所得
    ]
    
    # 如果需要 token，添加认证参数
    if token:
        cmd.extend(["--token", token])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行下载
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # 显示实时输出
        )
        print(f"\n数据集下载完成: {local_dir}")
        return local_dir
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"下载数据集失败，退出码: {e.returncode}\n"
            f"请检查网络连接和数据集名称是否正确。"
        ) from e

# -----------------------------------------------------------------------------
# Dataset loading and conversion

def load_jsonl_files_manually(data_dir: str, text_column: str = "text"):
    """
    手动加载 JSONL 文件，统一列结构以处理列不匹配问题。
    
    Args:
        data_dir: 包含 JSONL 文件的目录
        text_column: 文本列的名称
    
    Returns:
        Dataset 对象
    """
    jsonl_files = []
    # 递归查找所有 JSONL 文件
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    
    if not jsonl_files:
        raise ValueError(f"在 {data_dir} 中未找到 JSONL 文件")
    
    print(f"找到 {len(jsonl_files)} 个 JSONL 文件")
    
    all_examples = []
    total_chars = 0
    skipped_lines = 0
    for jsonl_file in jsonl_files:
        print(f"加载文件: {jsonl_file}")
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                        # 统一列结构：提取 text 字段
                        # 处理不同的列名变体，确保提取大段文本
                        text = ""
                        
                        def extract_text_from_value(value, max_depth=3, current_depth=0):
                            """递归提取文本内容，支持嵌套结构"""
                            if current_depth > max_depth:
                                return ""
                            
                            if isinstance(value, str):
                                return value if value.strip() else ""
                            elif isinstance(value, dict):
                                # 如果是字典，递归查找文本字段
                                text_parts = []
                                for key, val in value.items():
                                    if key.lower() in ["text", "content", "body", "data", "article", "document"]:
                                        extracted = extract_text_from_value(val, max_depth, current_depth + 1)
                                        if extracted:
                                            text_parts.append(extracted)
                                    elif isinstance(val, str) and len(val.strip()) > 100:  # 长文本字段
                                        text_parts.append(val)
                                return "\n".join(text_parts) if text_parts else ""
                            elif isinstance(value, list):
                                # 如果是列表，合并所有字符串元素
                                text_parts = []
                                for item in value:
                                    extracted = extract_text_from_value(item, max_depth, current_depth + 1)
                                    if extracted:
                                        text_parts.append(extracted)
                                return "\n".join(text_parts) if text_parts else ""
                            return ""
                        
                        # 优先尝试常见的文本字段名（按优先级排序）
                        text_candidates = [
                            text_column,  # 用户指定的列名
                            "text", "content", "body", "data",  # 常见文本字段
                            "article", "document", "passage", "paragraph",  # 文档类字段
                            "input", "output", "prompt", "response",  # 对话类字段
                            "source", "target", "translation"  # 翻译类字段
                        ]
                        
                        for candidate in text_candidates:
                            if candidate in example:
                                candidate_value = example[candidate]
                                # 支持嵌套结构
                                extracted_text = extract_text_from_value(candidate_value)
                                if extracted_text and len(extracted_text.strip()) > 0:
                                    text = extracted_text
                                    break
                                # 也支持直接的字符串值
                                elif isinstance(candidate_value, str) and len(candidate_value.strip()) > 0:
                                    text = candidate_value
                                    break
                        
                        # 如果还没找到，尝试找到最长的字符串字段作为文本
                        if not text:
                            longest_text = ""
                            longest_length = 0
                            for key, value in example.items():
                                # 跳过明显的元数据字段
                                if key.lower() in ["id", "url", "language", "metadata", "warc_record_id", 
                                                   "timestamp", "date", "author", "title", "source"]:
                                    continue
                                
                                # 尝试提取文本
                                extracted = extract_text_from_value(value)
                                if extracted and len(extracted) > longest_length:
                                    longest_text = extracted
                                    longest_length = len(extracted)
                            
                            if longest_text:
                                text = longest_text
                        
                        # 如果仍然没找到，尝试合并所有非元数据的字符串字段
                        if not text:
                            text_parts = []
                            for key, value in example.items():
                                # 跳过元数据字段
                                if key.lower() in ["id", "url", "language", "metadata", "warc_record_id",
                                                   "timestamp", "date", "author", "source"]:
                                    continue
                                
                                extracted = extract_text_from_value(value)
                                if extracted:
                                    text_parts.append(extracted)
                            
                            if text_parts:
                                # 合并文本，使用换行符分隔不同字段
                                text = "\n".join(text_parts)
                        
                        # 确保提取到的是完整的文本内容
                        if text:
                            # 移除多余的空白字符，但保留段落结构
                            text = text.strip()
                            # 如果文本很长，确保是完整的（不是截断的）
                            if len(text) > 0:
                                # 创建统一的示例，只保留 text 字段
                                unified_example = {text_column: text}
                                all_examples.append(unified_example)
                                total_chars += len(text)
                                
                                # 每处理 10000 条记录显示一次统计
                                if len(all_examples) % 10000 == 0:
                                    avg_length = total_chars / len(all_examples)
                                    print(f"  已处理 {len(all_examples):,} 条记录，平均文本长度: {avg_length:.0f} 字符")
                            else:
                                skipped_lines += 1
                        else:
                            skipped_lines += 1
                    except json.JSONDecodeError as e:
                        print(f"警告: 跳过 {jsonl_file} 第 {line_num} 行的无效 JSON: {e}")
                        skipped_lines += 1
                        continue
        except Exception as e:
            print(f"警告: 加载文件 {jsonl_file} 时出错: {e}")
            continue
    
    if not all_examples:
        raise ValueError("未能从 JSONL 文件中加载任何数据")
    
    avg_text_length = total_chars / len(all_examples) if all_examples else 0
    print(f"\n总共加载了 {len(all_examples):,} 个示例")
    print(f"总字符数: {total_chars:,}")
    print(f"平均文本长度: {avg_text_length:.0f} 字符")
    if skipped_lines > 0:
        print(f"跳过的无效/空记录: {skipped_lines:,}")
    
    # 显示一些示例的文本长度分布
    if all_examples:
        text_lengths = [len(ex[text_column]) for ex in all_examples[:1000]]  # 采样前1000条
        if text_lengths:
            min_len = min(text_lengths)
            max_len = max(text_lengths)
            median_len = sorted(text_lengths)[len(text_lengths) // 2]
            print(f"文本长度统计（前1000条样本）:")
            print(f"  最短: {min_len} 字符")
            print(f"  最长: {max_len:,} 字符")
            print(f"  中位数: {median_len} 字符")
    
    return Dataset.from_list(all_examples)


def load_openseek_dataset(
    dataset_name: str = "BAAI/OpenSeek-Pretrain-Data-Examples",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    local_dir: Optional[str] = None,
):
    """
    Load OpenSeek dataset from HuggingFace or local directory.
    
    Args:
        dataset_name: HuggingFace dataset identifier or local path
        cache_dir: Directory to cache the dataset
        streaming: Whether to stream the dataset (for large datasets)
        local_dir: Local directory path (if dataset is already downloaded)
    
    Returns:
        Dataset object from HuggingFace datasets library
    """
    # 如果提供了本地目录，优先使用本地数据
    if local_dir and os.path.exists(local_dir):
        print(f"从本地目录加载数据集: {local_dir}")
        
        # 检查是否有 JSONL 文件
        jsonl_files = list(Path(local_dir).rglob("*.jsonl"))
        if jsonl_files:
            print(f"检测到 {len(jsonl_files)} 个 JSONL 文件，使用手动加载方式以避免列不匹配问题")
            try:
                dataset = load_jsonl_files_manually(local_dir, text_column="text")
                return dataset
            except Exception as e:
                print(f"手动加载 JSONL 文件失败: {e}")
                print("尝试使用标准方式加载...")
        
        try:
            # 尝试使用 verification_mode 跳过列验证（如果支持）
            try:
                dataset = load_dataset(
                    str(local_dir),
                    cache_dir=cache_dir,
                    streaming=streaming,
                    verification_mode="no_checks"  # 跳过列验证
                )
                return dataset
            except TypeError:
                # 如果不支持 verification_mode，尝试普通加载
                dataset = load_dataset(
                    str(local_dir),
                    cache_dir=cache_dir,
                    streaming=streaming
                )
                return dataset
        except Exception as e:
            error_msg = str(e)
            # 检查是否是列不匹配错误
            if "columns" in error_msg.lower() or "matching columns" in error_msg.lower():
                print(f"检测到列不匹配错误，尝试手动加载 JSONL 文件...")
                try:
                    dataset = load_jsonl_files_manually(local_dir, text_column="text")
                    return dataset
                except Exception as e2:
                    print(f"手动加载也失败: {e2}")
            print(f"从本地目录加载失败: {e}")
            print("尝试从 HuggingFace 加载...")
    
    # 检查 DATA_DIR 是否存在（可能是通过 huggingface-cli 下载的）
    if os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR):
        # 检查是否包含数据集文件（通常会有 data-*.arrow 或 parquet 文件）
        has_data_files = any(
            f.endswith('.arrow') or f.endswith('.parquet') or f.endswith('.jsonl')
            for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
        )
        if has_data_files:
            print(f"从本地目录加载数据集: {DATA_DIR}")
            
            # 检查是否有 JSONL 文件
            jsonl_files = list(Path(DATA_DIR).rglob("*.jsonl"))
            if jsonl_files:
                print(f"检测到 {len(jsonl_files)} 个 JSONL 文件，使用手动加载方式以避免列不匹配问题")
                try:
                    dataset = load_jsonl_files_manually(DATA_DIR, text_column="text")
                    return dataset
                except Exception as e:
                    print(f"手动加载 JSONL 文件失败: {e}")
            
            try:
                # 尝试使用 verification_mode 跳过列验证（如果支持）
                try:
                    dataset = load_dataset(
                        str(DATA_DIR),
                        cache_dir=cache_dir,
                        streaming=streaming,
                        verification_mode="no_checks"
                    )
                    return dataset
                except TypeError:
                    dataset = load_dataset(
                        str(DATA_DIR),
                        cache_dir=cache_dir,
                        streaming=streaming
                    )
                    return dataset
            except Exception as e:
                error_msg = str(e)
                # 检查是否是列不匹配错误
                if "columns" in error_msg.lower() or "matching columns" in error_msg.lower():
                    print(f"检测到列不匹配错误，尝试手动加载 JSONL 文件...")
                    try:
                        dataset = load_jsonl_files_manually(DATA_DIR, text_column="text")
                        return dataset
                    except Exception as e2:
                        print(f"手动加载也失败: {e2}")
                print(f"从本地目录加载失败: {e}")
    
    # 尝试从 HuggingFace 加载
    print(f"从 HuggingFace 加载数据集: {dataset_name}")
    try:
        # 设置 HF_ENDPOINT 环境变量（如果未设置）
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"设置 HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
        
        # 尝试使用 verification_mode（如果支持）
        try:
            dataset = load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                streaming=streaming,
                verification_mode="no_checks"
            )
            return dataset
        except TypeError:
            # 如果不支持 verification_mode，使用普通加载
            dataset = load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                streaming=streaming
            )
            return dataset
    except Exception as e:
        error_msg = str(e)
        # 检查是否是列不匹配错误
        if "columns" in error_msg.lower() or "matching columns" in error_msg.lower():
            print(f"检测到列不匹配错误: {e}")
            print("提示：如果数据集已下载到本地，脚本会自动尝试手动加载 JSONL 文件")
        print(f"从 HuggingFace 加载失败: {e}")
        print("\n提示：可以使用以下命令先下载数据集：")
        print(f"  export HF_ENDPOINT=https://hf-mirror.com")
        print(f"  huggingface-cli download --repo-type dataset --resume-download {dataset_name} --local-dir {DATA_DIR}")
        raise

def convert_to_parquet(
    dataset,
    output_dir: str = PARQUET_DIR,
    shard_size: int = 250_000_000,  # ~250M chars per shard (similar to nanochat)
    rows_per_shard: int = -1,  # Maximum rows per shard (-1 to disable row limit)
    text_column: str = "text",
    max_shards: int = -1,
):
    """
    Convert OpenSeek dataset to parquet format compatible with nanochat.
    
    Args:
        dataset: HuggingFace dataset or iterable
        output_dir: Directory to save parquet files
        shard_size: Approximate characters per shard
        rows_per_shard: Maximum rows per shard (-1 to disable, use shard_size instead)
        text_column: Name of the text column in the dataset
        max_shards: Maximum number of shards to create (-1 for all)
    
    Returns:
        Number of shards created
    """
    os.makedirs(output_dir, exist_ok=True)
    
    shard_index = 0
    current_shard_texts = []
    current_shard_chars = 0
    
    def write_shard(texts: List[str], index: int):
        """Write a single parquet shard."""
        if not texts:
            print(f"警告: Shard {index} 没有文本数据，跳过写入")
            return
        
        filename = f"shard_{index:05d}.parquet"
        filepath = os.path.join(output_dir, filename)
        
        # Filter out None values and ensure all are strings
        filtered_texts = []
        for text in texts:
            if text is None:
                continue
            # Convert to string if not already
            if not isinstance(text, str):
                text = str(text)
            # Skip empty strings
            if text.strip():
                filtered_texts.append(text)
        
        if not filtered_texts:
            print(f"警告: Shard {index} 过滤后没有有效文本数据，跳过写入")
            return
        
        # Create parquet table
        try:
            table = pa.Table.from_arrays(
                [pa.array(filtered_texts, type=pa.string())],
                names=["text"]
            )
            
            # Write to parquet
            pq.write_table(table, filepath, compression="snappy")
            print(f"Written shard {index}: {filename} ({len(filtered_texts)} documents, ~{current_shard_chars:,} chars)")
        except Exception as e:
            print(f"错误: 写入 shard {index} 失败: {e}")
            raise
    
    print(f"Converting dataset to parquet format...")
    print(f"Output directory: {output_dir}")
    if rows_per_shard > 0:
        print(f"Target shard size: {rows_per_shard} rows per shard")
    else:
        print(f"Target shard size: ~{shard_size:,} characters")
    
    # Handle streaming datasets
    if hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__"):
        # Streaming dataset
        for example in dataset:
            # Get text from example
            text = example.get(text_column, "") if isinstance(example, dict) else ""
            
            # Convert to string if not already
            if text is not None and not isinstance(text, str):
                text = str(text)
            
            # Skip empty or invalid text
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                continue
            
            current_shard_texts.append(text)
            current_shard_chars += len(text)
            
            # Check if we should write shard (by rows or by size)
            should_write = False
            if rows_per_shard > 0:
                # Write when reaching row limit
                if len(current_shard_texts) >= rows_per_shard:
                    should_write = True
            else:
                # Write when reaching character limit
                if current_shard_chars >= shard_size:
                    should_write = True
            
            if should_write:
                write_shard(current_shard_texts, shard_index)
                shard_index += 1
                current_shard_texts = []
                current_shard_chars = 0
                
                if max_shards > 0 and shard_index >= max_shards:
                    break
    else:
        # Regular dataset
        if isinstance(dataset, dict):
            # DatasetDict, use train split
            dataset = dataset.get("train", list(dataset.values())[0])
        
        total_examples = len(dataset) if hasattr(dataset, "__len__") else None
        print(f"Processing {total_examples:,} examples..." if total_examples else "Processing examples...")
        
        for idx, example in enumerate(dataset):
            if idx % 10000 == 0 and total_examples:
                print(f"Progress: {idx:,}/{total_examples:,} examples")
            
            # Get text from example
            if isinstance(example, dict):
                text = example.get(text_column, "")
            elif hasattr(example, text_column):
                text = getattr(example, text_column, "")
            else:
                text = ""
            
            # Convert to string if not already
            if text is not None and not isinstance(text, str):
                text = str(text)
            
            # Skip empty or invalid text
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                continue
            
            current_shard_texts.append(text)
            current_shard_chars += len(text)
            
            # Check if we should write shard (by rows or by size)
            should_write = False
            if rows_per_shard > 0:
                # Write when reaching row limit
                if len(current_shard_texts) >= rows_per_shard:
                    should_write = True
            else:
                # Write when reaching character limit
                if current_shard_chars >= shard_size:
                    should_write = True
            
            if should_write:
                write_shard(current_shard_texts, shard_index)
                shard_index += 1
                current_shard_texts = []
                current_shard_chars = 0
                
                if max_shards > 0 and shard_index >= max_shards:
                    break
    
    # Write final shard if there's remaining data
    if current_shard_texts:
        write_shard(current_shard_texts, shard_index)
        shard_index += 1
    
    print(f"\nConversion complete! Created {shard_index} shards in {output_dir}")
    return shard_index

# -----------------------------------------------------------------------------
# Parquet file utilities (compatible with nanochat interface)

def list_parquet_files(data_dir: Optional[str] = None):
    """List all parquet files in the data directory (compatible with nanochat)."""
    data_dir = PARQUET_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split: str = "train", start: int = 0, step: int = 1):
    """
    Iterate through parquet files in batches (compatible with nanochat).
    
    Args:
        split: "train" or "val" (last file is used for validation)
        start: Starting index for DDP
        step: Step size for DDP
    
    Yields:
        Batches of text documents
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    # Only print initialization message once (use a static flag or check if first call)
    # We'll use a simpler approach: only print detailed info on first file
    parquet_paths = list_parquet_files()
    
    if not parquet_paths:
        raise ValueError(
            f"No parquet files found in {PARQUET_DIR}. "
            "Please run the dataset conversion first."
        )
    
    # Use last file for validation, rest for training
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    
    # Only print detailed info once (check if this is likely the first call)
    # We'll use a simple heuristic: check if we're at the start
    _first_call = not hasattr(parquets_iter_batched, '_initialized')
    if _first_call:
        print(f"[parquets_iter_batched] 开始迭代 (split={split}, start={start}, step={step})")
        print(f"[parquets_iter_batched] 找到 {len(parquet_paths)} 个 parquet 文件用于 {split}")
        parquets_iter_batched._initialized = True
    
    file_count = 0
    for filepath in parquet_paths:
        file_count += 1
        if _first_call and file_count == 1:
            print(f"[parquets_iter_batched] 处理文件 {file_count}/{len(parquet_paths)}: {os.path.basename(filepath)}")
        pf = pq.ParquetFile(filepath)
        if _first_call and file_count == 1:
            print(f"[parquets_iter_batched]   文件有 {pf.num_row_groups} 个 row groups")
        
        rg_count = 0
        for rg_idx in range(start, pf.num_row_groups, step):
            rg_count += 1
            if _first_call and file_count == 1 and rg_count == 1:
                print(f"[parquets_iter_batched]   读取第一个 row group (idx={rg_idx})...")
            rg = pf.read_row_group(rg_idx)
            
            # Check if 'text' column exists
            if 'text' not in rg.column_names:
                raise ValueError(
                    f"Parquet file {filepath} row group {rg_idx} does not have 'text' column. "
                    f"Available columns: {rg.column_names}"
                )
            
            texts = rg.column('text').to_pylist()
            
            # Filter out None values and ensure strings
            filtered_texts = []
            for text in texts:
                if text is None:
                    continue
                if not isinstance(text, str):
                    text = str(text)
                if text.strip():  # Skip empty strings
                    filtered_texts.append(text)
            
            if _first_call and file_count == 1 and rg_count == 1:
                print(f"[parquets_iter_batched]   ✓ 第一个 row group: {len(texts)} 个原始文本, {len(filtered_texts)} 个有效文本")
            
            if filtered_texts:
                yield filtered_texts

# -----------------------------------------------------------------------------
# Main entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert OpenSeek dataset to nanochat-compatible parquet format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BAAI/OpenSeek-Pretrain-Data-Examples",
        help="HuggingFace dataset name or local path"
    )
    parser.add_argument(
        "-n", "--num-shards",
        type=int,
        default=-1,
        help="Number of shards to create (-1 for all available data)"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=250_000_000,
        help="Approximate characters per shard (default: 250M, ignored if --rows-per-shard is set)"
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=-1,
        help="Maximum rows per shard (default: -1 to disable, use shard-size instead)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for parquet files (default: DATA_DIR/parquet_shards)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets"
    )
    parser.add_argument(
        "--download-with-cli",
        action="store_true",
        help="使用 huggingface-cli 下载数据集（通过 hf-mirror.com 镜像）"
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HuggingFace 镜像端点（默认: https://hf-mirror.com）"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token（用于需要认证的数据集）"
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or PARQUET_DIR
    
    print("=" * 60)
    print("OpenSeek to nanochat Dataset Converter")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    if args.rows_per_shard > 0:
        print(f"Shard size: {args.rows_per_shard} rows per shard")
    else:
        print(f"Shard size: ~{args.shard_size:,} characters")
    print(f"Max shards: {args.num_shards if args.num_shards > 0 else 'unlimited'}")
    print("=" * 60)
    print()
    
    # 如果指定使用 huggingface-cli 下载
    if args.download_with_cli:
        try:
            download_dataset_with_hf_cli(
                dataset_name=args.dataset,
                local_dir=DATA_DIR,
                hf_endpoint=args.hf_endpoint,
                token=args.hf_token,
            )
            print("\n下载完成，开始加载数据集...")
        except Exception as e:
            print(f"下载数据集失败: {e}")
            exit(1)
    
    # Load dataset
    # 如果 dataset_name 是本地路径，直接使用它作为 local_dir
    local_dir_to_use = None
    if os.path.exists(args.dataset) and os.path.isdir(args.dataset):
        local_dir_to_use = args.dataset
        print(f"检测到本地目录路径: {local_dir_to_use}")
    elif args.download_with_cli:
        local_dir_to_use = DATA_DIR
    
    try:
        dataset = load_openseek_dataset(
            dataset_name=args.dataset,
            streaming=args.streaming,
            local_dir=local_dir_to_use,
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Failed to load dataset: {e}")
        
        # 如果是列不匹配错误，尝试直接使用手动加载
        if "columns" in error_msg.lower() or "matching columns" in error_msg.lower():
            print("\n检测到列不匹配错误，尝试直接手动加载 JSONL 文件...")
            # 尝试从 DATA_DIR 或 args.dataset 加载
            jsonl_dir = local_dir_to_use if local_dir_to_use else (args.dataset if os.path.exists(args.dataset) else DATA_DIR)
            if os.path.exists(jsonl_dir):
                try:
                    print(f"从 {jsonl_dir} 手动加载 JSONL 文件...")
                    dataset = load_jsonl_files_manually(jsonl_dir, text_column=args.text_column)
                    print("手动加载成功！")
                except Exception as e2:
                    print(f"手动加载也失败: {e2}")
                    print("\n请确保:")
                    print("1. 数据集在 HuggingFace 上可用，或")
                    print("2. 已使用 huggingface-cli 下载数据集到本地")
                    print("\n建议使用以下命令下载:")
                    print(f"  export HF_ENDPOINT={args.hf_endpoint}")
                    print(f"  huggingface-cli download --repo-type dataset --resume-download {args.dataset} --local-dir {DATA_DIR}")
                    exit(1)
            else:
                print("\n请确保:")
                print("1. 数据集在 HuggingFace 上可用，或")
                print("2. 已使用 huggingface-cli 下载数据集到本地")
                print("\n建议使用以下命令下载:")
                print(f"  export HF_ENDPOINT={args.hf_endpoint}")
                print(f"  huggingface-cli download --repo-type dataset --resume-download {args.dataset} --local-dir {DATA_DIR}")
                exit(1)
        else:
            print("\n请确保:")
            print("1. 数据集在 HuggingFace 上可用，或")
            print("2. 已使用 huggingface-cli 下载数据集到本地")
            print("\n建议使用以下命令下载:")
            print(f"  export HF_ENDPOINT={args.hf_endpoint}")
            print(f"  huggingface-cli download --repo-type dataset --resume-download {args.dataset} --local-dir {DATA_DIR}")
            exit(1)
    
    if dataset is None:
        print("No dataset loaded. Please check your dataset path.")
        exit(1)
    
    # Convert to parquet
    start_time = time.time()
    num_shards = convert_to_parquet(
        dataset,
        output_dir=output_dir,
        shard_size=args.shard_size,
        rows_per_shard=args.rows_per_shard,
        text_column=args.text_column,
        max_shards=args.num_shards
    )
    elapsed = time.time() - start_time
    
    print(f"\nConversion completed in {elapsed:.2f} seconds")
    print(f"Total shards created: {num_shards}")

