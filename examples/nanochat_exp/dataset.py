"""
OpenSeek dataset adapter for nanochat.

This module adapts OpenSeek datasets to work with nanochat's training pipeline.
It supports loading data from HuggingFace datasets (OpenSeek-Pretrain-100B) and
converting them to the parquet format expected by nanochat.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset, Dataset
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
        # Default to OpenSeek-Pretrain-100B in the project root
        project_root = Path(__file__).parent.parent.parent.parent
        base_dir = project_root / "OpenSeek-Pretrain-100B"
    else:
        base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)

DATA_DIR = get_data_dir()
PARQUET_DIR = os.path.join(DATA_DIR, "parquet_shards")
os.makedirs(PARQUET_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset loading and conversion

def load_openseek_dataset(
    dataset_name: str = "BAAI/OpenSeek-Pretrain-100B",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
):
    """
    Load OpenSeek dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        cache_dir: Directory to cache the dataset
        streaming: Whether to stream the dataset (for large datasets)
    
    Returns:
        Dataset object from HuggingFace datasets library
    """
    print(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True
        )
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load from local directory...")
        # Try loading from local directory if HuggingFace download fails
        if os.path.exists(DATA_DIR):
            # Try to load as parquet files
            return None
        raise

def convert_to_parquet(
    dataset,
    output_dir: str = PARQUET_DIR,
    shard_size: int = 250_000_000,  # ~250M chars per shard (similar to nanochat)
    text_column: str = "text",
    max_shards: int = -1,
):
    """
    Convert OpenSeek dataset to parquet format compatible with nanochat.
    
    Args:
        dataset: HuggingFace dataset or iterable
        output_dir: Directory to save parquet files
        shard_size: Approximate characters per shard
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
        filename = f"shard_{index:05d}.parquet"
        filepath = os.path.join(output_dir, filename)
        
        # Create parquet table
        table = pa.Table.from_arrays(
            [pa.array(texts)],
            names=["text"]
        )
        
        # Write to parquet
        pq.write_table(table, filepath, compression="snappy")
        print(f"Written shard {index}: {filename} ({len(texts)} documents, ~{current_shard_chars:,} chars)")
    
    print(f"Converting dataset to parquet format...")
    print(f"Output directory: {output_dir}")
    print(f"Target shard size: ~{shard_size:,} characters")
    
    # Handle streaming datasets
    if hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__"):
        # Streaming dataset
        for example in dataset:
            text = example.get(text_column, "")
            if not text or len(text.strip()) == 0:
                continue
            
            current_shard_texts.append(text)
            current_shard_chars += len(text)
            
            if current_shard_chars >= shard_size:
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
            
            text = example.get(text_column, "")
            if not text or len(text.strip()) == 0:
                continue
            
            current_shard_texts.append(text)
            current_shard_chars += len(text)
            
            if current_shard_chars >= shard_size:
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
    parquet_paths = list_parquet_files()
    
    if not parquet_paths:
        raise ValueError(
            f"No parquet files found in {PARQUET_DIR}. "
            "Please run the dataset conversion first."
        )
    
    # Use last file for validation, rest for training
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
# Main entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert OpenSeek dataset to nanochat-compatible parquet format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BAAI/OpenSeek-Pretrain-100B",
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
        help="Approximate characters per shard (default: 250M)"
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
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or PARQUET_DIR
    
    print("=" * 60)
    print("OpenSeek to nanochat Dataset Converter")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Shard size: ~{args.shard_size:,} characters")
    print(f"Max shards: {args.num_shards if args.num_shards > 0 else 'unlimited'}")
    print("=" * 60)
    print()
    
    # Load dataset
    try:
        dataset = load_openseek_dataset(
            dataset_name=args.dataset,
            streaming=args.streaming
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("\nPlease ensure:")
        print("1. The dataset is available on HuggingFace, or")
        print("2. You have downloaded the dataset locally to OpenSeek-Pretrain-100B/")
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
        text_column=args.text_column,
        max_shards=args.num_shards
    )
    elapsed = time.time() - start_time
    
    print(f"\nConversion completed in {elapsed:.2f} seconds")
    print(f"Total shards created: {num_shards}")

