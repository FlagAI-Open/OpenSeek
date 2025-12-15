#!/usr/bin/env python3
"""
Train a BPE tokenizer using HuggingFace tokenizers library (easy alternative to rustbpe).

This script trains a BPE tokenizer from OpenSeek parquet data files and saves it
in a format compatible with nanochat's tokenizer interface.

Usage:
    python -m examples.nanochat_exp.tok_train [options]

This is a drop-in replacement for nanochat's tok_train.py that doesn't require rustbpe.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Iterator, Optional, Tuple, List

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import BertProcessing
except ImportError:
    raise ImportError(
        "tokenizers library is required. Please install it with: pip install tokenizers\n"
        "This is much easier than installing rustbpe (no Rust compilation needed)."
    )

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow is required. Please install it with: pip install pyarrow")

# Import dataset utilities, but make them optional
try:
    from .dataset import list_parquet_files, get_data_dir
    DATASET_MODULE_AVAILABLE = True
except ImportError:
    # If dataset.py requires datasets library, provide fallback functions
    DATASET_MODULE_AVAILABLE = False
    import glob
    import os
    
    def get_data_dir():
        """Get the base directory for OpenSeek nanochat data."""
        base_dir = os.environ.get("OPENSEEK_NANOCHAT_DATA_DIR")
        if base_dir is None:
            # Default location
            base_dir = os.path.join(os.path.expanduser("~"), ".cache", "openseek_nanochat")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    def list_parquet_files(data_dir: str) -> List[str]:
        """List all parquet files in the directory."""
        parquet_pattern = os.path.join(data_dir, "*.parquet")
        files = sorted(glob.glob(parquet_pattern))
        return files


def get_tokenizer_dir() -> str:
    """Get the tokenizer directory."""
    base_dir = os.environ.get("NANOCHAT_BASE_DIR") or os.environ.get("OPENSEEK_NANOCHAT_DATA_DIR")
    if base_dir:
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
    else:
        # Default location
        tokenizer_dir = os.path.join(get_data_dir(), "tokenizer")
    
    os.makedirs(tokenizer_dir, exist_ok=True)
    return tokenizer_dir


def text_iterator(data_dir: Optional[str] = None, split: str = "train") -> Iterator[str]:
    """
    Iterate through text from parquet files.
    
    Args:
        data_dir: Directory containing parquet files (default: from dataset.get_data_dir())
        split: "train" or "val" (last file is used for validation)
    
    Yields:
        Text strings from the dataset
    """
    if data_dir is None:
        from .dataset import PARQUET_DIR
        data_dir = PARQUET_DIR
    
    parquet_paths = list_parquet_files(data_dir)
    
    if not parquet_paths:
        raise ValueError(
            f"No parquet files found in {data_dir}. "
            "Please run dataset conversion first: python -m examples.nanochat_exp.dataset"
        )
    
    # Use last file for validation, rest for training
    # If only one file exists, use it for training (no validation split)
    if split == "train":
        if len(parquet_paths) > 1:
            parquet_paths = parquet_paths[:-1]  # All except last
        # If only one file, use it for training
    else:
        if len(parquet_paths) > 1:
            parquet_paths = parquet_paths[-1:]  # Only last file
        else:
            # If only one file and we need validation, use it anyway
            parquet_paths = parquet_paths
    
    print(f"Reading {len(parquet_paths)} parquet file(s) for {split} split...")
    
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        print(f"  Processing {os.path.basename(filepath)} ({pf.num_row_groups} row groups)...")
        
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            
            # Check if 'text' column exists
            if 'text' not in rg.column_names:
                print(f"警告: Parquet 文件 {os.path.basename(filepath)} row group {rg_idx} 没有 'text' 列")
                print(f"可用列: {rg.column_names}")
                continue
            
            texts = rg.column('text').to_pylist()
            
            # Filter and validate texts
            for text in texts:
                if text is None:
                    continue
                # Convert to string if not already
                if not isinstance(text, str):
                    text = str(text)
                # Skip empty strings
                if text and len(text.strip()) > 0:
                    yield text


def train_tokenizer(
    vocab_size: int = 50257,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_frequency: int = 2,
    special_tokens: Optional[list] = None
) -> str:
    """
    Train a BPE tokenizer from parquet data files.
    
    Args:
        vocab_size: Vocabulary size for the tokenizer
        data_dir: Directory containing parquet files
        output_dir: Directory to save the tokenizer (default: tokenizer directory)
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens to add (default: BOS, EOS, PAD, UNK)
    
    Returns:
        Path to the saved tokenizer.json file
    """
    if special_tokens is None:
        special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]
    
    print("=" * 60)
    print("Training BPE Tokenizer with HuggingFace tokenizers")
    print("=" * 60)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Min frequency: {min_frequency}")
    print(f"Special tokens: {special_tokens}")
    print("=" * 60)
    print()
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Create text iterator
    print("Reading training data...")
    text_iter = text_iterator(data_dir=data_dir, split="train")
    
    # Train tokenizer
    print("Training tokenizer (this may take a while)...")
    tokenizer.train_from_iterator(text_iter, trainer=trainer)
    
    # Set post-processor (add BOS/EOS tokens)
    tokenizer.post_processor = BertProcessing(
        sep=("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        cls=("<|bos|>", tokenizer.token_to_id("<|bos|>"))
    )
    
    # Save tokenizer
    if output_dir is None:
        output_dir = get_tokenizer_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    
    print(f"\nSaving tokenizer to {tokenizer_path}...")
    tokenizer.save(tokenizer_path)
    
    # Also save tokenizer info
    info_path = os.path.join(output_dir, "tokenizer_info.json")
    tokenizer_info = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "model_type": "BPE",
        "pre_tokenizer": "ByteLevel",
        "bos_token_id": tokenizer.token_to_id("<|bos|>"),
        "eos_token_id": tokenizer.token_to_id("<|eos|>"),
        "pad_token_id": tokenizer.token_to_id("<|pad|>"),
        "unk_token_id": tokenizer.token_to_id("<|unk|>")
    }
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_info, f, indent=2, ensure_ascii=False)
    
    print(f"Tokenizer info saved to {info_path}")
    print(f"\n✓ Tokenizer training complete!")
    print(f"  Tokenizer file: {tokenizer_path}")
    print(f"  Vocabulary size: {vocab_size}")
    
    return tokenizer_path


def check_existing_tokenizer(output_dir: Optional[str] = None) -> Tuple[bool, Optional[int]]:
    """
    Check if a tokenizer already exists and return its vocab size.
    
    Returns:
        (exists, vocab_size) tuple
    """
    if output_dir is None:
        output_dir = get_tokenizer_dir()
    
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        return False, None
    
    try:
        from tokenizers import Tokenizer as HFTokenizer
        existing_tokenizer = HFTokenizer.from_file(tokenizer_path)
        
        # Try to get vocab size
        vocab_size = None
        if hasattr(existing_tokenizer, 'get_vocab_size'):
            vocab_size = existing_tokenizer.get_vocab_size()
        elif hasattr(existing_tokenizer, 'get_vocab'):
            vocab = existing_tokenizer.get_vocab()
            if vocab:
                vocab_size = len(vocab)
        elif hasattr(existing_tokenizer, 'model'):
            model = existing_tokenizer.model
            if hasattr(model, 'get_vocab_size'):
                vocab_size = model.get_vocab_size()
            elif hasattr(model, 'vocab') and model.vocab:
                vocab_size = len(model.vocab)
            elif hasattr(model, 'merges') and model.merges:
                # BPE: base (256) + merges + special tokens
                vocab_size = 256 + len(model.merges) + 4
        
        return True, vocab_size
    except Exception as e:
        print(f"警告: 检查现有 tokenizer 时出错: {e}")
        return True, None  # Assume exists but can't read


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer using HuggingFace tokenizers (no rustbpe needed)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size for the tokenizer (default: 50257)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing parquet files (default: from OPENSEEK_NANOCHAT_DATA_DIR)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the tokenizer (default: tokenizer directory)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token to be included (default: 2)"
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="+",
        default=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"],
        help="Special tokens to add (default: <|bos|> <|eos|> <|pad|> <|unk|>)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if tokenizer already exists"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check existing tokenizer, don't train"
    )
    
    args = parser.parse_args()
    
    # Get data directory
    if args.data_dir is None:
        if DATASET_MODULE_AVAILABLE:
            try:
                from .dataset import PARQUET_DIR
                args.data_dir = PARQUET_DIR
            except:
                # Fallback to default location
                args.data_dir = os.path.join(get_data_dir(), "parquet_shards")
        else:
            args.data_dir = os.path.join(get_data_dir(), "parquet_shards")
    
    # Check existing tokenizer
    exists, existing_vocab_size = check_existing_tokenizer(args.output_dir)
    
    if exists:
        if existing_vocab_size is not None:
            print(f"检测到现有 tokenizer，词汇表大小: {existing_vocab_size}")
            
            # Check if vocab size is suspiciously small (likely untrained)
            if existing_vocab_size <= 10:
                print(f"\n⚠️  警告: Tokenizer 的词汇表大小只有 {existing_vocab_size}，这通常表示 tokenizer 未正确训练。")
                print("   只有特殊 tokens，无法用于实际训练。")
                
                if args.check_only:
                    print("\n建议: 使用 --force 选项重新训练 tokenizer")
                    return 1 if existing_vocab_size <= 4 else 0
                
                if not args.force:
                    print("\n是否重新训练 tokenizer？")
                    try:
                        response = input("输入 'y' 重新训练，或按 Enter 跳过: ").strip().lower()
                        if response != 'y':
                            print("跳过训练。使用 --force 选项强制重新训练。")
                            return 1
                    except (EOFError, KeyboardInterrupt):
                        print("\n跳过训练。")
                        return 1
            else:
                if args.check_only:
                    print(f"✓ Tokenizer 看起来正常（词汇表大小: {existing_vocab_size}）")
                    return 0
                
                if not args.force:
                    print(f"\nTokenizer 已存在（词汇表大小: {existing_vocab_size}）")
                    print("如果词汇表大小正确，通常不需要重新训练。")
                    try:
                        response = input("是否重新训练？(y/N): ").strip().lower()
                        if response != 'y':
                            print("跳过训练。使用现有 tokenizer。")
                            return 0
                    except (EOFError, KeyboardInterrupt):
                        print("\n跳过训练。")
                        return 0
        else:
            print("检测到现有 tokenizer，但无法读取词汇表大小")
            if not args.force:
                try:
                    response = input("是否重新训练？(y/N): ").strip().lower()
                    if response != 'y':
                        print("跳过训练。")
                        return 0
                except (EOFError, KeyboardInterrupt):
                    print("\n跳过训练。")
                    return 0
    
    # Train tokenizer
    try:
        tokenizer_path = train_tokenizer(
            vocab_size=args.vocab_size,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            min_frequency=args.min_frequency,
            special_tokens=args.special_tokens
        )
        
        # Verify the trained tokenizer
        print("\n验证训练后的 tokenizer...")
        from tokenizers import Tokenizer as HFTokenizer
        trained_tokenizer = HFTokenizer.from_file(tokenizer_path)
        
        # Get actual vocab size
        actual_vocab_size = None
        if hasattr(trained_tokenizer, 'get_vocab_size'):
            actual_vocab_size = trained_tokenizer.get_vocab_size()
        elif hasattr(trained_tokenizer, 'get_vocab'):
            vocab = trained_tokenizer.get_vocab()
            if vocab:
                actual_vocab_size = len(vocab)
        
        if actual_vocab_size:
            print(f"✓ 训练完成！实际词汇表大小: {actual_vocab_size}")
            if actual_vocab_size <= 10:
                print("⚠️  警告: 词汇表大小仍然很小，可能训练有问题。")
                print("   请检查数据文件是否正确。")
        else:
            print("✓ 训练完成！")
        
        print(f"\n✓ Success! Tokenizer saved to: {tokenizer_path}")
        print("\nYou can now use this tokenizer with OpenSeek's data loader.")
    except Exception as e:
        print(f"\n✗ Error training tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
