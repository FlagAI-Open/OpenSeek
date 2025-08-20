#!/usr/bin/env python3
"""
PyEdu Dataset Utilities for OpenSeek

This module provides utilities for downloading, preprocessing, and integrating
the PyEdu dataset (educational Python code from smollm-corpus) into OpenSeek
training pipelines.

PyEdu Dataset: https://huggingface.co/datasets/Leon-Leee/unofficial-pyedu
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")


class PyEduDatasetHandler:
    """Handler for PyEdu dataset operations."""
    
    DATASET_NAME = "Leon-Leee/unofficial-pyedu"
    DATASET_SIZE_GB = 6
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the PyEdu dataset handler.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/openseek/pyedu")
        self.dataset_path = None
        
    def download_dataset(self, output_dir: str) -> str:
        """Download the PyEdu dataset from Hugging Face.
        
        Args:
            output_dir: Directory to save the dataset
            
        Returns:
            Path to the downloaded dataset
            
        Raises:
            ImportError: If required libraries are not available
            RuntimeError: If download fails
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
            
        print(f"Downloading PyEdu dataset ({self.DATASET_SIZE_GB}GB) to {output_dir}")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Download dataset
            dataset = load_dataset(
                self.DATASET_NAME,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Save dataset in JSON format for preprocessing
            output_file = os.path.join(output_dir, "pyedu_raw.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for split_name, split_data in dataset.items():
                    print(f"Processing split: {split_name}")
                    for example in split_data:
                        # Extract text content (adjust key based on actual dataset structure)
                        text_content = example.get('text', example.get('content', ''))
                        if text_content:
                            json.dump({'text': text_content}, f, ensure_ascii=False)
                            f.write('\n')
            
            print(f"Dataset saved to: {output_file}")
            self.dataset_path = output_file
            return output_file
            
        except Exception as e:
            raise RuntimeError(f"Failed to download PyEdu dataset: {e}")
    
    def preprocess_for_training(self, 
                              input_file: str, 
                              output_prefix: str,
                              tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                              workers: int = 4,
                              chunk_size: int = 1000) -> Tuple[str, str]:
        """Preprocess PyEdu dataset for OpenSeek training.
        
        Args:
            input_file: Path to raw PyEdu JSONL file
            output_prefix: Prefix for output files
            tokenizer_name: Name of tokenizer to use
            workers: Number of worker processes
            chunk_size: Chunk size for processing
            
        Returns:
            Tuple of (binary_file_path, index_file_path)
        """
        # Import preprocessing script
        sys.path.append(os.path.dirname(__file__))
        from preprocess_data_args import main as preprocess_main
        
        # Prepare arguments for preprocessing
        preprocess_args = [
            '--input', input_file,
            '--json-keys', 'text',
            '--split-sentences',
            '--fill-in-middle',  # Enable FIM for code data
            '--fill-in-middle-percentage', '15',  # Higher percentage for code
            '--model-name', tokenizer_name,
            '--model-dir', os.path.join(self.cache_dir, 'tokenizers'),
            '--output-prefix', output_prefix,
            '--workers', str(workers),
            '--chunk-size', str(chunk_size),
            '--dataset-impl', 'mmap'
        ]
        
        # Save original sys.argv and replace with our arguments
        original_argv = sys.argv
        sys.argv = ['preprocess_data_args.py'] + preprocess_args
        
        try:
            print("Preprocessing PyEdu dataset for training...")
            preprocess_main()
            
            # Return paths to generated files
            bin_file = f"{output_prefix}_text_sentence.bin"
            idx_file = f"{output_prefix}_text_sentence.idx"
            
            return bin_file, idx_file
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
    
    def validate_dataset(self, dataset_path: str) -> Dict[str, any]:
        """Validate the PyEdu dataset and return statistics.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_examples': 0,
            'total_characters': 0,
            'avg_length': 0,
            'file_size_mb': 0,
            'contains_python_code': False
        }
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Get file size
        stats['file_size_mb'] = os.path.getsize(dataset_path) / (1024 * 1024)
        
        # Analyze content
        python_indicators = ['def ', 'import ', 'class ', 'if __name__', 'print(']
        python_code_count = 0
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    
                    stats['total_examples'] += 1
                    stats['total_characters'] += len(text)
                    
                    # Check for Python code indicators
                    if any(indicator in text for indicator in python_indicators):
                        python_code_count += 1
                        
                except json.JSONDecodeError:
                    continue
        
        if stats['total_examples'] > 0:
            stats['avg_length'] = stats['total_characters'] / stats['total_examples']
            stats['contains_python_code'] = python_code_count > 0
            stats['python_code_percentage'] = (python_code_count / stats['total_examples']) * 100
        
        return stats
    
    def create_training_config(self, 
                             dataset_path: str, 
                             output_config: str,
                             base_config: Optional[str] = None) -> str:
        """Create a training configuration that includes PyEdu dataset.
        
        Args:
            dataset_path: Path to preprocessed PyEdu dataset
            output_config: Path for output configuration file
            base_config: Optional base configuration to extend
            
        Returns:
            Path to created configuration file
        """
        # This would create a YAML configuration similar to what we created manually
        # For now, we'll create a simple template
        
        config_template = f"""# PyEdu Dataset Training Configuration
# Generated automatically by pyedu_dataset_utils.py

data:
  # PyEdu dataset integration
  data_path:
    # PyEdu dataset - high-quality educational Python code
    - 1.0  # Weight for PyEdu dataset
    - {dataset_path}
    
  split: 1
  no_mmap_bin_files: true
  tokenizer:
    tokenizer_type: QwenTokenizerFS
    tokenizer_path: ../hf_openseek/tokenizer
    vocab_size: 151851
    make_vocab_size_divisible_by: 64

# Note: This is a minimal configuration focusing on PyEdu dataset.
# For complete training, merge with existing model and system configurations.
"""
        
        os.makedirs(os.path.dirname(output_config), exist_ok=True)
        with open(output_config, 'w') as f:
            f.write(config_template)
        
        print(f"Training configuration created: {output_config}")
        return output_config


def main():
    """Main CLI interface for PyEdu dataset utilities."""
    parser = argparse.ArgumentParser(
        description="PyEdu Dataset Utilities for OpenSeek",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download PyEdu dataset
  python pyedu_dataset_utils.py download --output-dir ./data/pyedu

  # Preprocess for training
  python pyedu_dataset_utils.py preprocess --input ./data/pyedu/pyedu_raw.jsonl --output-prefix ./data/pyedu/pyedu

  # Validate dataset
  python pyedu_dataset_utils.py validate --dataset-path ./data/pyedu/pyedu_raw.jsonl

  # Create training config
  python pyedu_dataset_utils.py create-config --dataset-path ./data/pyedu/pyedu_text_sentence --output-config ./configs/pyedu_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download PyEdu dataset')
    download_parser.add_argument('--output-dir', required=True, help='Output directory for dataset')
    download_parser.add_argument('--cache-dir', help='Cache directory for downloads')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess dataset for training')
    preprocess_parser.add_argument('--input', required=True, help='Input JSONL file')
    preprocess_parser.add_argument('--output-prefix', required=True, help='Output file prefix')
    preprocess_parser.add_argument('--tokenizer-name', default='Qwen/Qwen2.5-Coder-7B-Instruct', help='Tokenizer name')
    preprocess_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    preprocess_parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--dataset-path', required=True, help='Path to dataset file')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create training configuration')
    config_parser.add_argument('--dataset-path', required=True, help='Path to preprocessed dataset')
    config_parser.add_argument('--output-config', required=True, help='Output configuration file')
    config_parser.add_argument('--base-config', help='Base configuration to extend')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    handler = PyEduDatasetHandler(cache_dir=getattr(args, 'cache_dir', None))
    
    try:
        if args.command == 'download':
            dataset_path = handler.download_dataset(args.output_dir)
            print(f"✓ PyEdu dataset downloaded successfully: {dataset_path}")
            
        elif args.command == 'preprocess':
            bin_file, idx_file = handler.preprocess_for_training(
                args.input, args.output_prefix, args.tokenizer_name,
                args.workers, args.chunk_size
            )
            print(f"✓ Dataset preprocessed successfully:")
            print(f"  Binary file: {bin_file}")
            print(f"  Index file: {idx_file}")
            
        elif args.command == 'validate':
            stats = handler.validate_dataset(args.dataset_path)
            print("✓ Dataset validation results:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        elif args.command == 'create-config':
            config_path = handler.create_training_config(
                args.dataset_path, args.output_config, args.base_config
            )
            print(f"✓ Training configuration created: {config_path}")
            
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()