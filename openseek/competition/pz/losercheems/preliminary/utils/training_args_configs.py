"""
Configuration classes for training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from transformers import TrainingArguments


@dataclass
class PTConfig(TrainingArguments):
    """
    Configuration for Pre-Training (PT).
    """
    
    # Dataset mixing parameters
    datasets_and_ratios: Optional[List[Dict[str, float]]] = field(
        default=None,
        metadata={"help": "List of datasets and their mixing ratios. Format: [{'dataset_name': ratio}, ...]"}
    )
    total_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of samples to use from mixed datasets"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "The field name containing text data in the dataset"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to pack sequences for efficient training"}
    )
    dataset_num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for dataset processing"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache processed datasets"}
    )
