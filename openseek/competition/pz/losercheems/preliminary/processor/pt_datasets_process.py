# Copyright 2025 The SmallDoge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Optional, Union, Callable
import json
import logging
import warnings
import re
from datasets import Dataset, IterableDataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.data_utils import pack_dataset, truncate_dataset
from argparse import ArgumentParser


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_dataset_ratios(datasets_and_ratios: List[Dict[str, float]]) -> None:
    """Validate that dataset ratios are properly formatted and sum to 1.0."""
    if not datasets_and_ratios:
        raise ValueError("datasets_and_ratios cannot be empty")
    
    total_ratio = 0.0
    for dataset_dict in datasets_and_ratios:
        if not isinstance(dataset_dict, dict) or len(dataset_dict) != 1:
            raise ValueError("Each item in datasets_and_ratios must be a dictionary with exactly one key-value pair")
        
        ratio = list(dataset_dict.values())[0]
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            raise ValueError(f"Ratio must be a positive number, got {ratio}")
        
        total_ratio += ratio
    
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Total ratio must be 1.0, but got {total_ratio}. Please check your ratios.")


def validate_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Validate tokenizer configuration."""
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        logger.warning("Tokenizer has no pad_token, using eos_token as pad_token")
        tokenizer.pad_token = tokenizer.eos_token


def prepare_dataset(
    dataset: Union[Dataset, IterableDataset],
    dataset_name: str,
    dataset_text_field: str,
    processing_class: Union[PreTrainedTokenizerBase],
    max_length: Optional[int],
    packing: Optional[bool],
    formatting_func: Optional[Callable[[dict], str]],
    dataset_num_proc: Optional[int],
) -> Union[Dataset, IterableDataset]:

    # If the dataset is already preprocessed, skip the processing step
    column_names = list(next(iter(dataset)).keys())
    is_processed = "input_ids" in column_names

    # Build the kwargs for the `map` function
    map_kwargs = {}
    if isinstance(dataset, Dataset):  # InterableDataset does not support num_proc
        map_kwargs["num_proc"] = dataset_num_proc
    
    # Apply the formatting function if any
    if formatting_func is not None and is_processed:
        warnings.warn(
            "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
            "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
            "`formatting_func` or pass a dataset that is not already processed.",
            UserWarning,
        )
    
    if formatting_func is not None and not is_processed:
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

        batched = isinstance(formatting_func(next(iter(dataset))), list)

        def _func(example):
            return {"text": formatting_func(example)}

        dataset = dataset.map(_func, batched=batched, **map_kwargs)

            
    if not is_processed:

        # Tokenize the dataset if needed
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

        def tokenize(example, processing_class, dataset_text_field):
            try:
                processed = processing_class(text=example[dataset_text_field])
                if (
                    processing_class.eos_token_id is not None
                    and len(processed["input_ids"]) > 0
                    and processed["input_ids"][-1] != processing_class.eos_token_id
                ):
                    processed["input_ids"] = processed["input_ids"] + [processing_class.eos_token_id]
                    processed["attention_mask"] = processed["attention_mask"] + [1]
                return processed
            except Exception as e:
                logger.error(f"Error tokenizing example: {e}")
                # Return empty tokenization on error
                return {
                    "input_ids": [processing_class.eos_token_id] if processing_class.eos_token_id is not None else [],
                    "attention_mask": [1] if processing_class.eos_token_id is not None else []
                }

        dataset = dataset.map(
            tokenize,
            fn_kwargs={"processing_class": processing_class, "dataset_text_field": dataset_text_field},
            **map_kwargs,
        )

    # Pack or truncate
    if packing:
        if max_length is None:
            raise ValueError("When packing is enabled, `max_length` can't be `None`.")
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Packing {dataset_name} dataset"
        dataset = dataset.select_columns("input_ids")
        dataset = pack_dataset(dataset, seq_length=max_length, strategy="bfd", map_kwargs=map_kwargs)
    elif max_length is not None:
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
        dataset = truncate_dataset(dataset, max_length, map_kwargs)
    return dataset


def mix_datasets_by_ratio(
    datasets_and_ratios: List[Dict[str, float]],
    total_sample_size: int,
    dataset_text_field: str,
    processing_class: Union[PreTrainedTokenizerBase],
    max_length: Optional[int],
    packing: Optional[bool],
    formatting_func: Optional[Callable[[dict], str]],
    dataset_num_proc: Optional[int],
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """
    Mix multiple datasets by ratio.

    Args:
        datasets_and_ratios: List of dictionaries, each containing a dataset and its ratio.
            Each dictionary contains one key-value pair where key is the dataset name and value is the mixing ratio.
        total_sample_size: Total sample size for the mixed training dataset.
        dataset_text_field: Name of the field in the dataset that contains the text.
        processing_class: Tokenizer class used for processing the text.
        max_length: Maximum length of processed sequences. Set to None for no limit.
        packing: Whether to pack sequences for efficiency.
        formatting_func: Optional formatting function to convert dataset entries to the desired text format.
        dataset_num_proc: Number of processes to use for dataset processing.
        seed: Random seed for dataset shuffling to ensure reproducibility.
        cache_dir: Directory to cache the datasets.
    
    Returns:
        DatasetDict: A dictionary containing all mixed and processed dataset splits.
    
    Example:
    ```python
        from transformers import AutoTokenizer

        # Define datasets and their mixing ratios
        datasets_and_ratios = [
            {"SmallDoge/MiniCorpus:web-en": 0.5},
            {"SmallDoge/MiniCorpus:web-zh": 0.2},
            {"SmallDoge/MiniCorpus:textbook-en": 0.15},
            {"SmallDoge/MiniCorpus:textbook-zh": 0.05},
            {"SmallDoge/MiniCorpus:code": 0.05},
            {"SmallDoge/MiniCorpus:math": 0.05},
        ]

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")
    
        # Mix datasets
        mixed_dataset = mix_datasets_by_ratio(
            datasets_and_ratios=datasets_and_ratios,
            total_sample_size=10000,
            dataset_text_field="text",
            processing_class=tokenizer,
            max_length=2048,
            packing=True,
            formatting_func=None,
            dataset_num_proc=4,
            seed=42,
            cache_dir="./cache",
        )
        print(mixed_dataset)
    ```"""
    # Validate input parameters
    validate_dataset_ratios(datasets_and_ratios)
    
    # Check if the dataset ratios sum to 1.0 (redundant but kept for backwards compatibility)
    total_ratio = sum([list(dataset.values())[0] for dataset in datasets_and_ratios])
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Total ratio must be 1.0, but got {total_ratio}. Please check your ratios.")

    final_mixed_dataset = {}

    for dataset_and_ratio in datasets_and_ratios:
        dataset_name, ratio = dataset_and_ratio.popitem()

        # Check subset name
        windows_drive_pattern = r'^[a-zA-Z]:.*'
        is_windows_path = bool(re.match(windows_drive_pattern, dataset_name))
        if ":" in dataset_name and not is_windows_path:
            dataset_name, subset_name = dataset_name.split(":")
        else:
            subset_name = None

        # If the dataset is a string, load it from the hub or disk
        if isinstance(dataset_name, str):
            if re.match(r"^[^/]+/[^/]+$", dataset_name):
                dataset = load_dataset(dataset_name, name=subset_name, cache_dir=cache_dir)
            else:
                if subset_name is not None:
                    warnings.warn(
                        f"You passed a local dataset path, subsetting is not supported, ignoring subset name: {subset_name}",
                        UserWarning,
                    )
                dataset = load_from_disk(dataset_name)

        # Process each split of the dataset
        for split_name, split_dataset in dataset.items():
            split_dataset = split_dataset.select_columns(["input_ids"])
            logger.info(f"Original dataset size for {dataset_name}: {subset_name}: {split_name}: {len(split_dataset)}")
            # Process the dataset from text to input_ids
            split_dataset = prepare_dataset(
                split_dataset,
                dataset_name=f"{dataset_name}: {split_name}" if subset_name is None else f"{dataset_name}: {subset_name}: {split_name}",
                dataset_text_field=dataset_text_field,
                processing_class=processing_class,
                max_length=max_length,
                packing=packing,
                formatting_func=formatting_func,
                dataset_num_proc=dataset_num_proc,
            )

            # Calculate the target size for the dataset
            if total_sample_size == -1:
                target_size = len(split_dataset)
            else:
                target_size = int(total_sample_size * ratio) if split_name == "train" else len(split_dataset)
            current_size = len(split_dataset)
            logger.info(f"Processed dataset size for {dataset_name}: {split_name}: {current_size}")
            logger.info(f"Target size for {dataset_name}: {split_name}: {target_size}")

            # If the dataset is smaller than the target size, repeat it
            if current_size < target_size:
                logger.warning(
                    f"Dataset {dataset_name}: {split_name} is smaller than the target size. "
                    f"Repeating the dataset to reach the target size."
                )
                indices = []
                full_copies = target_size // current_size
                remainder = target_size % current_size

                for _ in range(full_copies):
                    indices.extend(range(current_size))
                if remainder > 0:
                    indices.extend(range(remainder))
                
                split_dataset = split_dataset.select(indices)
            else:
                logger.warning(
                    f"Dataset {dataset_name}: {split_name} is larger than the target size. "
                    f"Truncating the dataset to reach the target size."
                )
                split_dataset = split_dataset.select(range(target_size))
            
            # Concatenate the split dataset with the final mixed dataset
            if split_name in final_mixed_dataset:
                final_mixed_dataset[split_name] = concatenate_datasets(
                    [final_mixed_dataset[split_name], split_dataset]
                )
            else:
                final_mixed_dataset[split_name] = split_dataset
         
    # Shuffle the train dataset
    if "train" in final_mixed_dataset:
        final_mixed_dataset["train"] = final_mixed_dataset["train"].shuffle(seed)

    # Create a DatasetDict with the merged datasets
    final_dataset = DatasetDict(final_mixed_dataset)
    return final_dataset
            

def main(args):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    validate_tokenizer(tokenizer)

    # Mix datasets
    mixed_dataset = mix_datasets_by_ratio(
        datasets_and_ratios=args.datasets_and_ratios,
        total_sample_size=args.total_sample_size,
        dataset_text_field=args.dataset_text_field,
        processing_class=tokenizer,
        max_length=args.max_length,
        packing=args.packing,
        formatting_func=None,
        dataset_num_proc=args.dataset_num_proc,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    
    # Save the mixed dataset
    mixed_dataset.save_to_disk(args.dataset_save_path)
    print(f"Mixed dataset saved to {args.dataset_save_path}")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_and_ratios", type=str, required=True,
                          help="JSON string of list of dictionaries with dataset names and mixing ratios")
    argparser.add_argument("--dataset_save_path", type=str, required=True,
                          help="Path to save the mixed dataset")
    argparser.add_argument("--total_sample_size", type=int, required=True,
                          help="Total sample size for the mixed training dataset")
    argparser.add_argument("--dataset_text_field", type=str, required=True,
                          help="Name of the field in the dataset that contains the text")
    argparser.add_argument("--tokenizer_name_or_path", type=str, required=True,
                          help="Tokenizer name or path")
    argparser.add_argument("--max_length", type=int, default=2048,
                          help="Maximum length of processed sequences")
    argparser.add_argument("--packing", action="store_true",
                          help="Whether to pack sequences for efficiency")
    argparser.add_argument("--dataset_num_proc", type=int, default=4,
                          help="Number of processes for dataset processing")
    argparser.add_argument("--seed", type=int, default=42,
                          help="Random seed for reproducibility")
    argparser.add_argument("--cache_dir", type=str, default="./cache",
                          help="Directory to cache datasets")
    args = argparser.parse_args()

    # Parse datasets_and_ratios from JSON string
    args.datasets_and_ratios = json.loads(args.datasets_and_ratios)

    main(args)
