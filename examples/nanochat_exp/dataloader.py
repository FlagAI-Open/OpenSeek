"""
Data loader for OpenSeek datasets in nanochat format.

This module provides a data loader compatible with nanochat's training pipeline,
using OpenSeek's converted parquet datasets.
"""

from collections import deque
import torch

# Import nanochat utilities (assuming nanochat is available in the environment)
try:
    from nanochat.common import get_dist_info
    from nanochat.tokenizer import get_tokenizer
except ImportError:
    # Fallback if nanochat is not directly importable
    import sys
    import os
    # Try to add nanochat to path if it exists
    nanochat_path = os.path.join(os.path.dirname(__file__), "../../../nanochat")
    if os.path.exists(nanochat_path):
        sys.path.insert(0, os.path.dirname(nanochat_path))
        from nanochat.common import get_dist_info
        from nanochat.tokenizer import get_tokenizer
    else:
        raise ImportError(
            "nanochat is required. Please install nanochat or ensure it's in your Python path."
        )

from .dataset import parquets_iter_batched

def tokenizing_distributed_data_loader(
    B: int,
    T: int,
    split: str = "train",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda"
):
    """
    Stream pretraining text from OpenSeek parquet files, tokenize, yield training batches.
    
    This function is compatible with nanochat's data loader interface.
    
    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        tokenizer_threads: Number of threads for tokenization
        tokenizer_batch_size: Batch size for tokenization
        device: Device to use ("cuda", "cpu", "mps")
    
    Yields:
        (inputs, targets) tuple of tensors
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1 for target at the last token
    
    # Get the tokenizer and BOS token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    
    # Scratch buffer holds tokens for one iteration
    token_buffer = deque()  # Stream tokens on the right, pop from the left
    
    # Infinite iterator over document batches
    def document_batches():
        while True:
            # Batch iterates in group size of parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(
                split=split,
                start=ddp_rank,
                step=ddp_world_size
            ):
                # For tokenizer, go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    
    batches = document_batches()
    batch_index = 0
    
    while True:
        # Accumulate enough tokens for one iteration before yielding
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(
                doc_batch,
                prepend=bos_token,
                num_threads=tokenizer_threads
            )
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        
        # Move tokens from deque into scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        # CUDA supports memory pinning for faster transfers
        scratch = torch.tensor(
            tokens,
            dtype=torch.int64,
            pin_memory=(device == "cuda")
        )
        
        # Create inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(
            device=device,
            dtype=torch.int32,
            non_blocking=True
        )
        targets = targets_cpu.view(B, T).to(
            device=device,
            dtype=torch.int64,
            non_blocking=True
        )
        
        yield inputs, targets

