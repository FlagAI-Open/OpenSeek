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
except ImportError:
    # Fallback if nanochat is not directly importable
    import sys
    import os
    # Try to add nanochat to path if it exists
    nanochat_path = os.path.join(os.path.dirname(__file__), "../../../nanochat")
    if os.path.exists(nanochat_path):
        sys.path.insert(0, os.path.dirname(nanochat_path))
        from nanochat.common import get_dist_info
    else:
        raise ImportError(
            "nanochat.common is required. Please install nanochat or ensure it's in your Python path."
        )

# Try to use our easy-to-install tokenizer first (uses HuggingFace tokenizers)
# Fallback to nanochat's rustbpe tokenizer if available
try:
    from .tokenizer import get_tokenizer
    _using_easy_tokenizer = True
except ImportError:
    # Fallback to nanochat's tokenizer
    try:
        from nanochat.tokenizer import get_tokenizer
        _using_easy_tokenizer = False
    except ImportError:
        import sys
        import os
        nanochat_path = os.path.join(os.path.dirname(__file__), "../../../nanochat")
        if os.path.exists(nanochat_path):
            sys.path.insert(0, os.path.dirname(nanochat_path))
            from nanochat.tokenizer import get_tokenizer
            _using_easy_tokenizer = False
        else:
            raise ImportError(
                "Either install tokenizers library (pip install tokenizers) or ensure nanochat tokenizer is available."
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
        print(f"[DataLoader] 开始加载数据 (rank={ddp_rank}, world_size={ddp_world_size})")
        try:
            from .dataset import list_parquet_files
            parquet_paths = list_parquet_files()
            print(f"[DataLoader] 找到 {len(parquet_paths)} 个 parquet 文件")
            if len(parquet_paths) == 0:
                raise ValueError(f"未找到 parquet 文件！请检查数据目录。")
        except Exception as e:
            print(f"[DataLoader] 错误: 无法列出 parquet 文件: {e}")
            raise
        
        batch_count = 0
        epoch_count = 0
        
        # Create the iterator once, reuse it across epochs
        parquet_iterator = None
        
        while True:
            # Only create iterator on first epoch or when it's exhausted
            if parquet_iterator is None:
                epoch_count += 1
                if epoch_count > 1:
                    print(f"[DataLoader] 开始第 {epoch_count} 个 epoch（重用数据迭代器）")
                parquet_iterator = parquets_iter_batched(
                    split=split,
                    start=ddp_rank,
                    step=ddp_world_size
                )
            
            # Batch iterates in group size of parquet files, usually e.g. 1024 rows
            try:
                batch = next(parquet_iterator)
                batch_count += 1
                if batch_count == 1:
                    print(f"[DataLoader] ✓ 成功加载第一个批次: {len(batch)} 个文档")
                # For tokenizer, go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
            except StopIteration:
                # Epoch finished, reset iterator for next epoch
                print(f"[DataLoader] 第 {epoch_count} 个 epoch 完成，准备开始下一个 epoch")
                parquet_iterator = None
                batch_count = 0
                continue
            except Exception as e:
                print(f"[DataLoader] 错误: 加载数据批次时出错: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    batches = document_batches()
    batch_index = 0
    
    print(f"[DataLoader] 初始化完成，等待第一个训练批次...")
    print(f"[DataLoader] 需要 tokens: {needed_tokens} (B={B}, T={T})")
    print(f"[DataLoader] 注意: 需要累积足够的 tokens 才会开始训练，这可能需要一些时间...")
    print(f"[DataLoader] CPU 会先进行数据加载和 tokenize，然后才会传输到 GPU")
    
    while True:
        # Accumulate enough tokens for one iteration before yielding
        while len(token_buffer) < needed_tokens:
            if batch_index == 0:
                print(f"[DataLoader] 开始获取第一个文档批次...")
            try:
                doc_batch = next(batches)
                if batch_index == 0:
                    print(f"[DataLoader] ✓ 获取到第一个文档批次: {len(doc_batch)} 个文档")
            except StopIteration:
                print(f"[DataLoader] 错误: 数据迭代器提前结束")
                raise
            except Exception as e:
                print(f"[DataLoader] 错误: 获取文档批次失败: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            if batch_index == 0:
                print(f"[DataLoader] 开始 tokenize 第一个批次...")
            try:
                token_lists = tokenizer.encode(
                    doc_batch,
                    prepend=bos_token,
                    num_threads=tokenizer_threads
                )
                if batch_index == 0:
                    total_tokens = sum(len(t) for t in token_lists)
                    print(f"[DataLoader] ✓ Tokenize 完成: {len(token_lists)} 个序列, {total_tokens} 个 tokens")
                    print(f"[DataLoader] Token buffer 当前大小: {len(token_buffer)}, 需要: {needed_tokens}")
            except Exception as e:
                print(f"[DataLoader] 错误: Tokenize 失败: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
            
            # 每处理一定数量的批次显示一次进度
            if batch_index % 10 == 0 or len(token_buffer) >= needed_tokens:
                progress_pct = min(100, len(token_buffer) * 100 // needed_tokens)
                print(f"[DataLoader] 进度: {batch_index} 个文档批次, "
                      f"Token buffer: {len(token_buffer)}/{needed_tokens} ({progress_pct}%)")
            
            if batch_index == 1:
                print(f"[DataLoader] Token buffer 更新后大小: {len(token_buffer)}")
        
        # Move tokens from deque into scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        if batch_index <= 2:
            print(f"[DataLoader] 准备创建第一个训练批次 (tokens: {len(tokens)})...")
        
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
        
        if batch_index <= 2:
            print(f"[DataLoader] ✓ 第一个训练批次已准备好: inputs.shape={inputs.shape}, targets.shape={targets.shape}")
            print(f"[DataLoader] 数据已移动到 {device}")
        
        yield inputs, targets


def tokenizing_distributed_data_loader_with_state(
    B: int,
    T: int,
    split: str = "train",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: dict = None
):
    """
    Stream pretraining text from OpenSeek parquet files, tokenize, yield training batches with state.
    
    This function supports approximate resume training by returning state_dict with every batch.
    The state_dict can be passed back via `resume_state_dict` to approximately resume from a checkpoint.
    
    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        tokenizer_threads: Number of threads for tokenization
        tokenizer_batch_size: Batch size for tokenization
        device: Device to use ("cuda", "cpu", "mps")
        resume_state_dict: Optional state dict to resume from (should contain 'pq_idx' and 'rg_idx')
    
    Yields:
        (inputs, targets, state_dict) tuple where state_dict contains {'pq_idx': int, 'rg_idx': int}
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1 for target at the last token
    
    # Get the tokenizer and BOS token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    
    # Scratch buffer holds tokens for one iteration
    token_buffer = deque()  # Stream tokens on the right, pop from the left
    
    # Import parquet utilities
    from .dataset import list_parquet_files
    import pyarrow.parquet as pq
    
    # Infinite iterator over document batches with state tracking
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx
        
        while True:  # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                
                # Start from resume point if resuming on same file, otherwise from DDP rank
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1  # advance by 1 to avoid repeating data
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    
                    # Check if 'text' column exists
                    if 'text' not in rg.column_names:
                        print(f"警告: Parquet 文件 {os.path.basename(filepath)} row group {rg_idx} 没有 'text' 列")
                        print(f"可用列: {rg.column_names}")
                        rg_idx += ddp_world_size
                        continue
                    
                    batch = rg.column('text').to_pylist()
                    
                    # Filter out None values and ensure strings
                    filtered_batch = []
                    for text in batch:
                        if text is None:
                            continue
                        if not isinstance(text, str):
                            text = str(text)
                        if text.strip():  # Skip empty strings
                            filtered_batch.append(text)
                    
                    # For tokenizer, go in smaller batches
                    for i in range(0, len(filtered_batch), tokenizer_batch_size):
                        yield filtered_batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size
                pq_idx += 1
            first_pass = False
    
    batches = document_batches()
    batch_index = 0
    
    while True:
        # Accumulate enough tokens for one iteration before yielding
        current_pq_idx = None
        current_rg_idx = None
        
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            current_pq_idx = pq_idx
            current_rg_idx = rg_idx
            
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
        
        # Create state dict for resume support
        state_dict = {
            "pq_idx": current_pq_idx if current_pq_idx is not None else 0,
            "rg_idx": current_rg_idx if current_rg_idx is not None else 0
        }
        
        yield inputs, targets, state_dict

