# 数据处理流程说明

## 数据处理流程概览

整个数据处理流程分为以下几个阶段：

```
原始数据 → 数据加载 → Parquet 转换 → Tokenization → 训练批次
```

## 详细流程

### 1. 原始数据阶段（dataset.py）

**处理内容**：从 HuggingFace 数据集或本地 JSONL 文件加载原始数据

**数据格式**：
- HuggingFace 数据集：`{"text": "文档内容", ...}`
- JSONL 文件：每行一个 JSON 对象，包含文本字段

**处理步骤**：
- 从 HuggingFace 下载或从本地加载数据集
- 提取 `text` 字段（或自动识别文本字段）
- 统一数据格式，只保留文本内容

**输出**：包含 `text` 列的 Dataset 对象

---

### 2. Parquet 转换阶段（dataset.py - `convert_to_parquet()`）

**处理内容**：将文本数据转换为 Parquet 格式，便于高效读取

**数据格式**：
- 输入：文本列表 `["文档1", "文档2", ...]`
- 输出：Parquet 文件，每行一个文档的文本内容

**处理步骤**：
- 将文本按字符数分片（默认每片 ~250M 字符）
- 写入 Parquet 文件（使用 Snappy 压缩）
- 文件命名：`shard_00000.parquet`, `shard_00001.parquet`, ...

**输出**：Parquet 文件目录 `parquet_shards/`

**关键代码位置**：
```python
# dataset.py:358-457
def convert_to_parquet(...):
    # 提取 text 字段
    text = example.get(text_column, "")
    # 写入 Parquet
    table = pa.Table.from_arrays([pa.array(texts)], names=["text"])
    pq.write_table(table, filepath, compression="snappy")
```

---

### 3. 数据加载阶段（dataloader.py - `parquets_iter_batched()`）

**处理内容**：从 Parquet 文件读取文本数据

**数据格式**：
- 输入：Parquet 文件
- 输出：文本批次 `["文档1", "文档2", ..., "文档128"]`

**处理步骤**：
- 按 DDP rank 分片读取（分布式训练）
- 从 Parquet 的 `text` 列读取文本
- 按 `tokenizer_batch_size`（默认 128）分批返回

**输出**：文本批次迭代器

**关键代码位置**：
```python
# dataset.py:475-504
def parquets_iter_batched(...):
    rg = pf.read_row_group(rg_idx)
    texts = rg.column('text').to_pylist()  # 读取 text 列
    yield texts
```

---

### 4. Tokenization 阶段（dataloader.py + tokenizer.py）⭐

**处理内容**：将文本转换为 token IDs（这是 tokenizer 处理的部分）

**数据格式**：
- 输入：文本批次 `["文档1", "文档2", ...]`
- 输出：Token ID 列表 `[[1, 234, 567, ...], [1, 890, 123, ...], ...]`

**处理步骤**：
1. **获取 tokenizer**：`tokenizer = get_tokenizer()`
2. **添加 BOS token**：在每个文档前添加 `<|bos|>` token
3. **编码文本**：`tokenizer.encode(doc_batch, prepend=bos_token)`
4. **返回 token IDs**：每个文档转换为一个 token ID 列表

**关键代码位置**：
```python
# dataloader.py:108-114
token_lists = tokenizer.encode(
    doc_batch,              # 文本批次：["文档1", "文档2", ...]
    prepend=bos_token,       # 在每个文档前添加 BOS token
    num_threads=tokenizer_threads
)
# 返回：[[1, 234, 567, ...], [1, 890, 123, ...], ...]
```

**Tokenizer 处理的具体内容**：
- ✅ **文本字符串** → **Token IDs**
- ✅ 使用 BPE（Byte Pair Encoding）算法
- ✅ 添加特殊 token（BOS、EOS 等）
- ✅ 处理 Unicode 字符和多语言文本

---

### 5. 批次构建阶段（dataloader.py）

**处理内容**：将 token IDs 组织成训练批次

**数据格式**：
- 输入：Token ID 流 `[1, 234, 567, 890, ...]`
- 输出：训练批次 `(inputs, targets)` tensors

**处理步骤**：
1. **累积 tokens**：从 token buffer 中取出 `B * T + 1` 个 tokens
2. **创建 inputs/targets**：
   - `inputs = tokens[:-1]` （前 B*T 个 tokens）
   - `targets = tokens[1:]` （后 B*T 个 tokens，用于预测）
3. **重塑形状**：`(B, T)` - B 个样本，每个 T 个 tokens
4. **移动到设备**：CPU → GPU（如果使用 CUDA）

**关键代码位置**：
```python
# dataloader.py:117-143
tokens = [token_buffer.popleft() for _ in range(needed_tokens)]  # B*T+1 个 tokens
inputs_cpu = scratch[:-1].to(dtype=torch.int32)  # 前 B*T 个
targets_cpu = scratch[1:]                         # 后 B*T 个
inputs = inputs_cpu.view(B, T).to(device=device)  # 重塑为 (B, T)
targets = targets_cpu.view(B, T).to(device=device)
```

---

## Tokenizer 处理的数据部分总结

### ✅ Tokenizer 处理的内容：

1. **文本字符串** → **Token IDs**
   - 输入：纯文本字符串（如 "Hello world"）
   - 输出：整数列表（如 `[15496, 1917]`）

2. **处理位置**：
   - 在 `dataloader.py` 的 `tokenizing_distributed_data_loader()` 函数中
   - 调用 `tokenizer.encode()` 方法

3. **处理时机**：
   - 从 Parquet 文件读取文本后
   - 在构建训练批次之前

4. **处理方式**：
   - 批量处理（默认 128 个文档一批）
   - 每个文档前添加 BOS token
   - 使用多线程加速（默认 4 个线程）

### ❌ Tokenizer 不处理的内容：

- ❌ 数据下载和加载（dataset.py）
- ❌ Parquet 文件转换（dataset.py）
- ❌ 批次构建和 tensor 操作（dataloader.py）
- ❌ GPU 内存管理（dataloader.py）

---

## 数据流示例

```
原始数据：
{"text": "这是一个测试文档。"}

↓ (dataset.py - convert_to_parquet)

Parquet 文件：
text: "这是一个测试文档。"

↓ (dataloader.py - parquets_iter_batched)

文本批次：
["这是一个测试文档。"]

↓ (dataloader.py + tokenizer.py - encode) ⭐ TOKENIZER 处理这里

Token IDs：
[[1, 234, 567, 890, 1234, 5678]]  # 1 是 BOS token

↓ (dataloader.py - 批次构建)

训练批次：
inputs:  [[1, 234, 567, 890, 1234]]      # shape: (B, T)
targets: [[234, 567, 890, 1234, 5678]]  # shape: (B, T)
```

---

## 关键文件说明

| 文件 | 职责 | 处理的数据部分 |
|------|------|---------------|
| `dataset.py` | 数据加载和转换 | 原始数据 → Parquet 文件 |
| `dataloader.py` | 数据加载和批次构建 | Parquet 文件 → 训练批次 |
| `tokenizer.py` | Tokenization | **文本字符串 → Token IDs** ⭐ |

---

## 总结

**Tokenizer 处理的是数据的 Tokenization 阶段**，具体来说：

1. **输入**：从 Parquet 文件读取的**文本字符串**
2. **处理**：使用 BPE 算法将文本转换为 **Token IDs**
3. **输出**：**Token ID 列表**，供后续批次构建使用

这是整个数据处理流程中的**关键步骤**，将人类可读的文本转换为模型可以处理的数字序列。
