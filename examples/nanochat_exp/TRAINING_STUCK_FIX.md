# 训练卡住问题解决方案

## 问题现象

训练时出现以下情况：
- ✅ 模型已初始化（GPU 显存占用 ~1GB）
- ✅ CPU 使用率很高（单进程）
- ❌ GPU 利用率为 0%（没有计算）
- ❌ 训练循环没有开始

## 根本原因

数据加载器需要**累积足够的 tokens** 才会 yield 第一个训练批次：

```
needed_tokens = B * T + 1
```

例如：
- `B = 32` (batch size)
- `T = 2048` (sequence length)  
- `needed_tokens = 32 * 2048 + 1 = 65,537`

数据加载器会：
1. 从 parquet 文件读取文档
2. 在 CPU 上 tokenize 文档
3. 累积 tokens 到 buffer
4. **只有当 buffer 中有 ≥ 65,537 个 tokens 时，才会 yield 第一个训练批次**

如果文档较短或数据较少，这个过程可能需要很长时间，导致：
- CPU 一直在 tokenize（单进程，CPU 占用高）
- GPU 在等待数据（GPU 利用率为 0%）

## 解决方案

### 1. 等待数据加载完成（推荐）

这是**正常现象**，只需要等待数据加载完成。现在已添加进度显示：

```
[DataLoader] 进度: 10 个文档批次, Token buffer: 5000/65537 (7%)
[DataLoader] 进度: 20 个文档批次, Token buffer: 12000/65537 (18%)
...
[DataLoader] ✓ 第一个训练批次已准备好
```

### 2. 优化数据加载速度

如果数据加载太慢，可以：

#### a) 增加 tokenizer 线程数
```python
# 在 nanochat 配置中增加 tokenizer_threads
tokenizer_threads = 8  # 默认是 4
```

#### b) 检查数据文件大小
```bash
# 检查 parquet 文件大小
ls -lh $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/ | head

# 如果文件太小，可能需要转换更多数据
python -m examples.nanochat_exp.dataset --dataset BAAI/OpenSeek-Pretrain-Data-Examples --num-shards -1
```

#### c) 使用更小的 batch size（临时测试）
```bash
# 使用更小的 batch size 可以更快看到第一个 batch
# 但这会影响训练效果，仅用于测试
export DEVICE_BATCH_SIZE=8  # 默认是 32
```

### 3. 检查数据是否正常

运行诊断脚本：
```bash
python -m examples.nanochat_exp.diagnose_training
```

检查：
- ✅ Parquet 文件是否存在且不为空
- ✅ Tokenizer 是否正常
- ✅ 数据加载是否正常

## 预期行为

正常的数据加载流程：

1. **模型初始化** (~5-10秒)
   - 加载模型权重到 GPU
   - 显存占用 ~1GB

2. **数据加载器初始化** (~1-2秒)
   - 列出 parquet 文件
   - 初始化 tokenizer

3. **Tokenize 和累积** (可能较慢，取决于数据)
   - CPU 进行 tokenize
   - 累积 tokens 到 buffer
   - **这是 CPU 占用高、GPU 空闲的阶段**

4. **第一个训练批次** (当 buffer 满了)
   - 数据传输到 GPU
   - GPU 开始计算
   - 训练循环开始

## 调试输出

现在数据加载器会显示详细的进度：

```
[DataLoader] 开始加载数据 (rank=0, world_size=1)
[DataLoader] 找到 10 个 parquet 文件
[parquets_iter_batched] 开始迭代 (split=train, start=0, step=1)
[parquets_iter_batched] 找到 10 个 parquet 文件
[parquets_iter_batched] 使用 9 个文件用于 train
[parquets_iter_batched] 处理文件 1/9: shard_00000.parquet
[parquets_iter_batched]   文件有 100 个 row groups
[parquets_iter_batched]   读取第一个 row group (idx=0)...
[parquets_iter_batched]   ✓ 第一个 row group: 1024 个原始文本, 1024 个有效文本
[DataLoader] ✓ 成功加载第一个批次: 1024 个文档
[DataLoader] ✓ 获取到第一个文档批次: 128 个文档
[DataLoader] 开始 tokenize 第一个批次...
[DataLoader] ✓ Tokenize 完成: 128 个序列, 50000 个 tokens
[DataLoader] Token buffer 当前大小: 0, 需要: 65537
[DataLoader] Token buffer 更新后大小: 50000
[DataLoader] 进度: 10 个文档批次, Token buffer: 50000/65537 (76%)
[DataLoader] 进度: 11 个文档批次, Token buffer: 65537/65537 (100%)
[DataLoader] 准备创建第一个训练批次 (tokens: 65537)...
[DataLoader] ✓ 第一个训练批次已准备好: inputs.shape=torch.Size([32, 2048]), targets.shape=torch.Size([32, 2048])
[DataLoader] 数据已移动到 cuda
```

## 如果仍然卡住

如果等待很长时间（>5分钟）仍然没有进展，检查：

1. **数据文件是否为空**
   ```bash
   python -c "import pyarrow.parquet as pq; pf = pq.ParquetFile('$OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/shard_00000.parquet'); print(f'Row groups: {pf.num_row_groups}')"
   ```

2. **文档是否太短**
   - 如果文档平均长度 < 100 tokens，需要很多文档才能累积到 65,537 tokens

3. **进程是否真的在运行**
   ```bash
   ps aux | grep python
   top -p <pid>
   ```

4. **查看完整日志**
   - 检查是否有错误信息
   - 查看调试输出，看卡在哪一步

## 总结

**CPU 占用高、GPU 空闲是数据加载阶段的正常现象**。只需要等待数据加载完成，GPU 就会开始计算。现在已添加进度显示，可以清楚地看到数据加载的进度。
