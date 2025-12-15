# Parquet Shard 行数限制功能

## 功能说明

现在数据转换支持限制每个 parquet shard 的行数，这样可以更好地控制数据文件的大小和训练时的数据加载速度。

## 使用方法

### 1. 通过环境变量设置（推荐）

在运行 `run_openseek_exp.sh` 时设置环境变量：

```bash
# 设置每个 shard 为 2048 行（默认值）
export ROWS_PER_SHARD=2048
bash run_openseek_exp.sh
```

### 2. 通过命令行参数设置

直接调用数据转换脚本：

```bash
python -m examples.nanochat_exp.dataset \
    --dataset BAAI/OpenSeek-Pretrain-Data-Examples \
    --rows-per-shard 2048 \
    --num-shards -1
```

### 3. 禁用行数限制（使用字符数限制）

如果想使用原来的字符数限制方式：

```bash
# 方法1: 设置 ROWS_PER_SHARD 为 -1
export ROWS_PER_SHARD=-1
bash run_openseek_exp.sh

# 方法2: 直接调用脚本时不指定 --rows-per-shard（默认使用字符数限制）
python -m examples.nanochat_exp.dataset \
    --dataset BAAI/OpenSeek-Pretrain-Data-Examples \
    --shard-size 250000000 \
    --num-shards -1
```

## 参数说明

- `--rows-per-shard`: 每个 parquet shard 的最大行数
  - 默认值: `-1`（禁用，使用字符数限制）
  - 推荐值: `2048`（已设置为脚本默认值）
  - 如果设置为 `-1`，则使用 `--shard-size` 参数（字符数限制）

- `--shard-size`: 每个 shard 的近似字符数（仅在 `rows_per_shard=-1` 时生效）
  - 默认值: `250000000` (250M 字符)

## 优势

1. **更精确的控制**: 按行数限制比按字符数更精确，每个 shard 的大小更一致
2. **更快的训练启动**: 较小的 shard 可以更快地加载第一个训练批次
3. **更好的内存管理**: 较小的 shard 占用更少内存

## 示例

### 示例 1: 使用默认的 2048 行限制

```bash
# 使用脚本默认值（2048 行）
bash run_openseek_exp.sh
```

### 示例 2: 自定义行数

```bash
# 每个 shard 512 行（更小的 shard）
export ROWS_PER_SHARD=512
bash run_openseek_exp.sh

# 每个 shard 4096 行（更大的 shard）
export ROWS_PER_SHARD=4096
bash run_openseek_exp.sh
```

### 示例 3: 使用字符数限制（旧方式）

```bash
# 禁用行数限制，使用字符数限制
export ROWS_PER_SHARD=-1
export SHARD_SIZE=100000000  # 100M 字符
bash run_openseek_exp.sh
```

## 注意事项

1. **行数 vs 字符数**: 
   - 如果设置了 `rows_per_shard > 0`，会优先使用行数限制
   - 如果 `rows_per_shard = -1`，则使用字符数限制

2. **数据过滤**: 
   - 空文本会被跳过，不计入行数
   - 只有有效的文本行才会被写入 shard

3. **最后一个 shard**: 
   - 最后一个 shard 可能少于指定的行数（如果数据不够）

4. **训练性能**: 
   - 较小的 shard（如 2048 行）可以更快地开始训练
   - 但会产生更多的 shard 文件

## 验证

转换完成后，可以检查 shard 的行数：

```bash
# 检查第一个 shard 的行数
python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('$OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/shard_00000.parquet')
print(f'Row groups: {pf.num_row_groups}')
for i in range(pf.num_row_groups):
    rg = pf.read_row_group(i)
    print(f'Row group {i}: {len(rg)} rows')
"
```
