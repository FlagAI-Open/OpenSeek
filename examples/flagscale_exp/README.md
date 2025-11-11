# OpenSeek × FlagScale 实验示例

本目录提供将 OpenSeek 数据集引入 [FlagScale](https://github.com/FlagOpen/FlagScale) 训练流水线的示例说明与数据预处理脚本，便于使用 FlagScale 的混合并行能力进行大模型训练。

## 目录结构

```
examples/flagscale_exp/
├── __init__.py          # Python 包入口
├── dataset.py           # OpenSeek -> FlagScale JSONL 分片转换脚本
└── README.md            # 使用说明
```

## 前置依赖

1. **克隆 FlagScale 仓库**
   ```bash
   git clone https://github.com/FlagOpen/FlagScale.git ~/workspace/FlagScale
   ```
2. **安装 FlagScale 环境**  
   按照 `OpenSeek/docs/FlagScale_Usage.md` 或 FlagScale 官方文档完成依赖安装（推荐使用官方 `install/install-requirements.sh` 脚本）。
3. **准备 OpenSeek 数据**  
   - 默认从 HuggingFace 自动下载 `BAAI/OpenSeek-Pretrain-100B`。
   - 也可以传入本地数据集路径（需包含 `text` 字段）。

## 步骤一：生成 JSONL 分片

使用本目录提供的 `dataset.py` 将 OpenSeek 数据转换为 FlagScale 可读取的 JSONL 分片（可选 gzip 压缩）。

```bash
python -m examples.flagscale_exp.dataset \
  --dataset "BAAI/OpenSeek-Pretrain-100B" \
  --split train \
  --samples-per-shard 200000 \
  --output-dir ~/data/openseek_flagscale/jsonl \
  --compression gzip \
  --streaming
```

常用参数：

| 参数 | 说明 |
| ---- | ---- |
| `--dataset` | HuggingFace 数据集名或本地路径 |
| `--split` | 需要导出的切分（默认 `train`） |
| `--samples-per-shard` | 单个分片包含的样本数（默认 200K） |
| `--max-samples` | 限制总样本数，便于快速验证 |
| `--compression` | `none` / `gzip`，启用后生成 `.jsonl.gz` |
| `--text-column` | 文本字段名称（默认 `text`） |

输出目录结构示例：

```
~/data/openseek_flagscale/jsonl/
├── shard_00000.jsonl.gz
├── shard_00001.jsonl.gz
└── ...
```

## 步骤二：可选 - 生成 Megatron/Energon 数据格式

FlagScale 官方推荐使用 Megatron/Energon 二进制数据格式。可在 FlagScale 仓库内运行：

```bash
cd ~/workspace/FlagScale/tools
python preprocess_data.py \
  --input ~/data/openseek_flagscale/jsonl \
  --input-type json \
  --map-keys text \
  --output-prefix ~/data/openseek_flagscale/energon/openseek \
  --tokenizer-path /path/to/tokenizer.model \
  --dataset-impl mmap \
  --workers 16
```

请根据实际情况修改 tokenizer 路径、并发 worker 数等参数。生成完成后会得到 `*.bin` 与 `*.idx` 文件。

## 步骤三：配置 FlagScale 训练

1. 复制 FlagScale 示例配置（例如 `examples/deepseek_v3/conf`）到新的实验目录。
2. 在任务级 YAML（如 `train/train_openseek.yaml`）中指定数据路径：

```yaml
task:
  dataset:
    type: megatron
    data_path: /home/xxx/data/openseek_flagscale/energon/openseek
    data_impl: mmap
    splits: 98,2,0
    map_keys: [text]
```

若直接使用 JSONL，可改用 Energon 的 JSON loader：

```yaml
task:
  dataset:
    type: json
    data_path: /home/xxx/data/openseek_flagscale/jsonl
    pattern: shard_*.jsonl.gz
    map_key: text
```

3. 根据集群规格调整 `system.tensor_model_parallel_size`、`pipeline_model_parallel_size`、`checkpoint.save_interval` 等配置。
4. 启动训练：

```bash
cd ~/workspace/FlagScale
python run.py \
  --config-path=/path/to/your/conf \
  --config-name=config_openseek.yaml \
  action=run
```

## 验证与调试建议

- 先设置 `--max-samples 10000` 生成小规模数据，确保预处理与配置无误。
- 使用 FlagScale 的 `action=dryrun` 检查调度脚本与分布式环境配置。
- 建议在正式训练前通过 `tools/preprocess_data.py --just-check`（若 FlagScale 后续提供）验证数据可读性。

## 参考资料

- `OpenSeek/docs/FlagScale_Usage.md`
- FlagScale 官方仓库：https://github.com/FlagOpen/FlagScale
- Megatron Energon 说明：https://github.com/FlagOpen/FlagScale/tree/main/megatron-energon

如有改进建议欢迎在 OpenSeek / FlagScale 仓库提交 Issue 或 PR。


