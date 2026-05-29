# 长上下文ICL自动数据标注方案

本方案基于Qwen3-4B大语言模型，采用In-Context Learning (ICL) 范式完成8个数据集的自动标注任务。

## 环境要求

- Python >= 3.8
- Huawei Ascend 910C GPU
- vLLM推理引擎 (支持Ascend)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型准备

下载Qwen3-4B模型到指定路径（默认为 `/root/Qwen3-4B`）：

- HuggingFace: https://huggingface.co/Qwen/Qwen3-4B
- ModelScope: https://modelscope.cn/models/Qwen/Qwen3-4B

## 启动vLLM服务

在使用本方案前，需要先启动vLLM推理服务：

```bash
# 使用vLLM Ascend版本启动服务
vllm serve /root/Qwen3-4B \
  --port 9010 \
  --dtype auto \
  --max-model-len 131072 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --block-size 16 \
  --trust-remote-code
```

服务启动后，API地址为：`http://localhost:9010/v1/`

## 数据准备

确保数据集已放置在正确路径：

```
/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/
```

## 运行标注任务

### 单个任务运行

```bash
python3 main.py \
  --task_id 1 \
  --tokenizer_path /root/Qwen3-4B \
  --log_path_prefix ./outputs/ \
  --max_input_length 128000
```

参数说明：
- `--task_id`: 任务ID (1-8)
- `--tokenizer_path`: Qwen3-4B模型路径
- `log_path_prefix`: 输出文件路径前缀
- `--max_input_length`: 最大输入长度（默认128000）

### 批量运行所有任务

```bash
#!/bin/bash
OUTPUT_DIR="./outputs"
TOKENIZER_PATH="/root/Qwen3-4B"

for task_id in 1 2 3 4 5 6 7 8; do
    python3 main.py \
        --task_id $task_id \
        --log_path_prefix $OUTPUT_DIR/ \
        --tokenizer_path $TOKENIZER_PATH
done
```

### 并行运行（4个任务同时）

```bash
#!/bin/bash
OUTPUT_DIR="./outputs"
TOKENIZER_PATH="/root/Qwen3-4B"
CONCURRENT=4

for task_id in 1 2 3 4 5 6 7 8; do
    python3 main.py \
        --task_id $task_id \
        --log_path_prefix $OUTPUT_DIR/ \
        --tokenizer_path $TOKENIZER_PATH > task${task_id}.log 2>&1 &
    
    if [ $((task_id % $CONCURRENT)) -eq 0 ]; then
        wait
    fi
done
wait
```

## 输出结果

每个任务会生成一个 `.jsonl` 文件，格式如下：

```json
{"test_sample_id": "1", "prediction": "Good Review"}
{"test_sample_id": "2", "prediction": "Bad Review"}
```

## 方案特点

1. **长上下文支持**：使用50个ICL示例，通过YARN RoPE scaling支持最长131,072 tokens
2. **提示工程**：优化的prompt设计，确保输出格式准确（使用<label>标签）
3. **高效推理**：基于vLLM引擎，支持OpenAI兼容API
4. **Ascend优化**：针对华为Ascend NPU进行了专门优化

## 技术方案说明

### 1. 提示策略设计
- 结构化prompt模板，明确角色定义和任务规则
- 使用`<label>`标签确保输出格式统一
- 鼓励内部推理过程，但只输出最终结果

### 2. 上下文构造
- 动态选择ICL示例，基于token长度控制上下文大小
- 使用Qwen3-4B原生tokenizer精确计算token数量
- 最多使用50个示例，平衡性能与上下文长度

### 3. 一致性保证
- 统一的prompt模板确保不同任务的输出格式一致
- 多轮对话场景下，使用相同的系统提示和示例选择策略
- 可扩展的架构设计，便于添加新任务

## 测试API连接

可以使用 `api_test.py` 测试vLLM服务是否正常运行：

```bash
python3 api_test.py
```

## 故障排除

1. **内存不足**：减少 `--max-model-len` 或 `--gpu-memory-utilization`
2. **推理速度慢**：增加 `--tensor-parallel-size`（如果有多张GPU）
3. **输出格式错误**：检查prompt中的<label>标签使用是否正确

## 版本信息

- Qwen3-4B
- vLLM 0.13.0+ascend
- Python 3.11.13

## 联系方式

如有问题，请通过比赛官方渠道联系。