# 超长上下文场景中LLM自动数据标注挑战赛

---

## 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境
conda create -n qwen3_annotation python=3.10 -y
conda activate qwen3_annotation

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装 FlagScale（源码安装）

FlagScale 需从 GitHub 源码安装：

```bash
git clone https://github.com/flagos-ai/FlagScale.git
cd FlagScale
pip install -e .
cd ..
```

验证安装：

```bash
pip list | grep flagscale
# 应输出: flagscale 1.0.0
```

### 3. 模型权重

下载 Qwen3-4B 模型权重至本地目录，例如 `/path/to/Qwen3-4B`：

```
Qwen3-4B/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
└── ...
```

### 4. 长文本配置（YaRN RoPE Scaling）

本方案使用 YaRN rope scaling 扩展上下文窗口至 **128K tokens**。需在 `config.json` 中配置：

```json
"rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}
```

### 5. 模型部署

#### 方式一：使用 FlagScale 启动

```bash
cd /path/to/FlagScale
python run.py \
    --config-path /path/to/LongContext-ICL-Annotation/src \
    --config-name llm_config \
    action=run
```

#### 方式二：直接使用 vLLM 启动（推荐）

```bash
# 设置模型路径
MODEL_PATH=/path/to/Qwen3-4B

# 启动 vLLM 推理服务
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 2026 \
    --model ${MODEL_PATH} \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --max-model-len 40000 \
    --max-num-batched-tokens 40000 \
    --max-num-seqs 4 \
    --enable-prefix-caching
```

参数说明：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--host` | 监听地址 | 0.0.0.0 |
| `--port` | 服务端口 | 2026 |
| `--gpu-memory-utilization` | GPU 显存利用率 | 0.90 |
| `--max-model-len` | 最大模型输入长度 | 40000 |
| `--max-num-batched-tokens` | 最大批处理 token 数 | 40000 |
| `--max-num-seqs` | 最大并发序列数 | 4 |
| `--enable-prefix-caching` | 启用前缀缓存 | 开启 |

#### 验证服务

```bash
curl http://0.0.0.0:2026/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/Qwen3-4B",
        "prompt": "Hello, who are you?",
        "max_tokens": 50
    }'
```

#### 停止服务

```bash
# 方式一：FlagScale 停止
cd /path/to/FlagScale
python run.py \
    --config-path /path/to/LongContext-ICL-Annotation/src \
    --config-name llm_config \
    action=stop

# 方式二：直接 kill vLLM 进程
pkill -f "vllm.entrypoints.openai.api_server"
```

### 6. 运行标注

进入 `qwen3_experiment/` 目录后执行：

#### 单任务运行

```bash
cd qwen3_experiment

# 运行单个任务（例如 Task 1）
python main.py \
    --task_id 1 \
    --max_input_length 128000 \
    --tokenizer_path /path/to/Qwen3-4B \
    --log_path_prefix ./outputs/ \
    --max_examples 100
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task_id` | 任务 ID（1-8） | 必填 |
| `--max_input_length` | 最大输入长度（tokens） | 128000 |
| `--tokenizer_path` | Qwen3-4B tokenizer 路径 | /root/autodl-tmp/qwen3-4b |
| `--log_path_prefix` | 输出结果目录 | ./outputs/ |
| `--num_samples` | 多次采样次数（>1 启用投票） | 1 |
| `--max_examples` | 最大 ICL 示例数 | 100 |
| `--no_alignment` | 禁用结构对齐后处理 | False |

#### 批量运行所有任务

```bash
cd qwen3_experiment
bash run_all.sh /path/to/Qwen3-4B ./outputs
```

### 7. 打包结果

```bash
cd qwen3_experiment
bash package_results.sh ./outputs result.zip
```

打包后的 `result.zip` 包含 8 个 JSONL 文件（`openseek-1-v1.jsonl` ~ `openseek-8-v1.jsonl`），每个文件包含 `test_sample_id` 和 `prediction` 两个字段。

---

## 方案说明

### 核心改进

本方案针对超长上下文 ICL 数据标注的三个核心问题进行了优化：

#### 1. 上下文压缩与结构对齐（Context Compression & Structure Alignment）

提出了一种轻量级的 **上下文压缩与结构对齐** 方法，通过以下技术提升标注质量：

- **上下文压缩**：将 ICL 示例压缩为紧凑的结构化模板，减少冗余信息对模型推理的干扰
- **结构对齐**：在模型输出后，通过结构对齐后处理，将原始输出与预计算的结构化模板进行匹配，确保输出格式的一致性和准确性
- **模板匹配**：利用 TF-IDF 加权结构哈希和输出模式归一化，实现高效的模板匹配


#### 2. 超长上下文下的 Prompt 设计

- **任务感知的 Prompt 模板**：为 6 种任务类型（数学推理、语言分析、分类、代码/字符串、开放生成、代码生成）分别设计了定制化的 Prompt 模板
- **结构化 Prompt 架构**：Role → Task → Instructions → Examples → Input → Output 的分层结构
- **输出格式约束**：在 Prompt 中明确要求 `<label>` 标签格式，并给出正反示例

#### 3. 示例选择策略（超过上下文容量时）

- **BM25 检索**：使用 BM25 算法检索与待标注样本最相关的示例
- **多样性采样**：按输出类型分组，轮询选择确保覆盖不同类别
- **动态长度控制**：根据任务类型动态调整上下文窗口
- **Token 精确计算**：使用 Qwen3-4B 的 tokenizer 精确计算每个示例的 token 数

#### 4. 推理与后处理

- **鲁棒的标签提取**：多层 fallback 策略
- **多次采样投票**：支持多次采样后投票取最一致结果
- **上下文长度检查**：自动检测超长输入并降级使用更少的示例
- **结构对齐后处理**：对模型输出进行结构对齐，提升输出质量

---

## 文件结构

```
LongContext-ICL-Annotation/
├── qwen3_experiment/           # 方案代码
│   ├── main.py                 # 主评估脚本
│   ├── method.py               # 核心方法（Prompt、示例选择、标注、对齐）
│   ├── requirements.txt        # 依赖列表
│   ├── run_all.sh              # 批量运行脚本
│   ├── package_results.sh      # 结果打包脚本
│   └── README.md               # 本文件
├── data/                       # 8个任务数据集
│   ├── openseek-1~8_*.json     # 8个 JSON 数据集文件
│   └── README.md / README_zh.md

```

---

## 数据集

| Task ID | 任务名 | 最短 ICL 上下文 | 测试样本数 | 任务类型 |
|---------|--------|----------------|-----------|---------|
| 1 | closest_integers | 30K | 500 | 数学推理 |
| 2 | count_nouns_verbs | 30K | 500 | 语言分析 |
| 3 | collatz_conjecture | 30K | 500 | 数学推理 |
| 4 | conala_concat_strings | 30K | 500 | 代码/字符串 |
| 5 | tweet_sadness_detection | 30K | 500 | 分类 |
| 6 | mnli_same_genre_classification | 30K | 500 | 分类 |
| 7 | jeopardy_answer_generation | 30K | 500 | 开放生成 |
| 8 | kernel_generation | 16K | 166 | 代码生成 |

数据集文件位于 `../data/` 目录下，每个任务对应一个 JSON 文件，包含 `Definition`、`examples` 和 `test_samples` 三个字段。

---

## 上下文压缩与结构对齐

### 压缩算法

上下文压缩算法使用以下技术将 ICL 示例压缩为紧凑的结构化模板：

- **TF-IDF 加权结构哈希**：对示例进行 TF-IDF 加权哈希，提取关键结构特征
- **输出归一化**：将输出格式归一化
- **模板去重**：通过局部敏感哈希（LSH）进行去重

---

## 常见问题

### Q1: 模型服务启动失败怎么办？

检查以下内容：
- FlagScale 是否正确安装（`pip list | grep flagscale`）
- 模型权重路径是否正确
- GPU 显存是否充足（Qwen3-4B 需要至少 8GB 显存）
- 端口 2026 是否被占用

### Q2: 如何调整上下文窗口大小？

修改 `--max_input_length` 参数即可。对于 Task 8（代码生成），建议使用较小的窗口（如 30000），因为代码生成的上下文需求较低。

### Q3: 如何启用多次采样投票？

```bash
python main.py --task_id 2 --num_samples 3 --max_input_length 128000
```

`num_samples=3` 表示对每个样本采样 3 次，然后通过投票选择最一致的结果。


### Q4: 如何复现完整环境？

```bash
pip install -r requirements.txt
```

核心依赖包括：`torch`、`transformers`、`vllm`、`requests`、`openai`、`tqdm`、`ctx-compress`。FlagScale 需从源码安装（见第2节）。
