# OpenSeek ICL 超长上下文数据标注方案

## 概述

FlagOS 超长上下文数据标注挑战赛解决方案，使用 Qwen3-4B 模型，在 30K token 上下文窗口内（Task 8 为 16K），通过 In-Context Learning 完成 8 个异构任务的自动标注。

## 环境配置

### 1. GPU 服务器环境安装

```bash
chmod +x create_env_nvidia.sh
./create_env_nvidia.sh
```

该脚本会安装 FlagScale 框架、CUDA 依赖、vLLM、Flash Attention 等全部组件。

### 2. Python 依赖

```bash
cd src
pip install -r requirements.txt
```

`requirements.txt` 包含：

- `rank_bm25` — BM25 检索
- `spacy` + `en_core_web_sm` — Task 2 的 POS 词性检索

### 3. 下载模型

```bash
mkdir -p /root/models/qwen
cd /root/models/qwen
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B
# 或使用镜像: https://hf-mirror.com/Qwen/Qwen3-4B
```

## 模型部署与推理

### 启动模型服务（FlagScale + vLLM）

```bash
cd FlagScale
# 启动服务
python3 run.py --config-path .. --config-name llm_config action=run
# 停止服务
python3 run.py --config-path .. --config-name llm_config action=stop
```

`llm_config.yaml` 配置了 vLLM 双卡 TP=2、max_model_len=65536、端口 2026。服务启动后监听 `http://127.0.0.1:2026/v1/`。

### 运行推理

**单任务：**

```bash
cd src
python main.py --task_id 1 --workers 8
```

**全部 8 个任务：**

```bash
nohup bash -c 'for i in {1..8}; do echo "=== Task $i $(date) ==="; python3 main.py --task_id $i --tokenizer_path /root/models/qwen/Qwen3-4B --workers 2  ; done' > run_v3_dynamic.log 2>&1### 参数说明
```

| 参数                   | 默认值                          | 说明           |
| -------------------- | ---------------------------- | ------------ |
| `--task_id`          | (必填)                         | 任务 ID (1-8)  |
| `--tokenizer_path`   | `/root/models/qwen/Qwen3-4B` | Tokenizer 路径 |
| `--workers`          | `4`                          | 并发请求数        |
| `--max_input_length` | `32000`                      | 最大输入长度       |
| `--log_path_prefix`  | `../outputs/`                | 输出文件前缀       |
| `--max_samples`      | `None`                       | 限制测试样本数（调试用） |

## 输出文件

运行后在输出目录生成：

- `openseek-{task_id}-v{N}.jsonl` — 提交格式结果
- `debug_task{task_id}.jsonl` — 调试信息（prompt、raw outputs、候选等）

### 提交格式

```json
{"test_sample_id": "openseek-1-xxxx", "prediction": "1"}
```

## 目录结构

```
src/
├── main.py                  # 入口脚本：数据加载、并发推理、结果写入
├── method.py                # API 调用封装（OpenAI 兼容接口）
├── retriever.py             # BM25 检索器
├── pos_retriever.py         # POS 词性检索器（Task 2）
├── shared_utils.py          # 共享工具函数
├── requirements.txt         # Python 依赖
├── llm_config.yaml          # FlagScale 模型启动配置（vLLM, TP=2）
├── create_env_nvidia.sh     # GPU 服务器环境安装脚本
├── tasks/
│   ├── __init__.py          # 任务注册表
│   ├── base.py              # 基类 BaseTask
│   ├── task_1.py            # Task 1: Closest Integers
│   ├── task_2.py            # Task 2: Count Nouns/Verbs
│   ├── task_3.py            # Task 3: Collatz Conjecture
│   ├── task_4.py            # Task 4: String Concatenation
│   ├── task_5.py            # Task 5: Tweet Sadness Detection
│   ├── task_6.py            # Task 6: MNLI Same Genre
│   ├── task_7.py            # Task 7: Jeopardy Answer
│   └── task_8.py            # Task 8: Triton Kernel Generation
```

## 各任务策略

| 任务  | 检索策略                  | 核心策略              |
| --- | --------------------- | ----------------- |
| T1  | 特征匹配 + 答案桶均衡          | 4 轮多轮重试 + 思维链提取兜底 |
| T2  | POS 检索 + 精选示例         | 4 轮：固定→动态→反馈→高温   |
| T3  | 长度匹配                  | 数学一致性验证 + 思维链提取   |
| T4  | 特征匹配 + 清洁示例           | 5 轮重试 + 长度/字符验证   |
| T5  | BM25 + 标签均衡           | Emoji 移除 + 难度排序   |
| T6  | 流派过滤 BM25 + 标签均衡      | CoT 样例注入          |
| T7  | Category-First + 答案索引 | 两轮验证              |
| T8  | 操作类型匹配                | 错误规则注入 + 代码验证     |

## 可复现性

- 所有任务的数据加载、示例选择、prompt 构建均为确定性操作
- 输出文件自动版本号，不覆盖已有结果
