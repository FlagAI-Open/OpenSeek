# FlagOS 赛题三: Long-Context ICL Annotation — Ascend 部署指南

> **底层框架**: FlagScale + MindIE/vLLM (昇腾适配版)  
> **模型**: Qwen3-4B  
> **运行设备**: 华为 Ascend 910C × 2  
> **容器端口**: 30000 → 公网映射端口 22653  
> **服务地址**: `https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1`  

---

## 目录

1. [环境要求](#环境要求)
2. [快速开始](#快速开始)
3. [详细步骤](#详细步骤)
   - [Step 1: 检测 Ascend 环境](#step-1-检测-ascend-环境)
   - [Step 2: 下载模型](#step-2-下载模型)
   - [Step 3: 修复长上下文配置](#step-3-修复长上下文配置)
   - [Step 4: 生成配置文件](#step-4-生成配置文件)
   - [Step 5: 启动服务](#step-5-启动服务)
   - [Step 6: 测试 API](#step-6-测试-api)
4. [调用示例](#调用示例)
5. [常见问题](#常见问题)

---

## 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 推荐使用 3.10 或 3.11 |
| 驱动 | Ascend Driver ≥ 24.1.RC1 | Huawei Ascend NPU 驱动 |
| CANN | 8.0+ | Compute Architecture for Neural Networks |
| NPU 卡数 | 2× Ascend 910C | FlagScale 自动调度两张卡 |
| 显存 | 单卡 ≥ 64GB | Qwen3-4B FP16 约需 8GB |

---

## 快速开始

```bash
cd env

# ① 安装依赖
pip install -r requirements.txt

# ② 一键全自动部署
python deploy_and_infer.py full

# ③ 部署后，其他程序可通过以下方式调用
python -c "
from openai import OpenAI
client = OpenAI(
    api_key='dummy',
    base_url='https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1'
)
resp = client.chat.completions.create(
    model='/Qwen3-4B/Qwen/Qwen3-4B',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=100
)
print(resp.choices[0].message.content)
"
```

---

## 详细步骤

### Step 1: 检测 Ascend 环境

确认服务器上的 Ascend NPU 已正确安装并可用：

```bash
python deploy_and_infer.py check-env
```

预期输出：
```
=== Ascend 环境检测 ===
  [OK] CANN 版本: 8.0.RC2
  [OK] npu-smi 可用
       Ascend 设备总数: 2
```

如果检测失败，请确认：
- Ascend 驱动已安装 (`npu-smi info` 可用)
- CANN 工具包已安装 (`ls /usr/local/Ascend/`)
- NPU 卡未被占用

### Step 2: 下载模型

```bash
python deploy_and_infer.py download-model
```

或使用命令行直接下载（模型将存放在 `env/Qwen3-4B`）：
```bash
pip install huggingface-hub
hf download Qwen/Qwen3-4B --local-dir env/Qwen3-4B
```

**大小约 8GB，首次下载需要一定时间。**

### Step 3: 修复长上下文配置

Qwen3-4B 默认仅支持 4K context，需要修改 `rope_scaling` 以支持 32K tokens：

```bash
python deploy_and_infer.py fix-context
```

该命令自动将 `env/Qwen3-4B/config.json` 中的 `rope_scaling` 修改为 yarn 模式。

**手动修改方式（如自动修复失败）：**

编辑 `env/Qwen3-4B/config.json`，搜索 `"rope_scaling"`，替换为：
```json
"rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}
```

### Step 4: 生成配置文件

FlagScale 需要 YAML 格式的服务配置文件：

```bash
python deploy_and_infer.py gen-config
```

生成的文件位于 `env/llm_config_ascend.yaml`，关键配置：
```yaml
serve:
- serve_id: ascend_vllm_model
  engine: mindie            # MindIE Engine (Ascend 推荐)
  engine_args:
    model: Qwen3-4B         # 模型路径 (相对于 env/)
    host: 0.0.0.0
    port: 30000             # 容器内监听端口
    num_gpus: 2             # Ascend 910C 卡数
    device_type: ascend
    npu_device_ids: "0,1"   # Ascend NPU 卡号

envs:
  ASCEND_RT_VISIBLE_DEVICES: "0,1"
```

### Step 5: 启动服务

```bash
python deploy_and_infer.py start
```

或使用全自动化流程：
```bash
python deploy_and_infer.py deploy
```

启动成功后会显示：
```
[OK] FlagScale 服务已启动!
============================================================
容器内地址: http://0.0.0.0:30000
公网映射地址: https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1
------------------------------------------------------------
[调用示例]
  client = OpenAI(
      api_key="dummy",
      base_url="https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1"
  )
```

**端口映射说明：**

容器内部监听 `30000` 端口，由 lab 平台自动映射到公网端口 `22653`：
```
容器内:  0.0.0.0:30000  ← FlagScale/vLLM 服务
         ↓ (lab 平台自动映射)
公网访问: https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1
```

如需修改容器内端口，启动时指定 `--port`：
```bash
python deploy_and_infer.py start --port 30000
```

**停止服务：**
```bash
python deploy_and_infer.py stop
```

### Step 6: 测试 API

验证远程服务是否正常响应：

```bash
# 仅检查连通性
python deploy_and_infer.py test-api

# 发送实际请求进行测试
python deploy_and_infer.py send-test
```

---

## 调用示例

部署完成后，任何客户端都可以通过 OpenAI 兼容接口调用服务：

```python
from openai import OpenAI

# 连接 Ascend 910C 集群上的 FlagScale 服务
client = OpenAI(
    api_key="dummy",
    base_url="https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1"
)

# Chat 接口
response = client.chat.completions.create(
    model="/Qwen3-4B/Qwen/Qwen3-4B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Classify this review: Great product!"},
    ],
    temperature=0.7,
    top_p=0.95,
    max_tokens=10_000,
)
print(response.choices[0].message.content)
```

**批量标注示例：**

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1"
)

# 读取数据文件
import json
with open("data/openseek-1_closest_integers.json") as f:
    data = json.load(f)

# 对每条数据进行标注
for sample in data["test"][:10]:  # 测试前 10 条
    response = client.chat.completions.create(
        model="/Qwen3-4B/Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": sample["input"]}],"temperature=0.7,
        max_tokens=10_000,
    )
    print(response.choices[0].message.content)
```

---

## 查看所有命令

```bash
python deploy_and_infer.py --help
```

| 命令 | 功能 |
|------|------|
| `check-env` | 检测 Ascend NPU 环境是否就绪 |
| `download-model` | 从 HuggingFace 下载 Qwen3-4B 模型权重 |
| `fix-context` | 修复 `config.json` 的 rope_scaling 为 yarn 模式 |
| `gen-config` | 生成 Ascend 专用 FlagScale 配置文件 |
| `deploy` | 半自动部署 (检测→下载→配置→生成→启动) |
| `start` | 启动 FlagScale 推理服务 |
| `stop` | 停止 FlagScale 推理服务 |
| `test-api` | 检查 API 连通性 |
| `send-test` | 发送实际测试请求 |
| `full` | 全自动全流程 (下载→修复→配置→启动→测试) |

---

## 支持的 Task 列表

| Task | 名称 | 最大上下文 | 测试样本数 |
|------|------|-----------|-----------|
| 1 | closest_integers | 30K | 500 |
| 2 | count_nouns_verbs | 30K | 500 |
| 3 | collatz_conjecture | 30K | 500 |
| 4 | conala_concat_strings | 30K | 500 |
| 5 | semeval_tweet_sadness | 30K | 500 |
| 6 | mnli_same_genre_class | 30K | 500 |
| 7 | jeopardy_answer_gen | 30K | 500 |
| 8 | kernel_generation | 16K | 166 |

---

## 常见问题

### Q1: `NpuSmiCommandExecFailed` 或 `npu-smi` 不可用

确认 Ascend 驱动已安装且当前用户有权限执行：
```bash
# 检查驱动
modprobe ahci

# 查看设备
npu-smi info

# 如无权限，尝试 sudo
sudo npu-smi info
```

### Q2: 启动服务时报错 `ModuleNotFoundError: flagScale`

```bash
pip install -r requirements.txt
```

如果仍然报错，确认 FlagScale 仓库已克隆到 env/ 目录下：
```bash
cd env
git clone https://github.com/FlagOpen/FlagScale.git
```

### Q3: 显存不足 / OOM

降低 `env/llm_config_ascend.yaml` 中的 `gpu_memory_utilization`：
```yaml
gpu_memory_utilization: 0.7   # 从 0.9 调低
```

### Q4: 上下文截断 (超过 max_length)

确认已执行 `fix-context` 步骤，且 `env/Qwen3-4B/config.json` 中 `rope_scaling` 已更新为 yarn 模式。

### Q5: 公网 API 无法访问

检查以下内容：
1. 容器内端口 `30000` 是否正确启动
2. Lab 平台的端口映射是否生效
3. 防火墙是否允许入站连接

```bash
# 在容器内检查端口监听
netstat -tlnp | grep 30000

# 在本地测试直连
curl http://localhost:30000/health
```

### Q6: 如何切换引擎 (MindIE vs vLLM)

编辑 `env/llm_config_ascend.yaml` 中的 `engine` 字段：

```yaml
# 使用 MindIE (推荐 Ascend)
engine: mindie

# 或使用昇腾适配版 vLLM
engine: vllm-ascend
```

### Q7: 更换容器内端口

启动时指定 `--port` 参数，并确保与 lab 平台的端口映射一致：
```bash
python deploy_and_infer.py start --port 30001
```

---

## env/ 目录结构

所有环境配置均在 `env/` 下完成：

```
env/
├── requirements.txt          # 依赖清单 (pip install -r)
├── deploy_and_infer.py       # FlagScale 部署管理脚本
├── llm_config_ascend.yaml    # FlagScale 服务配置 (gen-config 生成)
├── Qwen3-4B/                 # 模型权重 (~8GB) (download-model 下载)
├── FlagScale/                # FlagScale 框架 (git clone)
└── README.md                 # 本文件
```

