# 🚨 紧急修复指南 - 模型违规问题

## ⚠️ 问题描述

**违反比赛规则**：系统实际使用了 `Qwen/Qwen2.5-3B-Instruct`，而不是允许的 `Qwen3-4B`！

### 问题证据

```
[INFO] Tokenizer加载成功: Qwen/Qwen2.5-3B-Instruct  ❌ 错误！
```

**比赛规定**：
> 模型使用限制：参赛过程中禁止使用Qwen3-4B以外的模型。如违反该要求，其参赛成绩将被认定为无效。

---

## 🔍 问题根源

### 原因分析

1. **Tokenizer加载失败**：
   - 本地路径 `/home/Qwen3-4B` 等都不是有效的tokenizer路径
   - `AutoTokenizer.from_pretrained()` 无法识别这些路径

2. **自动回退到错误模型**：
   - main.py 第239行回退到 `Qwen/Qwen2.5-3B-Instruct`
   - 这违反了比赛规则

3. **Tokenizer vs 模型**：
   - Tokenizer用于计算token数量（不参与推理）
   - API调用时使用的是vLLM服务中的模型
   - 但两者都应该使用Qwen3-4B

---

## ✅ 解决方案

### 方案1：容器内一键修复（推荐）

```bash
# 进入容器
docker exec -it <容器名> bash

# 拉取最新代码
cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 执行修复脚本
chmod +x scripts/fix_model_issue.sh
./scripts/fix_model_issue.sh

# 检查vLLM是否使用正确模型
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 重跑任务
cd src
python main.py --task_id 8 --max_input_length 20000 --log_path_prefix ../outputs/
```

---

### 方案2：手动修复（详细步骤）

#### 步骤1：检查vLLM服务

```bash
# 检查vLLM使用的模型
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 应该看到："/home/Qwen/Qwen3-4B"
```

#### 步骤2：如果vLLM模型错误，重启服务

```bash
# 停止旧服务
pkill -f vllm

# 启动正确模型
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/Qwen/Qwen3-4B \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-model-len 32768 \
  > /tmp/vllm.log 2>&1 &

# 等待启动
sleep 30

# 验证
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

#### 步骤3：修复代码

```bash
cd /home/topgo-openseek

# 备份
cp src/main.py src/main.py.bak
cp src/method.py src/method.py.bak

# 修复method.py - 使用正确的模型路径
sed -i 's|model = "Qwen3-4B-ascend-flagos"|model = "/home/Qwen/Qwen3-4B"|g' src/method.py
sed -i 's|model = "Qwen/Qwen2.5-3B-Instruct"|model = "/home/Qwen/Qwen3-4B"|g' src/method.py

# 修复main.py - 移除Qwen2.5回退
sed -i "s|tokenizer_paths.append('Qwen/Qwen2.5-3B-Instruct')|# 移除Qwen2.5回退 - 违规|g" src/main.py
```

#### 步骤4：设置环境变量

```bash
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"
```

#### 步骤5：测试API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "测试"}],
    "max_tokens": 10
  }'
```

#### 步骤6：重跑任务

```bash
cd src
python main.py --task_id 8 --max_input_length 20000 --log_path_prefix ../outputs/
```

---

## 🔍 验证修复成功

### 检查点1：Tokenizer加载

运行任务时应该看到：

```
[INFO] Tokenizer加载成功: /home/Qwen/Qwen3-4B  ✅
```

或

```
[WARN] 所有Tokenizer加载尝试均失败，使用字符估算模式  ✅ 也可以接受
```

**不应该看到**：

```
[INFO] Tokenizer加载成功: Qwen/Qwen2.5-3B-Instruct  ❌ 违规！
```

### 检查点2：API调用

查看日志中的API请求，应该使用：

```
"model": "/home/Qwen/Qwen3-4B"  ✅
```

---

## 📊 已完成任务的状态

| 任务 | 状态 | 使用的模型 | 是否违规 |
|------|------|-----------|---------|
| 任务3-v1 | 100%有效 | **未知** | ⚠️ 需验证 |
| 任务7-v2 | 96.8%有效 | **Qwen2.5** | ❌ 违规 |
| 任务8-v1 | 100%有效 | **Qwen2.5** | ❌ 违规 |

---

## 🎯 后续行动

### 立即执行：

1. ✅ **修复模型问题** - 使用上述方案
2. 🔄 **重跑所有任务** - 确保使用正确模型
3. ✅ **验证修复** - 检查日志确认模型正确

### 重新评估：

由于使用了错误模型，**任务7和8的结果可能无效**！

建议：
- **方案A**：重跑所有任务（确保合规）
- **方案B**：联系组委会说明情况
- **方案C**：检查任务3是否也违规

---

## 📝 技术说明

### Tokenizer的作用

Tokenizer在系统中只用于：
1. 计算输入token数量
2. 控制输入长度不超过限制
3. **不参与实际推理**

### API调用使用的模型

实际推理通过vLLM API调用：
- method.py中的 `model = "/home/Qwen/Qwen3-4B"`
- API请求中的 `"model": "/home/Qwen/Qwen3-4B"`

### 为什么会违规？

虽然Tokenizer不参与推理，但比赛规则明确要求：
> 禁止使用Qwen3-4B以外的模型

这包括：
- Tokenizer
- 推理模型
- 任何其他模型组件

---

## 🆘 紧急联系

如果遇到问题：

1. 检查vLLM日志：`tail -50 /tmp/vllm.log`
2. 检查任务日志：`tail -f /tmp/rerun_tasks_*.log`
3. 验证API：`curl http://localhost:8000/v1/models`
