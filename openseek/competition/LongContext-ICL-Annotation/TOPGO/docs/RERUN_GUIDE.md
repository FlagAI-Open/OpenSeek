# 🚀 FlagOS OpenSeek 赛道三 - 重跑任务指南

## ✅ API问题已修复

**修复内容**：
- 模型名称从 `Qwen3-4B-ascend-flagos` 改为 `/home/Qwen/Qwen3-4B`
- 代码已推送到Gitee，容器拉取即可生效

---

## 📋 执行步骤

### 方式一：前台运行（推荐用于调试）

```bash
# 1. 进入容器
docker exec -it <容器名> bash

# 2. 拉取最新代码（包含API修复）
cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 3. 赋予执行权限
chmod +x scripts/rerun_tasks_fixed.sh

# 4. 执行脚本
./scripts/rerun_tasks_fixed.sh
```

**特点**：
- ✅ 实时查看输出
- ✅ 自动检查vLLM服务
- ✅ 自动验证API修复
- ✅ 自动统计结果
- ⚠️  终端关闭任务会中断

---

### 方式二：后台运行（推荐用于长时间运行）

```bash
# 1. 进入容器
docker exec -it <容器名> bash

# 2. 拉取最新代码
cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 3. 赋予执行权限
chmod +x scripts/rerun_background.sh

# 4. 后台执行
./scripts/rerun_background.sh

# 5. 查看日志（可退出终端）
tail -f /tmp/rerun_tasks_*.log

# 或查看最近20行
tail -20 /tmp/rerun_tasks_*.log
```

**特点**：
- ✅ 可以关闭终端
- ✅ 适合长时间运行
- ✅ 所有输出记录到日志
- ✅ 可随时查看进度

---

### 方式三：手动分步执行（精细控制）

```bash
# 1. 检查vLLM服务
curl http://localhost:8000/v1/models | python3 -m json.tool

# 如果没有运行，启动vLLM
nohup python -m vllm.entrypoints.openai.api_server \
    --model /home/Qwen/Qwen3-4B \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 32768 \
    > /tmp/vllm.log 2>&1 &

# 等待启动（30秒）
sleep 30

# 2. 设置环境变量
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

# 3. 拉取最新代码
cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 4. 单独运行每个任务
cd src

# 任务3（分类任务）
python main.py --task_id 3 --max_input_length 15000 --log_path_prefix ../outputs/

# 任务7（阅读理解）
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/

# 任务8（Triton代码生成）
python main.py --task_id 8 --max_input_length 15000 --log_path_prefix ../outputs/

cd ..
```

---

## 📊 预期结果

| 任务 | 样本数 | 当前状态 | 目标状态 | 预估时间 |
|------|--------|----------|----------|----------|
| 3 | ~500 | 50.3%有效 | >90%有效 | ~3小时 |
| 7 | ~500 | 26%有效 | >80%有效 | ~2小时 |
| 8 | ~166 | 全null | >50%有效 | ~30分钟 |

---

## 🔍 验证API修复

运行前先测试API是否正常：

```bash
# 测试API连接
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

**成功响应示例**：
```json
{
  "id": "cmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "/home/Qwen/Qwen3-4B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！有什么可以帮助你的吗？"
    },
    "finish_reason": "stop"
  }]
}
```

**如果返回404错误**，说明模型名称不对，请检查：
```bash
# 查看vLLM加载的模型名称
curl http://localhost:8000/v1/models | python3 -m json.tool
```

---

## 🚨 常见问题

### 1. vLLM服务启动失败
```bash
# 查看日志
tail -50 /tmp/vllm.log

# 常见原因：
# - GPU内存不足：降低 --gpu-memory-utilization
# - 模型路径错误：确认 /home/Qwen/Qwen3-4B 存在
```

### 2. Git拉取冲突
```bash
# 强制覆盖本地修改
git fetch origin
git reset --hard origin/master
```

### 3. 磁盘空间不足
```bash
# 检查磁盘
df -h

# 清理旧日志
rm -f /tmp/*.log

# 清理旧备份
rm -rf outputs_backup_*
```

### 4. 查看运行中的任务
```bash
# 查看Python进程
ps aux | grep python

# 查看vLLM进程
ps aux | grep vllm

# 查看后台任务
jobs -l
```

---

## 📥 下载结果

完成后，将结果文件下载到本地：

```bash
# 在本地Windows执行
scp root@<服务器IP>:/home/topgo-openseek/outputs/openseek-*-v1.jsonl c:/D/compet/dcic/FlagOS开放计算全球挑战赛/TOPGO-track3-solution/outputs/
```

或使用Docker CP：
```bash
docker cp <容器名>:/home/topgo-openseek/outputs/openseek-3-v1.jsonl c:/D/compet/dcic/FlagOS开放计算全球挑战赛/TOPGO-track3-solution/outputs/
docker cp <容器名>:/home/topgo-openseek/outputs/openseek-7-v1.jsonl c:/D/compet/dcic/FlagOS开放计算全球挑战赛/TOPGO-track3-solution/outputs/
docker cp <容器名>:/home/topgo-openseek/outputs/openseek-8-v1.jsonl c:/D/compet/dcic/FlagOS开放计算全球挑战赛/TOPGO-track3-solution/outputs/
```

---

## ✨ 下一步

1. ✅ **API已修复** - 代码已推送
2. 🔄 **容器拉取代码** - `git reset --hard origin/master`
3. 🚀 **运行重跑脚本** - 选择前台或后台方式
4. ⏱️ **等待完成** - 预计5-6小时
5. 📥 **下载结果** - 使用scp或docker cp
6. 🎯 **提交材料** - 准备最终提交

---

## 🔬 LDR集成 - Local Deep Research 增强功能

### LDR新增模块

| 模块 | 文件 | 功能 |
|------|------|------|
| EvidenceVerifier | `src/evidence_verifier.py` | 证据验证，升级质检为"规则+证据"模式 |
| DynamicRetriever | `src/dynamic_retriever.py` | 动态多源检索，支持fallback |
| SelfEvolvingKnowledgeBase | `src/dynamic_retriever.py` | 自进化知识库，数据飞轮 |

### 验证LDR模块

```bash
# 进入容器后
cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 验证模块存在
ls -la src/evidence_verifier.py src/dynamic_retriever.py

# 测试Python导入
python3 -c "from src import DynamicRetriever, EnhancedQualityInspector; print('LDR模块导入成功')"
```

### LDR功能测试

```bash
# 测试证据验证
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/topgo-openseek')
from src.evidence_verifier import EvidenceVerifier, EnhancedQualityInspector

verifier = EvidenceVerifier(max_duration=10)
qc = EnhancedQualityInspector(evidence_verifier=verifier)
decision = qc.verify_entity_with_evidence(
    entity="测试实体",
    entity_type="organization",
    context="测试上下文",
    task_domain="general"
)
print(f"决策: {decision.decision}, 置信度: {decision.confidence:.2f}")
PYEOF
```

### 技术文档

- 完整文档：`docs/LDR_INTEGRATION.md`
- 容器指南：`docs/LDR_CONTAINER_GUIDE.md`

---

## 📞 需要帮助？

- 查看实时日志：`tail -f /tmp/rerun_tasks_*.log`
- 检查API状态：`curl http://localhost:8000/v1/models`
- 查看vLLM日志：`tail -50 /tmp/vllm.log`
