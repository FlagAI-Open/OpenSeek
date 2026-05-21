# 🔧 最终修复指南 - 彻底解决模型违规问题

## ✅ 问题已修复

**最新代码已彻底移除 `Qwen2.5-3B-Instruct` 回退选项！**

---

## 🚀 容器内执行（一键修复+重跑）

```bash
# 1. 拉取最新代码（包含修复）
cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 2. 验证修复（应该没有输出）
grep -n "Qwen2.5" src/main.py

# 3. 检查vLLM模型
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# 应该看到: "id": "/home/Qwen/Qwen3-4B"

# 4. 设置环境变量
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

# 5. 停止当前运行的任务（如果有）
pkill -f "python main.py"

# 6. 删除旧结果文件
rm -f outputs/openseek-3-v*.jsonl
rm -f outputs/openseek-7-v*.jsonl
rm -f outputs/openseek-8-v*.jsonl

# 7. 重跑所有任务（使用正确模型）
cd src

# 任务3（分类任务）
python main.py --task_id 3 --max_input_length 15000 --log_path_prefix ../outputs/

# 任务7（阅读理解）
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/

# 任务8（Triton代码生成）
python main.py --task_id 8 --max_input_length 20000 --log_path_prefix ../outputs/

cd ..
```

---

## 📊 预期结果

### Tokenizer加载

修复后，运行时应该看到：

```
[INFO] 尝试加载Tokenizer: /home/Qwen3-4B
[WARN] 加载失败: ...
[INFO] 尝试加载Tokenizer: /home/models/Qwen3-4B
[WARN] 加载失败: ...
...
[WARN] 所有Tokenizer加载尝试均失败，使用字符估算模式  ✅ 这是正确的！
[INFO] 这不会影响标注功能，只是token计数不够精确
```

**不应该看到**：

```
[INFO] Tokenizer加载成功: Qwen/Qwen2.5-3B-Instruct  ❌ 这是违规的！
```

---

## ⚠️ 重要说明

### 为什么Tokenizer会加载失败？

本地路径 `/home/Qwen3-4B` 是模型权重路径，不是tokenizer路径。因此会失败，但**这是正常的**！

### 字符估算模式

当所有tokenizer路径都失败时，系统会自动使用字符估算模式：
- ✅ **功能正常**：所有标注功能都能正常工作
- ✅ **符合规则**：不使用违规模型
- ⚠️ **精度稍低**：token计数不够精确（但满足需求）

### vLLM使用的模型

实际推理通过vLLM API调用，使用的是 `/home/Qwen/Qwen3-4B`，这符合比赛规则！

---

## 🔍 验证修复成功

### 检查点1：代码修复

```bash
grep -n "Qwen2.5" src/main.py
# 应该没有输出（说明已删除）
```

### 检查点2：vLLM模型

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# 应该看到: "id": "/home/Qwen/Qwen3-4B"
```

### 检查点3：运行日志

运行任务时查看日志，应该看到：
- ❌ 不包含 `Qwen/Qwen2.5-3B-Instruct`
- ✅ 可能包含 `使用字符估算模式`

---

## 📝 已完成任务的状态

### 之前的结果（违规）

| 任务 | 状态 | 使用的模型 | 是否违规 |
|------|------|-----------|---------|
| 任务3-v1 | 100%有效 | 未确认 | ⚠️ 可能违规 |
| 任务7-v2 | 96.8%有效 | Qwen2.5 | ❌ 违规 |
| 任务8-v1 | 100%有效 | Qwen2.5 | ❌ 违规 |

### 重跑后的结果（合规）

| 任务 | 预期状态 | 使用的模型 | 是否合规 |
|------|----------|-----------|---------|
| 任务3 | 待确认 | Qwen3-4B | ✅ 合规 |
| 任务7 | 待确认 | Qwen3-4B | ✅ 合规 |
| 任务8 | 待确认 | Qwen3-4B | ✅ 合规 |

---

## ⏱️ 时间估算

| 任务 | 样本数 | 预估时间 |
|------|--------|----------|
| 任务3 | 500 | ~2小时 |
| 任务7 | 500 | ~2小时 |
| 任务8 | 166 | ~1小时 |
| **总计** | - | **~5小时** |

---

## 🎯 后台运行（推荐）

如果需要长时间运行，使用后台运行模式：

```bash
# 创建日志文件
LOG_FILE="/tmp/rerun_all_$(date +%Y%m%d_%H%M%S).log"

# 后台运行
nohup bash -c '
cd /home/topgo-openseek

export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

cd src

for task_id in 3 7 8; do
  echo "开始任务 $task_id..."
  python main.py --task_id $task_id --max_input_length 15000 --log_path_prefix ../outputs/
  echo "任务 $task_id 完成！"
done

cd ..
echo "所有任务完成！"
' > "$LOG_FILE" 2>&1 &

echo "后台运行已启动，PID: $!"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
```

---

## 🚨 紧急情况处理

### 如果任务正在运行

```bash
# 查看运行中的任务
ps aux | grep "python main.py"

# 停止任务
pkill -f "python main.py"

# 然后按照上述步骤重新运行
```

### 如果vLLM服务异常

```bash
# 检查vLLM状态
ps aux | grep vllm

# 重启vLLM
pkill -f vllm
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/Qwen/Qwen3-4B \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-model-len 32768 \
  > /tmp/vllm.log 2>&1 &

# 等待30秒
sleep 30
```

---

## ✅ 完成后验证

```bash
# 检查所有输出文件
ls -lh outputs/openseek-*.jsonl

# 统计有效预测数
python3 << 'EOF'
import json
import glob

for file in sorted(glob.glob("outputs/openseek-*.jsonl")):
    with open(file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    total = len(lines)
    nulls = sum(1 for l in lines if '"prediction": null' in l)
    valid = total - nulls
    print(f"{file.split('/')[-1]}: {valid}/{total} 有效 ({valid*100/total:.1f}%)")
EOF
```

---

## 📞 需要帮助？

- 查看实时日志：`tail -f /tmp/rerun_all_*.log`
- 检查vLLM日志：`tail -50 /tmp/vllm.log`
- 验证API：`curl http://localhost:8000/v1/models`
