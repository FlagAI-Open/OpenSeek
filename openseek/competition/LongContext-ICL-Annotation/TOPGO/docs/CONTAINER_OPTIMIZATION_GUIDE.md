# 🚀 容器内执行优化指南

## 📋 已推送内容

✅ **优化提示词文件**：
- `src/method_optimized_additions.py` - 优化的提示词函数
- `src/prompts_optimized.py` - 完整的优化版提示词模块
- `src/method.py.backup` - 原文件备份
- `docs/PROMPT_OPTIMIZATION_README.md` - 优化说明

---

## 🔧 容器内执行步骤

### 步骤1：拉取最新代码

```bash
# 进入容器
docker exec -it <容器名> bash

# 拉取代码
cd /home/topgo-openseek
git pull
```

### 步骤2：应用优化到method.py

```bash
cd /home/topgo-openseek

# 方式1：直接追加优化函数到method.py末尾
cat src/method_optimized_additions.py >> src/method.py

# 方式2：手动复制（推荐）
# 编辑 src/method.py，在末尾粘贴优化函数
```

### 步骤3：修改main.py调用优化函数

编辑 `src/main.py`，找到 `evaluate` 函数中的 prompt 构建部分：

```python
# 找到类似这样的代码：
input_prompt = build_prompt(task_description, text2annotate)

# 替换为：
if task_id == 7:
    input_prompt = build_prompt_for_task7_optimized(task_description, text2annotate)
elif task_id == 4:
    input_prompt = build_prompt_for_task4_optimized(task_description, text2annotate)
else:
    input_prompt = build_prompt(task_description, text2annotate)
```

### 步骤4：检查vLLM服务

```bash
# 确认vLLM运行正常
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 应该看到: "id": "/home/Qwen/Qwen3-4B"
```

### 步骤5：重跑优化任务

```bash
# 设置环境变量
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

cd src

# 优先级1：重跑任务7（问题最严重）
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/

# 优先级2：重跑任务4
python main.py --task_id 4 --max_input_length 20000 --log_path_prefix ../outputs/
```

---

## 📊 一键执行脚本

创建并运行这个脚本：

```bash
cat > /home/topgo-openseek/apply_optimization.sh << 'EOF'
#!/bin/bash
set -e

echo "============================================================"
echo "应用提示词优化"
echo "============================================================"

cd /home/topgo-openseek

# 1. 应用优化函数
echo "[1/4] 应用优化函数到method.py..."
if ! grep -q "build_prompt_for_task7_optimized" src/method.py; then
    cat src/method_optimized_additions.py >> src/method.py
    echo "✅ 优化函数已添加"
else
    echo "⚠️  优化函数已存在，跳过"
fi

# 2. 修改main.py（自动）
echo ""
echo "[2/4] 修改main.py调用优化函数..."
if ! grep -q "build_prompt_for_task7_optimized" src/main.py; then
    # 备份
    cp src/main.py src/main.py.backup_$(date +%Y%m%d_%H%M%S)

    # 在prompt构建处添加条件判断
    # 这里需要手动编辑，因为自动替换可能有风险
    echo ""
    echo "⚠️  需要手动编辑 src/main.py"
    echo ""
    echo "请找到以下代码："
    echo "  input_prompt = build_prompt(task_description, text2annotate)"
    echo ""
    echo "替换为："
    echo "  if task_id == 7:"
    echo "      input_prompt = build_prompt_for_task7_optimized(task_description, text2annotate)"
    echo "  elif task_id == 4:"
    echo "      input_prompt = build_prompt_for_task4_optimized(task_description, text2annotate)"
    echo "  else:"
    echo "      input_prompt = build_prompt(task_description, text2annotate)"
    echo ""
    read -p "按回车继续..."
else
    echo "✅ main.py已修改"
fi

# 3. 设置环境
echo ""
echo "[3/4] 设置环境变量..."
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"
echo "✅ 环境变量已设置"

# 4. 运行任务
echo ""
echo "[4/4] 运行优化任务..."
echo ""
echo "选择要运行的任务："
echo "1. 只运行任务7（推荐首先测试）"
echo "2. 运行任务7和任务4"
echo "3. 运行所有任务"
read -p "请选择 (1-3): " choice

case $choice in
    1)
        tasks="7"
        ;;
    2)
        tasks="7 4"
        ;;
    3)
        tasks="1 2 3 4 5 6 7 8"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

cd src
for task_id in $tasks; do
    echo ""
    echo "========== 开始任务 $task_id =========="
    python main.py --task_id $task_id --max_input_length 15000 --log_path_prefix ../outputs/
done

cd ..
echo ""
echo "============================================================"
echo "优化完成！"
echo "============================================================"
EOF

chmod +x /home/topgo-openseek/apply_optimization.sh
/home/topgo-openseek/apply_optimization.sh
```

---

## 🎯 预期效果

| 任务 | 当前状态 | 优化后预期 |
|------|----------|-----------|
| 任务7 | 大量特殊字符"৷" | **消除特殊字符** |
| 任务4 | 96个null (19.2%) | **减少到<20个** |
| **分数** | 55.98分 | **60-65分** |
| **排名** | 第13名 | **前8名** |

---

## ⚠️ 重要提示

1. **先测试任务7** - 问题最明显，效果最容易验证
2. **检查模型** - 确保使用Qwen3-4B，不是Qwen2.5
3. **保留备份** - 已自动备份到 src/main.py.backup_*
4. **分步验证** - 每优化一个任务就检查效果

---

## 📝 验证优化效果

```bash
# 运行后检查任务7的结果
python3 << 'PYEOF'
import json

with open('outputs/openseek-7-v1.jsonl', 'r') as f:
    lines = f.readlines()

# 检查特殊字符
special = sum(1 for l in lines if '৷' in l or '\\u09ed' in l)
nulls = sum(1 for l in lines if '"prediction": null' in l)

print(f"特殊字符答案: {special}")
print(f"null预测: {nulls}")
print(f"优化目标: 特殊字符=0, null<10")
PYEOF
```

---

## 🆘 遇到问题？

1. **找不到优化函数** - 确认method.py末尾已添加函数
2. **main.py报错** - 检查缩进和导入
3. **仍然有特殊字符** - 检查是否真的调用了优化函数
4. **分数没提升** - 检查答案格式是否正确

---

## 📞 快速命令

```bash
# 检查优化函数是否已添加
grep "build_prompt_for_task7_optimized" /home/topgo-openseek/src/method.py

# 检查main.py是否调用优化函数
grep "build_prompt_for_task7_optimized" /home/topgo-openseek/src/main.py

# 查看最新结果
tail -5 /home/topgo-openseek/outputs/openseek-7-v1.jsonl
```
