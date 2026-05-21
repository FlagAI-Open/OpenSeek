# 🚀 提示词优化应用指南

## ✅ 已完成的优化

### 1. 任务7专用提示词
- **文件**: `src/method_optimized_additions.py`
- **函数**: `build_prompt_for_task7_optimized()`
- **优化点**:
  - 明确要求纯英文输出
  - 禁止特殊字符（如৷）
  - 限制答案长度1-5词
  - 提供清晰的示例

### 2. 任务4专用提示词
- **文件**: `src/method_optimized_additions.py`
- **函数**: `build_prompt_for_task4_optimized()`
- **优化点**:
  - 明确代码格式要求
  - 要求完整可运行的代码
  - 提供代码模板
  - 减少null预测

---

## 🔧 如何应用优化

### 方式1：手动修改main.py（推荐）

在`src/main.py`中找到任务处理部分，添加任务类型判断：

```python
# 在evaluate函数中，修改prompt构建逻辑
if task_id == 7:
    input_prompt = build_prompt_for_task7_optimized(task_description, text2annotate)
elif task_id == 4:
    input_prompt = build_prompt_for_task4_optimized(task_description, text2annotate)
else:
    input_prompt = build_prompt(task_description, text2annotate)
```

### 方式2：直接替换（快速）

```bash
# 备份原文件
cp src/method.py src/method.py.backup

# 将优化函数添加到method.py末尾
cat src/method_optimized_additions.py >> src/method.py

# 然后修改main.py调用新函数
```

---

## 📦 推送到Gitee

```bash
cd c:/D/compet/dcic/FlagOS开放计算全球挑战赛/TOPGO-track3-solution

git add src/method_optimized_additions.py
git add src/prompts_optimized.py
git commit -m "添加优化提示词：任务7避免特殊字符，任务4减少null预测"
git push origin master
```

---

## 🚀 容器内执行

```bash
# 1. 进入容器
docker exec -it <容器名> bash

# 2. 拉取最新代码
cd /home/topgo-openseek
git pull

# 3. 应用优化（手动修改main.py）
# 编辑 src/main.py，在prompt构建部分添加任务判断

# 4. 重跑优化任务
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

cd src

# 只重跑任务7（最高优先级）
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/

# 重跑任务4
python main.py --task_id 4 --max_input_length 20000 --log_path_prefix ../outputs/
```

---

## 📊 预期效果

| 任务 | 当前问题 | 优化后预期 |
|------|----------|-----------|
| 任务7 | 大量特殊字符"৷" | **消除特殊字符，准确率+5%** |
| 任务4 | 96个null (19.2%) | **减少到<10个null** |
| **总提升** | 55.98分 | **预期60-65分（前8名）** |

---

## ⚠️ 注意事项

1. **测试后再推送**: 先在本地测试提示词是否有效
2. **分批优化**: 先优化任务7，看效果后再优化其他任务
3. **保留备份**: method.py.backup已保存原版本
4. **验证模型**: 确保使用Qwen3-4B，不是Qwen2.5

---

## 🎯 快速开始

**最简单的应用方式**：

直接复制优化后的函数到`src/method.py`末尾，然后在`src/main.py`中调用即可！

```python
# 在 src/main.py 的 evaluate 函数中

# 原代码：
input_prompt = build_prompt(task_description, text2annotate)

# 改为：
if task_id == 7:
    input_prompt = build_prompt_for_task7_optimized(task_description, text2annotate)
elif task_id == 4:
    input_prompt = build_prompt_for_task4_optimized(task_description, text2annotate)
else:
    input_prompt = build_prompt(task_description, text2annotate)
```

---

## 📝 后续优化

如果效果明显，可以继续优化：
- 任务2：词性标注格式
- 任务8：Triton代码完整性
- ICL示例筛选

**目标**: 冲击前5名（71分以上）
