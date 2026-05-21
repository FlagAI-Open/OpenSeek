# FlagOS OpenSeek 赛道三 - 优化部署指南

## 📊 当前状态

### 各任务有效率统计

| 任务 | 总数 | 有效 | 空值 | 有效率 | 状态 |
|------|------|------|------|--------|------|
| 任务1 | 500 | 500 | 0 | 100.0% | ✅ 优秀 |
| 任务2 | 500 | 469 | 31 | 93.8% | ⚠️ 需优化 |
| 任务3 | 500 | 500 | 0 | 100.0% | ✅ 优秀 |
| **任务4** | **500** | **404** | **96** | **80.8%** | 🔴 **需优化** |
| 任务5 | 500 | 500 | 0 | 100.0% | ✅ 优秀 |
| 任务6 | 500 | 492 | 8 | 98.4% | ✅ 良好 |
| **任务7** | **500** | **487** | **13** | **97.4%** | ⚠️ **需优化** |
| 任务8 | 166 | 166 | 0 | 100.0% | ✅ 优秀 |

### 问题诊断

#### 任务4 - 字符串连接任务
- **问题**: 模型输出思考过程而非答案，答案提取失败
- **样本分析**:
  - 正确答案: `pthat.o`
  - 错误答案: `标签里。首先，我得仔细看看示例...`（思考过程）
  - 部分正确: `'that`（提取不完整）
- **根本原因**: 提示词设计导致模型输出思考过程

#### 任务7 - 阅读理解任务
- **问题**: 约3%的预测包含思考标记污染
- **样本分析**:
  - 正确答案: `salman rushdie`
  - 污染答案: ``（思考结束标记）
- **根本原因**: 模型使用思考模式，答案提取未过滤

---

## 🚀 已实施的优化

### 1. 任务特定答案提取函数

#### `extract_task4_answer()`
- 专门处理字符串连接任务
- 过滤包含中文的思考过程
- 提取纯英文连接结果

#### `extract_task7_answer()`
- 专门处理阅读理解任务
- 过滤思考标记（``, ``等）
- 限制答案长度

#### `extract_task8_answer()`
- 专门处理Triton代码生成任务
- 支持长代码输出
- 提取代码块

### 2. 任务特定提示词

#### `build_prompt_task4()`
```python
# 简化的提示词结构
### 任务
将字符串列表连接成一个字符串。

### 规则
1. 直接将列表中的所有字符串连接在一起
2. 不要添加任何额外字符
3. 只输出连接后的结果

### 示例
输入: ['p', 'that.', 'o']
输出: <label>pthat.o</label>
```

#### `build_prompt_task7()`
```python
# 简洁答案提示词
### 任务
根据问题给出最简洁准确的答案。

### 规则
1. 只输出答案，不要解释
2. 答案要简洁（通常1-5个单词）
3. 答案必须包裹在<label>标签中
```

### 3. 参数传递优化

修改了 `main.py` 和 `method.py`，支持传递 `task_id` 参数：
- 提示词构建函数接收 `task_id`
- 答案提取函数接收 `task_id`
- 标注函数接收 `task_id`

---

## 📋 部署步骤

### 步骤1: 同步代码到容器

```bash
# 在容器内执行
cd /home/TOPGO

# 拉取最新代码
git pull origin master

# 如果有冲突，强制同步
git fetch origin
git reset --hard origin/master
```

### 步骤2: 备份现有结果

```bash
# 备份outputs目录
mkdir -p outputs_backup
cp -r outputs/* outputs_backup/
```

### 步骤3: 重新运行需要优化的任务

```bash
# 设置环境变量
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

# 确保vLLM服务运行
ps aux | grep vllm
# 如果没有运行，启动vLLM
# bash scripts/start_vllm.sh

# 重新运行任务4
cd src
python main.py --task_id 4 --max_input_length 10000 --log_path_prefix ../outputs/

# 重新运行任务7
python main.py --task_id 7 --max_input_length 10000 --log_path_prefix ../outputs/
```

### 步骤4: 运行后处理（可选）

如果无法重新运行，可以运行后处理脚本清理已有结果：

```bash
cd /home/TOPGO

# 处理任务4
python scripts/postprocess_results.py \
    --task_id 4 \
    --input outputs/openseek-4-v1.jsonl \
    --output outputs/openseek-4-v2-cleaned.jsonl

# 处理任务7
python scripts/postprocess_results.py \
    --task_id 7 \
    --input outputs/openseek-7-v1.jsonl \
    --output outputs/openseek-7-v2-cleaned.jsonl
```

### 步骤5: 验证结果

```bash
# 检查结果质量
python scripts/quick_stats.py

# 或使用详细分析
python scripts/analyze_results.py --outputs_dir outputs/
```

---

## 🔧 故障排除

### 问题1: API连接失败

```
[ERROR] Ascend API调用失败: Connection error
```

**解决方案**:
```bash
# 检查vLLM服务状态
curl http://localhost:8000/v1/models

# 如果没有响应，重启vLLM
pkill -f vllm
bash scripts/start_vllm.sh
```

### 问题2: 模型路径错误

```
The model Qwen3-4B-ascend-flagos does not exist.
```

**解决方案**:
已修改代码使用正确的模型路径 `/home/Qwen/Qwen3-4B`

### 问题3: Tokenizer加载失败

```
[WARN] 所有Tokenizer加载尝试均失败
```

**解决方案**:
```bash
# 检查模型路径
ls -la /home/Qwen/Qwen3-4B

# 如果模型不存在，设置正确的路径
export TOKENIZER_PATH="/path/to/Qwen3-4B"
```

---

## 📈 预期提升

| 任务 | 优化前有效率 | 预期优化后有效率 | 提升 |
|------|------------|----------------|------|
| 任务4 | 80.8% | 95%+ | +14% |
| 任务7 | 97.4% | 99%+ | +2% |
| 总体 | 96.0% | 98%+ | +2% |

---

## 📁 文件变更清单

### 修改的文件
- `src/method.py`:
  - 添加 `count_answer(task_id)` 支持
  - 添加 `build_prompt(task_id)` 支持
  - 新增 `extract_task4_answer()`
  - 新增 `extract_task7_answer()`
  - 新增 `extract_task8_answer()`
  - 新增 `build_prompt_task4()`
  - 新增 `build_prompt_task7()`

- `src/main.py`:
  - 传递 `task_id` 到提示词构建函数
  - 传递 `task_id` 到标注函数

### 新增的文件
- `scripts/postprocess_results.py`: 后处理脚本
- `scripts/run_postprocess.sh`: 一键后处理脚本
- `scripts/quick_stats.py`: 快速统计脚本
- `scripts/analyze_results.py`: 详细分析脚本
- `docs/OPTIMIZATION_DEPLOYMENT_GUIDE.md`: 本文档

---

## ⏭️ 下一步计划

### Phase 2: 系统性优化（如果时间允许）

1. **示例质量筛选**
   - 筛选格式正确的ICL示例
   - 过滤异常长度的示例

2. **Self-Consistency验证**
   - 多次采样取最一致答案
   - 提升答案准确率

3. **答案后处理管道**
   - 实现流水线式的后处理
   - 支持多种验证规则

---

## 📞 联系方式

如有问题，请联系团队负责人。
