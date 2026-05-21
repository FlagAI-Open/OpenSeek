# FlagOS OpenSeek 赛道三 - 多智能体答案提取系统

## 🎯 系统概述

我们实现了一个**多智能体协作系统**来改进答案提取质量，解决以下问题：

1. **推理过程污染** - 答案中包含"因为"、"所以"等推理词
2. **答案不完整** - 模型推理到一半被截断
3. **格式错误** - 输出格式不符合要求
4. **答案提取失败** - 无法从模型输出中正确提取答案

---

## 🤖 智能体架构

### 四大智能体分工

```
┌─────────────────┐
│  模型输出文本   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  提取智能体     │ ← 提取候选答案
│ ExtractionAgent │   (label/answer标签、代码块、列表等)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  验证智能体     │ ← 验证答案正确性
│ValidationAgent  │   (污染检测、格式验证、置信度评分)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  格式化智能体   │ ← 确保输出格式
│FormattingAgent  │   (移除标点、标准化、任务特定格式)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   最终答案      │
└─────────────────┘
```

### 各智能体功能

#### 1. 提取智能体 (ExtractionAgent)
**职责：** 从模型输出中提取候选答案

**提取优先级：**
1. `<answer>` 标签
2. `<label>` 标签
3. 代码块 (\`\`\`python ... \`\`\`)
4. 列表格式 ([1, 2, 3])
5. 最后一行（对于短答案任务）

**关键代码：**
```python
# 提取<answer>标签
answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
answer_matches = re.findall(answer_pattern, text, re.DOTALL)

# 提取<label>标签
label_pattern = r'<label>\s*(.*?)\s*</label>'
label_matches = re.findall(label_pattern, text, re.DOTALL)
```

#### 2. 验证智能体 (ValidationAgent)
**职责：** 验证答案的正确性和质量

**验证规则：**
- ✅ 检查答案是否为空
- ✅ 检测推理过程污染（"因为"、"所以"等）
- ✅ 检查答案长度
- ✅ 验证任务特定格式（数字、Y/N等）
- ✅ 计算置信度评分

**关键代码：**
```python
# 污染检测
reasoning_keywords = ['因为', '所以', '首先', '然后', '我认为', '因此']
has_pollution = any(kw in answer for kw in reasoning_keywords)

# 任务特定验证
if task_id == 2:  # 计数任务
    if not answer.strip().isdigit():
        numbers = re.findall(r'\d+', answer)
        return numbers[-1] if numbers else None
```

#### 3. 格式化智能体 (FormattingAgent)
**职责：** 确保答案格式正确

**格式化规则：**
- 移除包裹标记（`<label>`, `<answer>`）
- 移除末尾标点符号
- 任务特定格式化（数字、Y/N、短答案等）

**关键代码：**
```python
# 移除包裹标记
answer = re.sub(r'</?(?:label|answer|think)>', '', answer)

# 任务特定格式化
if task_id == 6:  # MNLI任务
    answer_upper = answer.strip().upper()
    if answer_upper in ['YES', 'Y']:
        return 'Y'
    elif answer_upper in ['NO', 'N']:
        return 'N'

if task_id == 7:  # 阅读理解
    answer = answer.rstrip('。，,.!?')  # 移除末尾标点
```

---

## 📊 测试结果

### 单元测试

```
多智能体答案提取系统测试
======================================================================
【阅读理解 - 带推理过程】 ✅ 通过
【阅读理解 - 推理污染】 ✅ 通过
【计数任务 - 数字提取】 ✅ 通过
【MNLI - YES/NO转换】 ✅ 通过
【字符串连接】 ✅ 通过
======================================================================
测试结果: 通过 5/5
======================================================================
```

### 真实案例对比

| 任务 | 问题类型 | 原输出 | 多智能体输出 | 状态 |
|------|---------|--------|-------------|------|
| 任务2 | 计数 | "经过分析，这个句子包含3个名词，因此答案是3。" | "3" | ✅ |
| 任务4 | 字符串 | "<label>helloworld</label>" | "helloworld" | ✅ |
| 任务6 | MNLI | "<label>YES</label>" | "Y" | ✅ |
| 任务7 | 阅读理解 | "因为法国的首都是巴黎，所以答案是Paris。" | "Paris" | ✅ |

---

## 🚀 使用方法

### 方法1: 直接使用多智能体函数

```python
from multi_agent_method import count_answer_multi_agent

# 在 method.py 中替换原有的 count_answer 函数
prediction = count_answer_multi_agent(
    model_output,
    task_id=task_id,
    original_input=text2annotate
)
```

### 方法2: 集成到现有代码

修改 `src/method.py`:

```python
# 导入多智能体模块
from multi_agent_method import count_answer_multi_agent

def count_answer(text: str, task_id: int = None) -> Optional[str]:
    """提取答案（使用多智能体系统）"""
    return count_answer_multi_agent(text, task_id, '')
```

### 方法3: 在容器中运行

```bash
cd /home/topgo-openseek

# 拉取最新代码
git fetch origin && git reset --hard origin/master

# 运行任务（使用多智能体系统）
cd src
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/
```

---

## 📝 优化要点

### 已实施的优化

1. ✅ **任务7提示词优化**
   - 强制直接输出答案
   - 移除"答案:"前缀
   - 直接在label标签中开始答案

2. ✅ **max_tokens调整**
   - 任务7: 500 → 2000
   - 确保模型有足够空间完成推理

3. ✅ **多智能体答案提取**
   - 推理智能体：允许模型思考
   - 提取智能体：提取最终答案
   - 验证智能体：验证答案质量
   - 格式化智能体：确保格式正确

4. ✅ **推理过程过滤**
   - 过滤思考标记 (`<thinking>`, `heed>`)
   - 过滤推理关键词（因为、所以等）
   - 过滤中文推理过程

### 进一步优化方向

1. **提示词工程**
   - 为每个任务设计特定的提示词
   - 使用few-shot学习
   - 添加思维链引导

2. **模型参数调优**
   - 调整temperature（任务7使用0.1）
   - 调整top_p
   - 调整repetition_penalty

3. **后处理优化**
   - 添加答案验证规则
   - 添加置信度阈值
   - 添加重试机制

---

## 🎯 预期提升

| 任务 | 当前分数 | 预期分数 | 提升 |
|------|---------|---------|------|
| 任务2 | 93.8% | 98%+ | +4% |
| 任务4 | 80.8% | 95%+ | +14% |
| 任务6 | 98.4% | 99%+ | +1% |
| 任务7 | 97.4% | 99%+ | +2% |
| **整体** | **67.9** | **75+** | **+7** |

---

## 📦 文件清单

```
TOPGO-track3-solution/
├── src/
│   ├── method.py                      # 主方法文件（已集成优化）
│   └── multi_agent_method.py          # 多智能体系统
├── scripts/
│   ├── test_multi_agent.py            # 多智能体测试脚本
│   └── clean_task7_complete.py        # 任务7清理脚本
└── docs/
    └── MULTI_AGENT_GUIDE.md            # 本文档
```

---

## 🔧 容器运行命令

```bash
# 一键运行
cd /home/topgo-openseek
git pull origin master

export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

cd src

# 运行优化后的任务
python main.py --task_id 1 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 2 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 3 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 4 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 5 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 6 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 8 --max_input_length 15000 --log_path_prefix ../outputs/
```

---

**文档版本：** 1.0
**最后更新：** 2026-04-11
**提交ID：** aed3f90