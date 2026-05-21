# 🎯 FlagOS OpenSeek 赛道三 - 全面优化方案

## 📊 问题诊断结果

### 任务7问题（严重）
- **null预测**: 13个 (2.6%)
- **特殊字符**: 大量"৷"（孟加拉数字符）
- **根本原因**: 模型输出了无效字符而非正确答案

### 任务4问题（严重）
- **null预测**: 96个 (19.2%)
- **代码长度**: 平均仅70-90字符（太短）
- **根本原因**: 代码生成提示词不够明确

### 任务2问题
- **格式问题**: 469个可能不符合要求
- **根本原因**: 输出格式不正确

### 任务8问题
- **代码质量**: 虽然都包含triton关键字，但功能可能不完整
- **根本原因**: 缺少完整的代码结构要求

---

## 🚀 优化方案

### 方案1：优化提示词（最重要）

#### 任务7优化

**当前问题**: 模型输出特殊字符"৷"

**优化策略**:
```python
# 在 src/method.py 中优化 build_prompt_for_task7()

def build_prompt_for_task7(task_description, text2annotate):
    prompt = f"""{task_description}

IMPORTANT INSTRUCTIONS:
1. Provide a concise answer (1-5 words maximum)
2. Use ONLY plain English text
3. Do NOT use any special characters, symbols, or non-English characters
4. Do NOT include explanations or reasoning
5. Output ONLY the answer, nothing else

Examples:
Question: What is the capital of France?
Answer: Paris

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: {text2annotate}
Answer:"""
    return prompt
```

#### 任务4优化

**当前问题**: 19.2%的null预测

**优化策略**:
```python
def build_prompt_for_task4(task_description, text2annotate):
    prompt = f"""{task_description}

IMPORTANT INSTRUCTIONS:
1. Generate complete, working Python code
2. Include all necessary imports
3. Follow the exact function signature required
4. Ensure the code is syntactically correct
5. Include comments for clarity

Example format:
```python
def function_name(param1, param2):
    # Your implementation here
    result = param1 + param2
    return result
```

Now implement the following:
{text2annotate}

Provide the complete code:"""
    return prompt
```

#### 任务8优化

**当前问题**: 代码功能可能不完整

**优化策略**:
```python
def build_prompt_for_triton(task_description, text2annotate):
    prompt = f"""{task_description}

TRITON KERNEL STRUCTURE:
1. Import required libraries (torch, triton, triton.language as tl)
2. Define the kernel function with @triton.jit decorator
3. Implement the kernel logic with proper parallelization
4. Define a wrapper function to launch the kernel
5. Handle edge cases and boundary conditions

Example structure:
```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(...):
    # Kernel implementation
    pass

def wrapper(...):
    # Wrapper to launch kernel
    pass
```

Now implement:
{text2annotate}

Provide complete Triton code:"""
    return prompt
```

---

### 方案2：优化ICL示例

#### 当前问题
- 使用前100个示例，质量参差不齐
- 没有根据任务难度排序

#### 优化策略

```python
# 在 src/main.py 中优化示例选择

def select_high_quality_examples(examples, n=100):
    """选择高质量示例"""
    # 1. 过滤掉有问题的示例
    valid_examples = []
    for ex in examples:
        if ex.get('output') and len(str(ex['output'])) > 0:
            # 检查输出质量
            output = str(ex['output'])
            # 移除包含特殊字符的示例
            if not any(char in output for char in ['৷', '<', '>', '\\u']):
                valid_examples.append(ex)

    # 2. 按长度排序（简单到复杂）
    valid_examples.sort(key=lambda x: len(str(x.get('input', ''))))

    # 3. 选择前n个
    return valid_examples[:n]
```

---

### 方案3：后处理优化

#### 任务7清理

```python
def clean_task7_prediction(pred):
    """清理任务7的预测"""
    if pred is None:
        return None

    # 移除特殊字符
    pred = re.sub(r'[^\w\s\-\'\.]', '', str(pred))

    # 移除多余空格
    pred = ' '.join(pred.split())

    # 限制长度
    if len(pred) < 3:
        return None

    return pred.strip()
```

#### 任务4验证

```python
def validate_task4_code(code):
    """验证任务4的代码"""
    if not code or len(code) < 20:
        return None

    # 检查是否包含函数定义
    if 'def ' not in code:
        return None

    # 尝试编译检查语法
    try:
        compile(code, '<string>', 'exec')
        return code
    except:
        return None
```

---

### 方案4：参数调优

#### 推荐参数

```python
# 在 src/method.py 中

# 任务7（阅读理解）
params_task7 = {
    'temperature': 0.1,  # 更确定性
    'max_tokens': 50,    # 短答案
    'top_p': 0.9
}

# 任务4（代码生成）
params_task4 = {
    'temperature': 0.3,
    'max_tokens': 500,   # 足够长的代码
    'top_p': 0.95
}

# 任务8（Triton代码）
params_task8 = {
    'temperature': 0.2,
    'max_tokens': 2000,  # 完整代码
    'top_p': 0.95
}
```

---

## 📈 预期提升

| 优化项 | 预期提升 |
|--------|----------|
| 任务7提示词优化 | +3-5分 |
| 任务4提示词优化 | +2-4分 |
| 任务8代码优化 | +2-3分 |
| ICL示例优化 | +3-5分 |
| 后处理优化 | +1-2分 |
| **总计** | **+11-19分** |

**目标分数**: 67-75分（前5名）

---

## 🔧 实施步骤

### 步骤1：修改method.py（本地）

在本地修改 `src/method.py` 中的提示词模板

### 步骤2：推送代码

```bash
git add src/method.py
git commit -m "优化提示词：改进任务4、7、8的提示词模板"
git push origin master
```

### 步骤3：容器内重跑

```bash
# 在容器内
cd /home/topgo-openseek
git pull
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

cd src
# 只重跑有问题的任务
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 4 --max_input_length 20000 --log_path_prefix ../outputs/
python main.py --task_id 8 --max_input_length 20000 --log_path_prefix ../outputs/
```

### 步骤4：后处理

```bash
# 运行清理脚本
python scripts/clean_predictions.py
```

---

## 💡 关键建议

1. **优先级1**: 优化任务7的提示词（解决特殊字符问题）
2. **优先级2**: 优化任务4的提示词（减少null预测）
3. **优先级3**: 优化任务8的代码结构要求
4. **优先级4**: 优化ICL示例选择
5. **优先级5**: 后处理优化

**立即行动**: 我可以帮您修改 `src/method.py` 中的提示词！
