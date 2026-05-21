"""
FlagOS OpenSeek 赛道三 - TOPGO团队
优化版提示词模块

优化重点：
1. 任务7：明确要求纯英文、无特殊字符
2. 任务4：明确代码格式、减少null
3. 任务8：强化Triton代码结构
"""

import re
from typing import Optional


def build_prompt(task_description: str, text2annotate: str) -> str:
    """
    构建高质量的标注提示词（通用版本）
    """
    prompt = (
        "### 角色定义\n"
        "你是一个专业的数据标注专家。你的工作必须严格遵循任务规则，充分学习提供的示例。\n\n"

        "### 核心任务\n"
        f"{task_description}\n\n"

        "### 关键标注指南\n"
        "1. **示例学习**: 完全学习示例的标注逻辑和格式。\n"
        "2. **输出规则**: 最终结果必须包裹在<label>标签中。\n"
        "   - 正确: <label>答案</label>\n\n"

        "### 示例\n"
        "[[EXAMPLES]]\n\n"

        "### 待标注文本\n"
        f"{text2annotate}\n\n"

        "### 最终要求\n"
        "提供最终标注结果，包裹在<label>标签中。\n"
    )
    return prompt


def build_prompt_for_task7(task_description: str, text2annotate: str) -> str:
    """
    任务7专用提示词 - 阅读理解

    优化目标：
    1. 避免特殊字符（如৷）
    2. 要求简洁准确的答案
    3. 明确输出格式
    """
    prompt = f"""You are an expert at answering trivia questions concisely and accurately.

### Task Description
{task_description}

### CRITICAL OUTPUT REQUIREMENTS
1. **Provide ONLY the answer** - no explanations, no reasoning
2. **Use ONLY plain English text** - no special characters, no symbols, no non-English characters
3. **Keep it SHORT** - 1-5 words maximum
4. **Format**: Wrap your answer in <label> tags
   - Correct: <label>Paris</label>
   - Correct: <label>William Shakespeare</label>
   - Wrong: <label>৷</label>
   - Wrong: <label>The answer is Paris</label>

### Examples
Question: What is the capital of France?
Answer: <label>Paris</label>

Question: Who wrote Romeo and Juliet?
Answer: <label>William Shakespeare</label>

Question: In what year did World War II end?
Answer: <label>1945</label>

Question: What is the largest planet in our solar system?
Answer: <label>Jupiter</label>

### Your Question
{text2annotate}

### Your Answer (REMEMBER: Only plain English, 1-5 words, in <label> tags)
<label>"""
    return prompt


def build_prompt_for_task4(task_description: str, text2annotate: str) -> str:
    """
    任务4专用提示词 - 代码生成

    优化目标：
    1. 减少null预测
    2. 明确代码格式要求
    3. 提供完整示例
    """
    prompt = f"""You are an expert Python programmer. Generate complete, working Python code.

### Task Description
{task_description}

### CRITICAL CODE REQUIREMENTS
1. **Generate COMPLETE code** - include all imports and functions
2. **Match the EXACT function signature** required
3. **Make it WORK** - no placeholders, no TODOs
4. **Keep it SIMPLE** - minimal but complete
5. **Format**: Wrap code in <label> tags

### Code Template
<label>
```python
# Required imports
import ...

def required_function_name(parameters):
    \"\"\"
    Function implementation
    \"\"\"
    # Your code here
    return result
```
</label>

### Example
Task: Create a function that adds two numbers
Answer:
<label>
```python
def add_numbers(a, b):
    return a + b
```
</label>

### Your Task
{text2annotate}

### Your Complete Code (in <label> tags)
<label>
```python
"""
    return prompt


def build_prompt_for_triton(task_description: str, text2annotate: str) -> str:
    """
    任务8专用提示词 - Triton代码生成（优化版）

    优化目标：
    1. 更清晰的代码结构要求
    2. 完整的示例模板
    3. 强调边界处理
    """
    prompt = f"""You are a Triton GPU programming expert. Generate complete, working Triton kernel code.

### Task Description
{task_description}

### MANDATORY CODE STRUCTURE
Your code MUST include ALL of these components:

1. **Imports** (required)
```python
import torch
import triton
import triton.language as tl
```

2. **Kernel Function** (with @triton.jit decorator)
```python
@triton.jit
def kernel_name(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Step 1: Get program ID
    pid = tl.program_id(axis=0)

    # Step 2: Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Step 3: Create mask (CRITICAL!)
    mask = offsets < n_elements

    # Step 4: Load data with mask
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Step 5: Compute
    result = your_computation(data)

    # Step 6: Store with mask
    tl.store(output_ptr + offsets, result, mask=mask)
```

3. **Wrapper Function**
```python
def wrapper(input_tensor: torch.Tensor):
    output = torch.empty_like(input_tensor)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel_name[grid](input_tensor, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### CRITICAL REMINDERS
- ALWAYS use mask for boundary checking
- ALWAYS declare BLOCK_SIZE as tl.constexpr
- ALWAYS use proper dtype conversions
- NO placeholders or TODOs

### Input
{text2annotate}

### Your Complete Triton Code
<label>
```python
import torch
import triton
import triton.language as tl

@triton.jit
def your_kernel_name(
    # Define your parameters
):
    # Implement kernel
    pass

def your_wrapper_name(
    # Define wrapper
):
    pass
```
</label>

Generate complete code now:"""
    return prompt


def build_prompt_for_task2(task_description: str, text2annotate: str) -> str:
    """
    任务2专用提示词 - 词性标注

    优化目标：
    1. 明确输出格式
    2. 减少格式错误
    """
    prompt = f"""You are an expert at part-of-speech tagging.

### Task Description
{task_description}

### Output Format Requirements
1. **Output format**: JSON object
2. **Required keys**: "nouns" and "verbs"
3. **Value type**: List of strings

### Example
Input: "The cat sat on the mat"
Output: <label>{{"nouns": ["cat", "mat"], "verbs": ["sat"]}}</label>

### Your Task
{text2annotate}

### Your Output (in <label> tags, JSON format only)
<label>"""
    return prompt


def select_examples(all_examples: list, task_description: str, text2annotate: str,
                    tokenizer=None) -> str:
    """
    选择高质量示例（优化版）

    优化点：
    1. 过滤低质量示例
    2. 按长度排序
    3. 控制总长度
    """
    target_length = 6000  # 减少到6K，留更多空间给输出

    # 过滤高质量示例
    valid_examples = []
    for ex in all_examples:
        try:
            output = ex.get('output', '')
            if output and len(str(output)) > 0:
                # 检查输出质量
                output_str = str(output)
                # 移除包含特殊字符的示例
                if not re.search(r'[৷\u09ed]', output_str):
                    valid_examples.append(ex)
        except:
            pass

    # 按长度排序（简单到复杂）
    valid_examples.sort(key=lambda x: len(str(x.get('input', ''))))

    examples_str = ""
    token_count = 0

    for ex in valid_examples:
        try:
            input_text = ex.get('input', '')
            output_text = ex.get('output', '')

            example_str = f"Input: {input_text}\nOutput: <label>{output_text}</label>\n\n"

            # 简单的token估算（字符数/4）
            example_tokens = len(example_str) // 4

            if token_count + example_tokens > target_length:
                break

            examples_str += example_str
            token_count += example_tokens
        except:
            continue

    return examples_str


def count_answer(text: str) -> Optional[str]:
    """
    提取标签中的答案（优化版）

    优化点：
    1. 更健壮的提取逻辑
    2. 清理特殊字符
    3. 验证答案质量
    """
    if not text:
        return None

    # 尝试提取<label>标签内容
    label_pattern = r'<label>(.*?)</label>'
    matches = re.findall(label_pattern, text, re.DOTALL)

    if matches:
        answer = matches[0].strip()

        # 清理markdown代码块标记
        answer = re.sub(r'```\w*\n?', '', answer)
        answer = answer.strip()

        # 检查是否包含特殊字符（任务7的问题）
        if re.search(r'[৷\u09ed]', answer):
            # 尝试移除特殊字符
            answer = re.sub(r'[৷\u09ed]', '', answer).strip()

        # 验证答案不为空
        if len(answer) < 1:
            return None

        return answer

    # 如果没有找到标签，尝试其他提取方式
    # 代码块提取
    code_pattern = r'```(?:python)?\s*(.*?)\s*```'
    code_matches = re.findall(code_pattern, text, re.DOTALL)
    if code_matches:
        return code_matches[0].strip()

    # JSON提取
    json_pattern = r'\{[^}]+\}'
    json_matches = re.findall(json_pattern, text)
    if json_matches:
        return json_matches[0]

    return None
