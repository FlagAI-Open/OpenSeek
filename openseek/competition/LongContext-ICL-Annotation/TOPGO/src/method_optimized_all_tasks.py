"""
FlagOS OpenSeek 赛道三 - 全面优化提示词
针对所有需要优化的任务
"""

# ========== 任务2优化 - 词性标注 ==========

def build_prompt_for_task2_optimized(task_description: str, text2annotate: str) -> str:
    """任务2专用 - 词性标注，减少null，确保JSON格式正确"""
    prompt = f"""You are an expert at part-of-speech tagging in English text.

### Task Description
{task_description}

### CRITICAL OUTPUT REQUIREMENTS
1. **Output ONLY valid JSON** - no explanations, no markdown
2. **Required keys**: "nouns" and "verbs"
3. **Value format**: Lists of strings, even if empty
4. **Format**: <label>{{"nouns": ["word1", "word2"], "verbs": ["word3"]}}</label>

### Examples
Input: "The cat sat on the mat"
Output: <label>{{"nouns": ["cat", "mat"], "verbs": ["sat"]}}</label>

Input: "She runs fast"
Output: <label>{{"nouns": ["she"], "verbs": ["runs"]}}</label>

Input: "Beautiful weather today"
Output: <label>{{"nouns": ["weather", "today"], "verbs": []}}</label>

### Your Task
{text2annotate}

### Your Output (ONLY JSON in <label> tags, no markdown)
<label>"""
    return prompt


# ========== 任务4优化 - 代码生成 ==========

def build_prompt_for_task4_optimized(task_description: str, text2annotate: str) -> str:
    """任务4专用 - 代码生成，减少null，确保代码完整"""
    prompt = f"""You are an expert Python programmer. Generate complete, working Python code.

### Task Description
{task_description}

### CRITICAL CODE REQUIREMENTS
1. **Generate COMPLETE, WORKING code** - no placeholders or TODOs
2. **Include ALL necessary imports**
3. **Match EXACT function signature** as required
4. **Keep it SIMPLE but FUNCTIONAL**
5. **Handle edge cases**
6. **Format**: Wrap code in <label> tags

### Code Structure Template
<label>
```python
# Import statements
import ...

def function_name(parameters):
    \"\"\"
    Function description
    \"\"\"
    # Implementation
    result = ...

    return result
```
</label>

### Example
Task: Write a function that adds two numbers
<label>
```python
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b
```
</label>

### Your Task
{text2annotate}

### Generate COMPLETE Python code (in <label> tags, no placeholders)
<label>
```python
"""
    return prompt


# ========== 任务7优化 - 阅读理解 ==========

def build_prompt_for_task7_optimized(task_description: str, text2annotate: str) -> str:
    """任务7专用 - 阅读理解，消除特殊字符，提高准确性"""
    prompt = f"""You are an expert at answering trivia questions accurately and concisely.

### Task Description
{task_description}

### CRITICAL OUTPUT REQUIREMENTS
1. **Provide ONLY the answer** - no explanations or reasoning
2. **Use ONLY plain English** - NO special characters, NO symbols, NO non-English characters
3. **Keep it SHORT** - 1-5 words maximum
4. **Be ACCURATE** - this is critical for scoring
5. **Format**: <label>answer</label>

### Examples
Q: What is the capital of France?
A: <label>Paris</label>

Q: Who wrote Romeo and Juliet?
A: <label>William Shakespeare</label>

Q: What year did World War II end?
A: <label>1945</label>

Q: What is the largest planet?
A: <label>Jupiter</label>

Q: Who painted the Mona Lisa?
A: <label>Leonardo da Vinci</label>

### Your Question
{text2annotate}

### Your Answer (ONLY plain English, 1-5 words, in <label> tags)
<label>"""
    return prompt


# ========== 任务8优化 - Triton代码 ==========

def build_prompt_for_task8_optimized(task_description: str, text2annotate: str) -> str:
    """任务8专用 - Triton代码，确保功能完整"""
    prompt = f"""You are a Triton GPU kernel programming expert. Generate complete, working Triton code.

### Task Description
{task_description}

### MANDATORY CODE STRUCTURE

**1. Imports** (REQUIRED)
```python
import torch
import triton
import triton.language as tl
```

**2. Kernel Function** (with @triton.jit)
```python
@triton.jit
def kernel_name(
    # Pointers
    input_ptr, output_ptr,
    # Dimensions
    n_elements,
    # Block size (MUST be tl.constexpr)
    BLOCK_SIZE: tl.constexpr,
):
    # Step 1: Program ID
    pid = tl.program_id(axis=0)

    # Step 2: Offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Step 3: Mask (CRITICAL for boundaries)
    mask = offsets < n_elements

    # Step 4: Load with mask
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Step 5: Compute
    result = your_computation(data)

    # Step 6: Store with mask
    tl.store(output_ptr + offsets, result, mask=mask)
```

**3. Wrapper Function**
```python
def wrapper(input_tensor: torch.Tensor):
    output = torch.empty_like(input_tensor)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel_name[grid](input_tensor, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### CRITICAL REQUIREMENTS
- ALWAYS use mask for boundary checking
- ALWAYS declare BLOCK_SIZE as tl.constexpr
- ALWAYS handle edge cases
- NO placeholders or incomplete code

### Input
{text2annotate}

### Generate COMPLETE Triton code
<label>
```python
import torch
import triton
import triton.language as tl

@triton.jit
def your_kernel(...):
    # Implementation
    pass

def your_wrapper(...):
    # Implementation
    pass
```
</label>

Generate complete, working Triton code:"""
    return prompt


# ========== 通用优化提示词 ==========

def build_prompt_optimized(task_description: str, text2annotate: str) -> str:
    """优化后的通用提示词模板"""
    prompt = f"""You are a professional data annotation expert.

### Task
{task_description}

### Output Requirements
1. Follow the examples exactly
2. Provide clear, accurate results
3. Wrap your answer in <label> tags

### Examples
[[EXAMPLES]]

### Your Task
{text2annotate}

### Your Answer
<label>"""
    return prompt


# ========== 示例选择优化 ==========

def select_high_quality_examples(all_examples: list, max_examples: int = 100) -> list:
    """
    选择高质量示例

    优化点：
    1. 过滤低质量示例
    2. 按难度排序
    3. 控制总长度
    """
    valid_examples = []

    for ex in all_examples:
        try:
            output = ex.get('output', '')
            if not output:
                continue

            output_str = str(output)

            # 质量检查
            # 1. 不能为空
            if len(output_str.strip()) < 1:
                continue

            # 2. 不能包含特殊字符（任务7的问题）
            if '৷' in output_str or '\u09ed' in output_str:
                continue

            # 3. JSON格式检查（如果需要）
            # 4. 代码格式检查（如果需要）

            valid_examples.append(ex)

        except:
            continue

    # 按输入长度排序（简单到复杂）
    valid_examples.sort(key=lambda x: len(str(x.get('input', ''))))

    # 返回前max_examples个
    return valid_examples[:max_examples]
