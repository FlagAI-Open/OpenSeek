

# ====== 任务专用优化提示词函数 ======

def build_prompt_for_task7_optimized(task_description: str, text2annotate: str) -> str:
    """
    任务7专用提示词 - 阅读理解（优化版）

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


def build_prompt_for_task4_optimized(task_description: str, text2annotate: str) -> str:
    """
    任务4专用提示词 - 代码生成（优化版）

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
