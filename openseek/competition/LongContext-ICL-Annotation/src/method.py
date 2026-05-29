
import re
from collections import Counter
from transformers import AutoTokenizer

""" Here is an example of implementation of Long-Context Data Annotation. """

def build_prompt____(task_description: str, text2annotate: str) -> str:
    """
    Build a high-precision English prompt for long-context data annotation (optimized for Qwen3-4B).
    Core requirement: Final answer MUST be wrapped in <label> tags (no extra content outside tags).
    """
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specializing in long-context text labeling. "
        "Your work must strictly comply with the following rules, with the highest priority given to output format accuracy.\n\n"
        
        "### Core Annotation Task\n"
        f"{task_description}\n\n"
        
        "### Non-Negotiable Annotation Rules (Highest Priority)\n"
        "1. **Final Output Mandate**: Your annotation result MUST be wrapped in <label> tags — NO text, symbols, spaces, or explanations are allowed outside the tags.\n"
        "2. **Internal Reasoning Permission**: You may perform logical reasoning, text analysis, or context comprehension internally (in your thought process), but NONE of these thoughts may appear in the final output.\n"
        "3. **Label Format Strictness**: <label> is the opening tag and </label> is the closing tag — they must appear in pairs, with NO extra spaces or characters inside the tags (e.g., <label>  Good Review  </label> is invalid).\n"
        "4. **Prohibited Outputs**: \n"
        "   - ❌ Prohibited: 'After analysis, this is a positive review: <label>Good Review</label>' (extra text outside tags)\n"
        "   - ❌ Prohibited: 'Bad Review' (missing <label> tags entirely)\n"
        "   - ❌ Prohibited: '<label>Bad Review' (unpaired/closing tag missing)\n\n"
        
        "### Correct vs. Incorrect Examples\n"
        "✅ Correct Example 1: <label>answer</label>\n"
        "✅ Correct Example 2: <label>Bad Review</label>\n"
        "❌ Incorrect Example 1: I think this review is negative → <label>Bad Review</label>\n"
        "❌ Incorrect Example 2: <label>  Neutral Review  </label> (extra spaces inside tags)\n"
        "❌ Incorrect Example 3: Neutral Review (no label tags)\n\n"
        
        "### Reference Annotation Examples\n"
        "{EXAMPLES}\n\n"
        
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        
        "### Final Output Command (Re-emphasized)\n"
        "You may complete any internal reasoning process, but your FINAL OUTPUT MUST consist solely of the annotation result wrapped in <label> tags (no other content whatsoever).\n"
        "Annotation Result: "
    )
    return prompt

def build_prompt(task_description: str, text2annotate: str) -> str:
    """
    Construct a high-precision prompt for long-context data annotation (optimized for Qwen3-4B).
    task_description: Clear description of the annotation task (e.g., "Classify English product reviews as Good Review/Bad Review").
    text2annotate: The text to be annotated (single text or batch texts).
    """
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, and ensure the final annotation result is 100% enclosed in <label> tags.\n\n"
        
        "### Core Task\n"
        f"{task_description}\n\n"
        
        "### Critical Annotation Guidelines\n"
        "1. **Example Learning Requirement**: Thoroughly analyze and fully learn from the annotation logic, format, and criteria in the Examples section. "
        "Your annotation must align with the style, judgment standards, and tag usage shown in the examples.\n"
        "2. **Thinking Process**: You may (and are encouraged to) explain your annotation reasoning step by step (e.g., key information extraction, judgment basis, rule matching).\n"
        "3. **Mandatory Output Rule**: Regardless of any thinking process you provide, your final annotation result MUST be enclosed in <label> tags (this is non-negotiable).\n"
        "   - Correct example: \n"
        "     Reasoning: This review mentions 'excellent quality' and 'very satisfied', which meets the criteria for a Good Review.\n"
        "     <label>Good Review</label>\n"
        "   - Wrong example 1 (missing tags): This review is negative.\n"
        "   - Wrong example 2 (incomplete tags): Bad Review</label>\n"
        "4. **Length Adaptation**: For long texts, maintain complete thinking process and ensure the final <label> tags contain the accurate annotation result (no truncation).\n\n"
        
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        
        "### Final Requirement Summary\n"
        "1. You can (and should) provide clear thinking process for your annotation.\n"
        "2. The final annotation result MUST be wrapped in <label> tags (no exceptions).\n"
        "3. All annotation logic must strictly follow the examples provided above.\n"
    )
    return prompt

def build_prompt_cot(task_description: str, text2annotate: str, task_id: int) -> str:
    """
    Build a Chain-of-Thought (CoT) prompt for complex reasoning tasks (Task 3, 8).
    This encourages the model to show step-by-step reasoning before final answer.
    """
    if task_id == 3:
        # Task 3: Collatz Conjecture - Mathematical Reasoning
        prompt = (
            "### Role Definition\n"
            "You are a mathematical reasoning expert specializing in the Collatz conjecture. "
            "You excel at systematic step-by-step mathematical reasoning and verification.\n\n"
            
            "### Core Task\n"
            f"{task_description}\n\n"
            
            "### Critical Reasoning Guidelines\n"
            "1. **Step-by-Step Reasoning**: For each input number, you MUST show your complete reasoning process:\n"
            "   - Step 1: Identify the current number\n"
            "   - Step 2: Apply the Collatz rule (if even: n/2; if odd: 3n+1)\n"
            "   - Step 3: Calculate the next number\n"
            "   - Step 4: Continue until reaching 1\n"
            "   - Step 5: Determine the closest integer to 1\n\n"
            
            "2. **Verification**: Always verify your calculations:\n"
            "   - Check if the rule was applied correctly\n"
            "   - Confirm the sequence reaches 1\n"
            "   - Double-check the final answer\n\n"
            
            "3. **Output Format**: Your response must follow this structure:\n"
            "   **Reasoning Process:**\n"
            "   [Show your step-by-step calculations here]\n\n"
            "   **Final Answer:** <label>[closest integer]</label>\n\n"
            
            "### Examples (Must Be Fully Followed)\n"
            "[[EXAMPLES]]\n\n"
            
            "### Text to Annotate\n"
            f"{text2annotate}\n\n"
            
            "### Final Requirement Summary\n"
            "1. Show your complete step-by-step reasoning process.\n"
            "2. Verify each calculation step.\n"
            "3. Final answer MUST be wrapped in <label> tags.\n"
        )
    elif task_id == 8:
        # Task 8: Kernel Generation - Code Generation
        prompt = (
            "### Role Definition\n"
            "You are an expert programmer specializing in Linux kernel development. "
            "You excel at writing correct, efficient, and well-structured kernel code.\n\n"
            
            "### Core Task\n"
            f"{task_description}\n\n"
            
            "### Critical Code Generation Guidelines\n"
            "1. **Step-by-Step Approach**: Before writing code, think through:\n"
            "   - Step 1: Understand the kernel function requirements\n"
            "   - Step 2: Identify necessary kernel APIs and data structures\n"
            "   - Step 3: Design the function structure\n"
            "   - Step 4: Write the code with proper error handling\n"
            "   - Step 5: Review for common kernel coding issues\n\n"
            
            "2. **Code Quality Requirements**:\n"
            "   - Use correct kernel APIs (e.g., copy_from_user, copy_to_user)\n"
            "   - Handle all error cases properly\n"
            "   - Follow kernel coding style\n"
            "   - Ensure memory safety\n\n"
            
            "3. **Output Format**: Your response must follow this structure:\n"
            "   **Analysis:**\n"
            "   [Explain your approach and reasoning]\n\n"
            "   **Code:**\n"
            "   <label>[your complete kernel code here]</label>\n\n"
            
            "### Examples (Must Be Fully Followed)\n"
            "[[EXAMPLES]]\n\n"
            
            "### Text to Annotate\n"
            f"{text2annotate}\n\n"
            
            "### Final Requirement Summary\n"
            "1. Analyze the requirements step-by-step.\n"
            "2. Write correct kernel code with proper error handling.\n"
            "3. Final code MUST be wrapped in <label> tags.\n"
        )
    else:
        # Fallback to standard prompt for other tasks
        prompt = build_prompt(task_description, text2annotate)
    
    return prompt

def build_prompt_backup(task_description:str, text2annotate:str)->str:
    """
        Construct the prompt for annotation based on the task description.
        task_description: 
            The description of the annotation task. 
            For example, ``Given an English language product review, 
            determine if it is a Good Review or a Bad Review.`` 
        text2annotate:
            The text that needs to be annotated.
            For example, ``My son received this book as a gift. I was extremely disappointed.``
    """
    prompt = (
        "You are a data annotation assistant. "
        "Your task is to label the given texts according to the task description "
        "and annotation guidelines provided below.\n\n"
        f"[Task Description]\n {task_description}\n\n"
        "[Examples]\n {EXAMPLES}\n\n"
        "Please follow these instructions when labeling:\n"
        "1. **Output Format**: Annotate the text directly by wrapping each labeled "
        "span with <label> tags in the following format: <label> annotation result </label>.\n"
        # "2. Do not add any extra text, explanations, or commentary in the labeled spans.\n\n"
        f"[Task Description (repeat)] \n {task_description}\n\n"
        f"[Input Texts]\n {text2annotate}\n\n"
        "Please output the annotation results: "
    )
    return prompt

def select_examples_backup(all_examples:list[dict], task_description:str, text2annotate:str)->str:
    """
        Select examples from all_examples to fit into the target context length.
        all_examples:
            A list of examples, where each example is a dict with keys 'input', 'output', and 'length'.
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review", "length": 79``},
        task_description:
            The description of the annotation task which may be used for example evaluation. 
            For example, ``Given an English language product review, 
            determine if it is a Good Review or a Bad Review.`` 
        text2annotate:
            The text that needs to be annotated  which may be used for example retrieval.
            For example, ``My son received this book as a gift. I was extremely disappointed.``
        
    """
    # Notice that the maximum context length is restricted.
    target_length = 10_000
    
    input_list = [example['input'] for example in all_examples]
    output_list = [example['output'][0] for example in all_examples]
    length_list = [example['length'] for example in all_examples]
    
    # <label> have 2 tokens; </label> have 3 tokens; \n have 1 token; # have 1 token.
    examples_str, token_num = "", 0
    for i, (input_text, output_text, length) in enumerate(zip(input_list, output_list, length_list)):
        if length + token_num <= target_length:
            token_num += (length + 2 + 3 + 1 + 1)
            example_str = f"# {input_text} <label> {output_text} </label>\n"
            examples_str += example_str
        else:
            return examples_str, i
    return examples_str

# M15优化：全局缓存机制
_tokenizer_cache = None
_example_length_cache = {}

def get_tokenizer():
    """M15优化：获取缓存的tokenizer实例"""
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = AutoTokenizer.from_pretrained("/root/Qwen3-4B", trust_remote_code=True)
    return _tokenizer_cache

def compute_example_length(example: dict, tokenizer) -> int:
    """
    M15优化：计算单个示例的token长度，带缓存
    """
    # 使用示例的唯一标识作为缓存键
    example_key = f"{example['input'][:100]}_{example['output'][0] if isinstance(example['output'], list) else example['output']}"
    
    if example_key in _example_length_cache:
        return _example_length_cache[example_key]
    
    # 计算长度
    input_text = example['input']
    output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
    
    input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
    output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    length = input_tokens + output_tokens
    
    # 缓存结果
    _example_length_cache[example_key] = length
    
    return length

def precompute_all_examples_length(all_examples: list[dict]):
    """
    M15优化：预先计算所有示例的长度
    """
    global _example_length_cache
    tokenizer = get_tokenizer()
    
    for i, example in enumerate(all_examples):
        try:
            compute_example_length(example, tokenizer)
        except (KeyError, IndexError) as e:
            print(f"警告：示例{i}长度计算失败，跳过")
            continue

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    """
    M15优化版本：示例缓存与长度预计算方案
    通过预先计算和缓存示例长度，提高运行效率
    
    Parameters:
        all_examples: 所有示例列表，每个示例包含'input'和'output'键
        task_description: 任务描述
        text2annotate: 待标注文本
    """
    # M15优化：预先计算所有示例长度（仅第一次调用时执行）
    if not _example_length_cache:
        print("M15优化：开始预计算所有示例长度...")
        precompute_all_examples_length(all_examples)
        print(f"M15优化：长度预计算完成，已缓存{len(_example_length_cache)}个示例")
    
    # 获取缓存的tokenizer
    tokenizer = get_tokenizer()
    
    # 最大上下文长度限制
    target_length = 8192
    
    examples_str, token_num = "", 0
    selected_count = 0
    
    # 遍历所有示例，使用缓存的长度信息
    for i, example in enumerate(all_examples):
        try:
            input_text = example['input']
            output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
            
            # M15优化：使用缓存的长度信息
            length = compute_example_length(example, tokenizer)
            
            # 校验当前示例是否能加入
            if length + token_num <= target_length:
                token_num += (length + 2 + 3 + 1 + 1)  # <label>2 + </label>3 + \n1 + #1
                example_str = f"# {input_text} <label> {output_text} </label>\n"
                examples_str += example_str
                selected_count += 1
            else:
                # 超过长度限制，停止选择
                break
        except (KeyError, IndexError) as e:
            print(f"警告：示例{i}缺少必要键或格式错误，跳过该示例")
            continue
    
    print(f"M15优化：从{len(all_examples)}个示例中选择了{selected_count}个示例（使用缓存长度信息）")
    return examples_str




def count_answer(text: str) -> tuple[list, dict]:
    """
    提取字符串中<label>标签内的所有内容（字符串形式），统计出现次数最多的内容
    :param text: 包含<label>标签的原始字符串
    :return: 出现次数最多的内容列表、所有内容的频次统计字典
    """
    pattern = r'<label>\s*(.+?)\s*</label>'
    content_matches = re.findall(pattern, text, re.DOTALL) 
    
    content_counter = Counter(content_matches)
    if not content_counter:
        return None
    
    max_count = max(content_counter.values())
    answer = [content for content, count in content_counter.items() if count == max_count]
    
    return answer[0]


def annotate_nvidia(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (nvidia GPU).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import requests
    URL="http://0.0.0.0:2026/v1/completions"
    
    data = {
        "model": "../Qwen3-4B",
        "prompt": input_prompt,
        "max_tokens": 1024, # max_token = 10k
    }

    try:
        resp = requests.post(URL, json=data)
        whole_result = resp.json()["choices"][0]["text"]
    except Exception as e:
        whole_result = "None"


    prediction = count_answer(whole_result)
    return prediction

def annotate_ascend(input_prompt:str, task_id:int=None)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (Huawei Ascend).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
        
        Optimization for Account 3: Differentiated strategy based on task type
        - Task 3, 4: CoT reasoning with lower temperature (effective for math and string tasks)
        - Task 8: Standard configuration (CoT harmful for code generation)
        - Other tasks: Moderate temperature for balanced performance
    """
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "/root/Qwen3-4B"

    # Adjust temperature based on task (Differentiated Strategy)
    if task_id in [3, 4]:
        # Lower temperature for CoT reasoning tasks (Task 3: math, Task 4: strings)
        # This reduces randomness and improves accuracy
        temperature = 0.3
    elif task_id == 8:
        # Standard temperature for code generation (CoT was harmful in Account 2)
        temperature = 0.7
    else:
        # Moderate temperature for other tasks (balanced randomness and accuracy)
        temperature = 0.5

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt}
    ]
    
    # Adjust max_tokens based on task
    if task_id in [3, 4]:
        # Increased max_tokens for CoT tasks (supports longer reasoning chains)
        max_tokens = 2048
    else:
        # Standard max_tokens for other tasks
        max_tokens = 1024
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=False,
    )
    whole_result = response.choices[0].message.content
    
    # Special handling for Task 8 (code generation): return raw model output
    # Task 8 generates Triton code without <label> tags
    if task_id == 8:
        return whole_result.strip()
    
    # For other tasks, extract label-tagged content
    prediction = count_answer(whole_result)
    return prediction
