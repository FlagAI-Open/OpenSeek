
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

def compute_similarity(text1: str, text2: str) -> float:
    """
    M02优化：计算两个文本的相似度（基于词重叠）
    使用简单的词重叠计算相似度，避免引入复杂依赖
    """
    # 将文本转换为小写并分词
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # 计算Jaccard相似度
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    """
    M02优化版本：按样本动态选例方案
    为每个样本动态选择最相关的示例，而非按固定顺序选择
    
    Parameters:
        all_examples: 所有示例列表，每个示例包含'input'和'output'键
        task_description: 任务描述
        text2annotate: 待标注文本（用于相似度计算）
    """
    # 初始化Qwen3-4B的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/Qwen3-4B", trust_remote_code=True)
    
    # 最大上下文长度限制
    target_length = 8192
    
    # M02核心：为每个示例计算与待标注文本的相似度
    example_scores = []
    for i, example in enumerate(all_examples):
        try:
            input_text = example['input']
            output_text = example['output'][0]
            
            # 计算token长度
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens
            
            # 计算相似度
            similarity = compute_similarity(text2annotate, input_text)
            
            example_scores.append({
                'index': i,
                'example': example,
                'length': length,
                'similarity': similarity,
                'input_text': input_text,
                'output_text': output_text
            })
        except (KeyError, IndexError) as e:
            print(f"警告：示例{i}缺少必要键或格式错误，跳过该示例")
            continue
    
    # M02核心：按相似度降序排序，选择最相关的示例
    example_scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 动态选择示例，确保不超过token限制
    examples_str, token_num = "", 0
    selected_count = 0
    
    for scored_example in example_scores:
        length = scored_example['length']
        input_text = scored_example['input_text']
        output_text = scored_example['output_text']
        
        # 检查是否超过长度限制
        if length + token_num <= target_length:
            # 累加token数（示例文本 + 格式符号）
            token_num += (length + 2 + 3 + 1 + 1)  # <label>2 + </label>3 + \n1 + #1
            example_str = f"# {input_text} <label> {output_text} </label>\n"
            examples_str += example_str
            selected_count += 1
        else:
            # 超过长度限制，停止选择
            break
    
    print(f"M02动态选择：从{len(example_scores)}个示例中选择了{selected_count}个最相关的示例")
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
