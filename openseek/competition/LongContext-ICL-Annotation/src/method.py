
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

def build_prompt_by_task_type(task_id: int, task_description: str, text2annotate: str) -> str:
    """
    M03优化版本：任务分型Prompt路由方案
    根据任务类型选择最合适的prompt策略
    """
    
    # 任务类型分类
    math_tasks = [1, 3]  # Task 1: Closest Integers, Task 3: Collatz Conjecture
    string_tasks = [2, 4]  # Task 2: Count Nouns & Verbs, Task 4: Concat Strings
    classification_tasks = [5, 6]  # Task 5: Tweet Sadness, Task 6: MNLI
    generation_tasks = [7, 8]  # Task 7: Jeopardy Answers, Task 8: Kernel Generation
    
    if task_id in math_tasks:
        return build_prompt_math(task_description, text2annotate)
    elif task_id in string_tasks:
        return build_prompt_string(task_description, text2annotate)
    elif task_id in classification_tasks:
        return build_prompt_classification(task_description, text2annotate)
    elif task_id in generation_tasks:
        return build_prompt_generation(task_description, text2annotate)
    else:
        return build_prompt(task_description, text2annotate)

def build_prompt_math(task_description: str, text2annotate: str) -> str:
    """
    M03优化：数学推理任务的专用prompt
    针对Task 1 (Closest Integers)和Task 3 (Collatz Conjecture)
    """
    prompt = (
        "### Role Definition\n"
        "You are a mathematical reasoning expert specializing in numerical analysis and mathematical problem-solving. "
        "You excel at systematic step-by-step reasoning and precise calculations.\n\n"
        
        "### Core Task\n"
        f"{task_description}\n\n"
        
        "### Critical Mathematical Reasoning Guidelines\n"
        "1. **Step-by-Step Analysis**: For mathematical problems, show your reasoning:\n"
        "   - Break down the problem into clear steps\n"
        "   - Verify each calculation carefully\n"
        "   - Explain the logic behind each step\n\n"
        
        "2. **Precision Requirements**:\n"
        "   - Ensure all calculations are accurate\n"
        "   - Double-check numerical operations\n"
        "   - Pay attention to edge cases\n\n"
        
        "3. **Output Format**: Follow this structure:\n"
        "   **Analysis:** [Your step-by-step reasoning]\n"
        "   **Answer:** <label>[numerical result]</label>\n\n"
        
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        
        "### Mathematical Problem to Solve\n"
        f"{text2annotate}\n\n"
        
        "### Final Answer\n"
        "Provide your analysis and final numerical answer in <label> tags."
    )
    return prompt

def build_prompt_string(task_description: str, text2annotate: str) -> str:
    """
    M03优化：字符串处理任务的专用prompt
    针对Task 2 (Count Nouns & Verbs)和Task 4 (Concat Strings)
    """
    prompt = (
        "### Role Definition\n"
        "You are a text analysis expert specializing in linguistic analysis and string manipulation. "
        "You excel at understanding text structure, linguistic patterns, and precise string operations.\n\n"
        
        "### Core Task\n"
        f"{task_description}\n\n"
        
        "### Critical Text Analysis Guidelines\n"
        "1. **Linguistic Analysis**: For text analysis tasks:\n"
        "   - Analyze the text structure carefully\n"
        "   - Identify linguistic patterns correctly\n"
        "   - Apply the specified rules precisely\n\n"
        
        "2. **Precision Requirements**:\n"
        "   - Follow the exact specification for counting/manipulation\n"
        "   - Ensure no elements are missed or counted twice\n"
        "   - Verify your results against the rules\n\n"
        
        "3. **Output Format**: Follow this structure:\n"
        "   **Analysis:** [Your text analysis reasoning]\n"
        "   **Answer:** <label>[text analysis result]</label>\n\n"
        
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        
        "### Text to Analyze\n"
        f"{text2annotate}\n\n"
        
        "### Final Answer\n"
        "Provide your analysis and result in <label> tags."
    )
    return prompt

def build_prompt_classification(task_description: str, text2annotate: str) -> str:
    """
    M03优化：分类任务的专用prompt
    针对Task 5 (Tweet Sadness Detection)和Task 6 (MNLI Classification)
    """
    prompt = (
        "### Role Definition\n"
        "You are a classification expert specializing in sentiment analysis and natural language inference. "
        "You excel at understanding subtle linguistic cues and making accurate classification decisions.\n\n"
        
        "### Core Task\n"
        f"{task_description}\n\n"
        
        "### Critical Classification Guidelines\n"
        "1. **Evidence-Based Classification**:\n"
        "   - Analyze the text for relevant evidence\n"
        "   - Consider multiple factors before deciding\n"
        "   - Justify your classification with specific reasons\n\n"
        
        "2. **Precision Requirements**:\n"
        "   - Match the exact label format from examples\n"
        "   - Be decisive in your classification\n"
        "   - Consider the full context, not just keywords\n\n"
        
        "3. **Output Format**: Follow this structure:\n"
        "   **Reasoning:** [Your classification reasoning]\n"
        "   **Label:** <label>[exact classification label]</label>\n\n"
        
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        
        "### Text to Classify\n"
        f"{text2annotate}\n\n"
        
        "### Final Classification\n"
        "Provide your reasoning and exact label in <label> tags."
    )
    return prompt

def build_prompt_generation(task_description: str, text2annotate: str) -> str:
    """
    M03优化：生成任务的专用prompt
    针对Task 7 (Jeopardy Answer Generation)和Task 8 (Kernel Generation)
    """
    prompt = (
        "### Role Definition\n"
        "You are a generation expert specializing in answer generation and code synthesis. "
        "You excel at producing accurate, contextually appropriate responses and functional code.\n\n"
        
        "### Core Task\n"
        f"{task_description}\n\n"
        
        "### Critical Generation Guidelines\n"
        "1. **Context Understanding**: For generation tasks:\n"
        "   - Thoroughly understand the input context\n"
        "   - Generate content that is relevant and accurate\n"
        "   - Follow the expected format and style\n\n"
        
        "2. **Quality Requirements**:\n"
        "   - Ensure generated content is grammatically correct\n"
        "   - For code: ensure it's syntactically valid and functional\n"
        "   - For answers: ensure they're informative and precise\n\n"
        
        "3. **Output Format**: Follow this structure:\n"
        "   **Context Analysis:** [Your understanding of the input]\n"
        "   **Generated Content:** <label>[your generated content]</label>\n\n"
        
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        
        "### Input for Generation\n"
        f"{text2annotate}\n\n"
        
        "### Final Generation\n"
        "Provide your analysis and generated content in <label> tags."
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

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str, task_id: int = None, sample_index: int = 0) -> str:
    """
        Select examples from all_examples to fit into the target context length (适配Qwen3-4B的token计算).
        all_examples:
            A list of examples, where each example is a dict with keys 'input' and 'output' (no 'length' needed).
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review"}``,
        task_description:
            The description of the annotation task which may be used for example evaluation. 
        text2annotate:
            The text that needs to be annotated  which may be used for example retrieval.
        task_id:
            Task ID to determine the minimum context length requirement.
        sample_index:
            Index of the current sample (0-based) to implement mixed context length strategy.
    """
    # 初始化Qwen3-4B的tokenizer（自动下载/加载千问3-4B的分词器）
    # 若本地已下载模型，可替换为本地路径，如 "./qwen3-4b"
    tokenizer = AutoTokenizer.from_pretrained("/root/Qwen3-4B", trust_remote_code=True)
    
    # 混合上下文长度策略：满足官方30k/16k要求，同时提高效率
    # Task 8: 16k上下文 (官方特殊要求)
    # Task 1-7: 前50个样本用30k上下文 (满足"至少一次30k"要求)，其余用8k上下文 (提高效率)
    if task_id == 8:
        target_length = 16000  # Task 8官方要求16k
    elif task_id is not None and sample_index < 50:
        target_length = 30000  # 前50个样本用30k上下文，满足官方要求
    else:
        target_length = 8192   # 其余样本用8k上下文，提高效率
    
    # print(all_examples[0])  # 打印第一个示例，便于调试

    examples_str, token_num = "", 0
    # 遍历所有示例，基于Qwen3-4B的tokenizer计算token数
    for i, example in enumerate(all_examples):
        try:
            # 提取input和output（兼容output是列表的情况）
            input_text = example['input']
            output_text = example['output'][0]
            
            # 核心：用Qwen3-4B的tokenizer计算input+output的token数（替代原length键）
            # encode返回token id列表，len即为token数
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens  # 等效原示例的length值
            
            # 校验当前示例是否能加入（总长度不超限制）
            if length + token_num <= target_length:
                # 累加总token数：示例文本长度 + 格式符号的token数（<label>2 + </label>3 + \n1 + #1）
                # 注：格式符号的token数是原代码约定，Qwen3-4B对这些符号的实际编码可能略有差异，若需精准可改为：
                # symbol_tokens = len(tokenizer.encode(f"# <label> </label>\n", add_special_tokens=False))
                # token_num += (length + symbol_tokens)
                token_num += (length + 2 + 3 + 1 + 1)
                # 拼接单个示例字符串
                example_str = f"# {input_text} <label> {output_text} </label>\n"
                examples_str += example_str
            else:
                # 超过长度限制，返回已拼接的示例和已选数量
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue
    # 遍历完所有示例且未超长度，返回完整拼接结果
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
    
    # M12优化: Task 8去思维链清洗
    if task_id == 8:
        # Step 1: 过滤<think&rt;标签内容（去思维链）
        cleaned_result = re.sub(r'<think&rt;.*?</think&rt;', '', whole_result, flags=re.DOTALL)
        
        # Step 2: 优先提取代码块（```python 或 ```triton）
        code_blocks = re.findall(r'```(?:python|triton)?\s*(.*?)\s*```', cleaned_result, re.DOTALL)
        if code_blocks:
            # 如果找到代码块，合并所有代码块
            return '\n\n'.join(code_blocks).strip()
        
        # Step 3: 没有代码块时，仅保留import/def/@triton.jit之后的代码正文
        lines = cleaned_result.split('\n')
        code_lines = []
        in_code_section = False
        
        for line in lines:
            # 检测代码开始标记
            if any([
                line.strip().startswith('import '),
                line.strip().startswith('from '),
                line.strip().startswith('def '),
                line.strip().startswith('@triton'),
                line.strip().startswith('class '),
            ]):
                in_code_section = True
            
            # 如果在代码区域，保留该行
            if in_code_section:
                code_lines.append(line)
        
        # 返回清洗后的代码
        return '\n'.join(code_lines).strip() if code_lines else whole_result.strip()
    
    # For other tasks, extract label-tagged content
    prediction = count_answer(whole_result)
    return prediction

# M05方案：相似度+多样性混合检索
def select_examples_M05(all_examples: list[dict], task_description: str, text2annotate: str, task_id: int = None, sample_index: int = 0) -> str:
    """
    M05优化：相似度+多样性混合检索选择示例
    """
    import re
    from transformers import AutoTokenizer
    
    def calculate_similarity(text1: str, text2: str) -> float:
        """计算文本相似度（基于词重叠）"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/Qwen3-4B", trust_remote_code=True)
    
    # 混合上下文长度策略
    if task_id == 8:
        target_length = 16000
    elif task_id is not None and sample_index < 50:
        target_length = 30000
    else:
        target_length = 8192
    
    # M05核心：计算所有示例与待标注文本的相似度
    example_scores = []
    for i, example in enumerate(all_examples):
        try:
            input_text = example['input']
            output_text = example['output'][0]
            
            # 计算相似度（主要基于输入文本）
            similarity = calculate_similarity(input_text, text2annotate)
            
            # 计算token长度
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens
            
            example_scores.append({
                'index': i,
                'similarity': similarity,
                'length': length,
                'input': input_text,
                'output': output_text
            })
        except (KeyError, IndexError) as e:
            print(f"警告：示例{i}处理失败: {e}")
            continue
    
    # M05核心：按相似度降序排序
    example_scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    # M05核心：多样性过滤
    selected_examples = []
    selected_patterns = set()
    current_length = 0
    
    for example in example_scores:
        # 检查是否能加入（长度限制）
        if current_length + example['length'] > target_length:
            break
        
        # 多样性过滤：检查是否与已选示例高度相似
        pattern = example['input'][:50].lower()
        
        if pattern not in selected_patterns:
            selected_examples.append(example)
            selected_patterns.add(pattern)
            current_length += example['length']
        
        # 如果已经选择了足够多的示例，可以提前停止
        if len(selected_examples) >= 10:  # 最多选择10个示例
            break
    
    # 构建示例字符串
    examples_str = ""
    for example in selected_examples:
        examples_str += f"# {example['input']} <label> {example['output']} </label>\n"
    
    print(f"M05相似度+多样性检索：从{len(all_examples)}个示例中选择了{len(selected_examples)}个最相关且多样的示例")
    return examples_str
