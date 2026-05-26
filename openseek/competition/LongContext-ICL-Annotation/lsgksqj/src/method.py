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
        "0. **Absolute Authority of Examples**: The provided Examples are the SOLE and ABSOLUTE source of truth for annotation logic. Even if an example contradicts your pre-trained linguistic knowledge or standard grammar (e.g., splitting compound nouns), you MUST mirror the example's logic perfectly. NEVER question or analyze why an example is correct—simply execute the same pattern."
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
    """
    # 使用 f-string 配合 三引号，保证格式整齐且逻辑清晰
    prompt = f"""### Role: Expert Context Annotator
        ### Task Description
        {task_description}
        
        ### General Workflow
        1. <think>: Identify task type (1-8), extract info, and solve step-by-step.
        2. <label>: Output FINAL result ONLY. No spaces/extra text outside tags.
        3. Terminate with <END>.
        
        ### Task-Specific Logic (Apply ONLY the relevant one)
        - [Int/String Ops]: (Task 1, 3, 4) For concatenation, treat as RAW PASSWORD. No auto-spaces. Count chars to verify: Sum(input_chars) == count(output_chars).
        - [NLP Classification]: (Task 2, 5, 6) 
          - Sadness: Check tone/sarcasm. Anger/Neutral != Sad.
          - Genre: Dual-Independent Verification. Both sentences must STRONGLY match. If one is vague, output <label>N</label>.
        - [Trivia]: (Task 7) lowercase only. Match all clue constraints.
        - [Triton Code]: (Task 8) Python ONLY. Use `a.numel()`, `tl.math.tanh` for GELU.
        
        ### Strict Output Constraints
        - NO <label> or <END> inside <think>.
        - Format: Reasoning... <label>Result</label><END>
        - Character-level precision for all math/string tasks.
        
        ### Input Text to Annotate
        {text2annotate}
        """
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

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str, tokenizer: AutoTokenizer) -> str:
    """
        Select examples from all_examples to fit into the target context length (适配Qwen3-4B的token计算).
        all_examples:
            A list of examples, where each example is a dict with keys 'input' and 'output' (no 'length' needed).
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review"}``,
        task_description:
            The description of the annotation task which may be used for example evaluation. 
        text2annotate:
            The text that needs to be annotated  which may be used for example retrieval.
    """
    
    # 最大上下文长度限制（Qwen3-4B的上下文窗口默认是8k/32k，可根据实际调整）
    target_length = 8192  # 若需严格适配Qwen3-4B，建议改为8192（8k）
    
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




def count_answer(text: str):
    if not text or text == "None":
        return None
    
    content = text.strip()
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    pattern = r'[<\[]label[>\]](.*?)[<\[]\/label[>\]]'
    matches = re.findall(pattern, content, flags=re.DOTALL | re.IGNORECASE)

    if matches:
        final_ans = matches[-1].strip()
        if len(final_ans) > 0:
            return final_ans
    start_tags = ["<label>", "[label>", "<label]", "[label]"]
    for tag in start_tags:
        if tag.lower() in content.lower():
            last_tag_idx = content.lower().rfind(tag.lower())
            incomplete = content[last_tag_idx + len(tag):].strip()
            incomplete = re.split(r'<END>|###', incomplete, flags=re.IGNORECASE)[0].strip()
            
            if len(incomplete) > 0:
                return incomplete

    return None


def annotate_nvidia(input_prompt:str)->tuple:
    import requests
    URL="http://127.0.0.1:2026/v1/chat/completions"
    
    data = {
        "model": "qwen3-4b", 
        "messages": [
            {"role": "system", "content": "You are a precise data annotation assistant. Use <think> tags for reasoning and always end your response with <label>RESULT</label>.Your answer MUST start with the character sequence <label> and end with </label>. Do not omit the l, a, b, e, l or the brackets."},
            {"role": "user", "content": input_prompt}
        ],
        "max_tokens": 10000, 
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "presence_penalty": 0.0
    }

    try:
        resp = requests.post(URL, json=data)
        res_json = resp.json()
        # 从 chat 接口正确提取内容
        whole_result = res_json["choices"][0]["message"]["content"]
        print(f"--- 模型原始回复 ---\n{whole_result}\n------------------")
    except Exception as e:
        print(f"标注出错: {e}")
        whole_result = "None"

    prediction = count_answer(whole_result)
    return prediction, whole_result

def annotate_ascend(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (Huawei Ascend).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:2026/v1"
    model = "/root/autodl-tmp/Qwen-Final"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=10_000,
        stream=False,
    )
    whole_result = response.choices[0].message.content
    prediction = count_answer(whole_result)
    return prediction
