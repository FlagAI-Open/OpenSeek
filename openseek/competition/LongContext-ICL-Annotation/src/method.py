
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

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
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
    # 初始化Qwen3-4B的tokenizer（自动下载/加载千问3-4B的分词器）
    # 若本地已下载模型，可替换为本地路径，如 "./qwen3-4b"
    tokenizer = AutoTokenizer.from_pretrained("/share/project/wuhaiming/spaces/data_agent/OpenSeek-main/openseek/competition/LongContext-ICL-Annotation/src/Qwen3-4B", trust_remote_code=True)
    
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
    
    if (len(answer[0]) >= 100):
        return None
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
        "max_tokens": 10_000, # max_token = 10k
    }

    try:
        resp = requests.post(URL, json=data)
        whole_result = resp.json()["choices"][0]["text"]
    except Exception as e:
        whole_result = "None"


    prediction = count_answer(whole_result)
    return prediction

def annotate_ascend(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (Huawei Ascend).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"

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
