
import re
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

""" Here is an example of implementation of Long-Context Data Annotation. """

# 全局变量：延迟加载的 TF-IDF 向量化器
_tfidf_vectorizer = None

def get_tfidf_vectorizer():
    """
    获取或初始化 TF-IDF 向量化器（延迟加载，避免重复加载）
    使用 sklearn 的 TfidfVectorizer，不需要下载额外模型
    """
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # 使用简单的 TF-IDF 配置，支持字符级和词级特征
        _tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # 使用字符 n-gram（对数字、符号等效果更好）
            ngram_range=(1, 3),  # 1-3 字符 n-gram
            max_features=5000,   # 限制特征数量
            sublinear_tf=True    # 使用对数缩放
        )
    return _tfidf_vectorizer

def compute_tfidf_similarity(texts: list[str]) -> np.ndarray:
    """
    计算 texts 列表中所有文本之间的 TF-IDF 余弦相似度矩阵
    :param texts: 文本列表，第一个是测试样本，后面是示例
    :return: 相似度矩阵的第一行（测试样本与所有示例的相似度）
    """
    vectorizer = get_tfidf_vectorizer()
    try:
        # 如果 vectorizer 还未 fit，先 fit 所有文本
        tfidf_matrix = vectorizer.fit_transform(texts)
    except:
        # 如果已经 fit 过，使用 transform
        tfidf_matrix = vectorizer.transform(texts)
    
    # 计算第一个文本（测试样本）与所有其他文本的余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    return similarities

def select_examples_by_similarity(all_examples: list[dict], text2annotate: str, top_k: int = None) -> list[dict]:
    """
    Combination-01: 语义相似度示例选择（使用 TF-IDF 实现）
    核心策略：
    1. 使用 TF-IDF 向量化计算文本特征
    2. 基于余弦相似度选择最相关的 top-k 示例
    3. 按照示例难度（输入长度）从简单到复杂排序
    
    :param all_examples: 所有可用示例列表
    :param text2annotate: 待标注的文本
    :param top_k: 选择的示例数量上限（默认为全部）
    :return: 按相似度排序且按难度重新排列的示例列表
    """
    if not all_examples:
        return []
    
    if top_k is None:
        top_k = len(all_examples)
    
    # 准备所有文本：测试样本 + 所有示例输入
    all_texts = [text2annotate] + [ex['input'] for ex in all_examples]
    
    # 计算 TF-IDF 相似度
    similarities = compute_tfidf_similarity(all_texts)
    
    # 计算每个示例的相似度和输入长度
    example_scores = []
    for i, example in enumerate(all_examples):
        input_text = example['input']
        similarity = similarities[i]  # 相似度已按顺序对应
        input_length = len(input_text)
        example_scores.append({
            'example': example,
            'index': i,
            'similarity': float(similarity),
            'input_length': input_length
        })
    
    # 按相似度降序排序，选择 top-k 个示例
    example_scores.sort(key=lambda x: x['similarity'], reverse=True)
    selected = example_scores[:top_k]
    
    # 按输入长度（难度）从简单到复杂排序
    selected.sort(key=lambda x: x['input_length'])
    
    return [item['example'] for item in selected]

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
    Construct a concise prompt for Qwen3-4B annotation.
    明确要求模型直接输出答案，不要思考过程，并用 <label> 标签包裹结果。
    使用 /no_think 前缀禁用 Qwen3 的思考模式。
    """
    prompt = (
        "/no_think\n"
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, and ensure the final annotation result is enclosed in <label> tags.\n\n"
        "### Core Task\n"
        f"Task: {task_description}\n\n"
        "### Critical Annotation Guidelines\n"
        "1. **Example Learning Requirement**: Thoroughly analyze and fully learn from the annotation logic, format, and criteria in the Examples section. "
        "Your annotation must align with the style, judgment standards, and tag usage shown in the examples.\n"
        "2. **Silent Reasoning Requirement**: Think through the task internally, but do NOT output your reasoning, analysis, explanation, or any extra words.\n"
        "3. **Mandatory Output Rule**: Output only the final annotation result enclosed in <label> tags.\n"
        "   - Correct example 1: <label>3</label>\n"
        "   - Correct example 2: <label>Good Review</label>\n"
        "   - Wrong example 1: Reasoning: ... <label>3</label>\n"
        "   - Wrong example 2: 3\n"
        "   - Wrong example 3: <label>3\n"
        "4. **Brevity Requirement**: The final answer must be a single short label only. Do not repeat the input text. Do not add prefixes such as 'Answer:' or 'Reasoning:'.\n\n"
        
        "Examples:\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"Input: {text2annotate}\n"
        "Output:"
        "### Final Requirement Summary\n"
        "1. Think silently and do not output analysis.\n"
        "2. Output exactly one final answer wrapped in <label> and </label>.\n"
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

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str, is_code_generation: bool = False, use_similarity: bool = True) -> str:
    """
        Combination-01: 语义相似度示例选择
        核心策略：
        1. 使用 sentence-transformers 计算文本 embedding
        2. 基于余弦相似度选择最相关的 top-k 示例
        3. 按照示例难度（输入长度）从简单到复杂排序
        4. 在上下文长度限制内填充示例
        
        :param all_examples: 所有可用示例列表
        :param task_description: 任务描述
        :param text2annotate: 待标注的文本
        :param is_code_generation: 是否为代码生成任务
        :param use_similarity: 是否使用语义相似度选择（默认True，启用Combination-01方案）
    """
    # 初始化Qwen3-4B的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/flagos/Qwen3-4B", trust_remote_code=True)
    
    # 最大上下文长度限制
    target_length = 8192
    
    # 计算固定的 prompt 部分的 token 数（不包括 ICL 示例）
    fixed_prompt = (
        f"Task: {task_description}\n\n"
        "Examples:\n"
        "[[EXAMPLES]]\n\n"
        f"Input: {text2annotate}\n"
        "Output: "
    )
    fixed_tokens = len(tokenizer.encode(fixed_prompt, add_special_tokens=False))
    
    # ICL 示例可用的最大 token 数
    available_tokens_for_examples = target_length - fixed_tokens
    
    if available_tokens_for_examples <= 0:
        print(f"警告：任务描述和测试样本输入已超过上下文长度限制（{fixed_tokens} > {target_length} tokens）")
        return ""
    
    # Combination-01 核心逻辑：语义相似度选择 + 难度排序
    if use_similarity and len(all_examples) > 0:
        # 使用语义相似度选择最相关的 top-k 示例，并按难度排序
        selected_examples = select_examples_by_similarity(all_examples, text2annotate, top_k=len(all_examples))
    else:
        # 回退到原始顺序
        selected_examples = all_examples
    
    examples_str, token_num = "", 0
    # 遍历选中的示例，填充到上下文长度限制内
    for i, example in enumerate(selected_examples):
        try:
            input_text = example['input']
            output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
            
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens
            
            if is_code_generation:
                format_tokens = len(tokenizer.encode(f"Input: \nOutput: \n\n", add_special_tokens=False))
                example_str = f"Input: {input_text}\nOutput: {output_text}\n\n"
            else:
                format_tokens = len(tokenizer.encode(f"# <label> </label>\n", add_special_tokens=False))
                example_str = f"# {input_text} <label> {output_text} </label>\n"
            
            if length + format_tokens + token_num <= available_tokens_for_examples:
                token_num += (length + format_tokens)
                examples_str += example_str
            else:
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue
    
    return examples_str




def count_answer(text: str) -> tuple[list, dict]:
    """
    提取字符串中<label>标签内的所有内容（字符串形式），统计出现次数最多的内容
    :param text: 包含<label>标签的原始字符串
    :return: 出现次数最多的内容列表、所有内容的频次统计字典
    """
    # 处理 Qwen3 思考模式：提取 answer 标签后的内容
    if '<|answer|>' in text:
        parts = text.split('<|answer|>')
        if len(parts) > 1:
            text = parts[-1].split('<|/answer|>')[0] if '<|/answer|>' in parts[-1] else parts[-1]
    
    # 优先匹配完整的 <label>...</label> 标签
    pattern = r'<label>\s*(.+?)\s*</label>'
    content_matches = re.findall(pattern, text, re.DOTALL)
    
    # 如果没有匹配到完整标签，尝试匹配未闭合的 <label>...（模型可能输出 <label>答案 后就停止了）
    if not content_matches:
        pattern_unclosed = r'<label>\s*(.+)'
        unclosed_matches = re.findall(pattern_unclosed, text, re.DOTALL)
        if unclosed_matches:
            # 取第一个匹配，去掉可能的后续换行或多余内容
            content = unclosed_matches[0].strip().split('\n')[0].strip()
            if content and len(content) < 100:
                content_matches = [content]
    
    content_counter = Counter(content_matches)
    if not content_counter:
        return None
    
    max_count = max(content_counter.values())
    answer = [content for content, count in content_counter.items() if count == max_count]
    
    if (len(answer[0]) >= 100):
        return None
    return answer[0]


def clean_code_generation_output(text: str) -> str:
    """
    清理代码生成任务（Task 8）的输出，移除多余的标记和标签
    
    处理逻辑：
    1. 移除 Qwen3 思考模式的结束标记 ``
    2. 移除 `<label>` 和 `</label>` 标签（如果存在）
    3. 保留完整的 Python 代码内容
    4. 移除开头和结尾的多余空白字符
    
    :param text: 模型原始输出
    :return: 清理后的代码输出
    """
    # 移除 Qwen3 思考模式的结束标记
    # 这个标记通常出现在输出的开头，后面跟着换行符
    # 使用 Unicode 编码 \u25b6 来表示黑色右向三角形字符
    try:
        think_end_marker = chr(0x25b6)  # Unicode: ▶
        if think_end_marker in text:
            text = text.split(think_end_marker, 1)[1]
    except:
        pass  # 如果出现问题，跳过此步骤
    
    # 处理 <label> 标签
    # 策略：如果整个输出被 <label>...</label> 包裹，则移除标签
    # 如果标签中只有简短内容（如函数名），则不要只提取标签内容，而是保留所有非标签内容
    label_pattern = r'<label>\s*(.+?)\s*</label>'
    label_matches = re.findall(label_pattern, text, re.DOTALL)
    
    if label_matches:
        # 检查是否有多个 label 标签
        if len(label_matches) > 1:
            # 多个标签，取最长的一组（通常是完整的代码）
            cleaned_text = max(label_matches, key=len)
        else:
            # 只有一个标签，检查标签内容的长度
            label_content = label_matches[0].strip()
            # 如果标签内容很短（少于100字符），可能是只提取了函数名
            # 这种情况下，我们不应该只返回标签内容，而是返回整个文本（去掉标签）
            if len(label_content) < 100:
                # 标签内容太短，移除所有标签标签，保留所有其他内容
                cleaned_text = re.sub(r'<[^>]+>', '', text)
            else:
                # 标签内容足够长，很可能是完整代码，直接使用
                cleaned_text = label_content
    else:
        # 没有 label 标签，直接使用原始文本
        cleaned_text = text
    
    # 移除开头和结尾的多余空白字符
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


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

def annotate_ascend(input_prompt:str, max_tokens:int=2048, use_count_answer:bool=True, task_id:int=None)->tuple:
    """
        Annotate the unlabeled data using an LLM API (Huawei Ascend).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
        max_tokens:
            Maximum tokens to generate (default: 2048 for code generation tasks).
        use_count_answer:
            Whether to use count_answer to extract label from response (default: True).
            For code generation tasks, set to False to return the full response.
        task_id:
            Task ID (default: None). If task_id == 8 (code generation), will apply special cleaning.
        返回: (prediction, raw_output) 元组，方便调试
    """
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {"role": "user", "content": input_prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=False
    )
    whole_result = response.choices[0].message.content
    
    if use_count_answer:
        # Task 1-7: 使用 count_answer 提取标签内容
        prediction = count_answer(whole_result)
    else:
        # Task 8: 代码生成任务，需要清理输出
        if task_id == 8:
            # 清理代码生成任务的输出，移除多余的标记和标签
            prediction = clean_code_generation_output(whole_result)
        else:
            # 其他任务直接返回原始输出
            prediction = whole_result
    
    return prediction, whole_result


def annotate_batch(prompts: list[str], num_workers: int = 4, max_tokens: int = 128, use_count_answer: bool = True, task_id: int = None) -> list[str]:
    """
        Batch annotate with parallel requests.
        prompts: List of prompts to annotate.
        num_workers: Number of parallel workers.
        max_tokens: Maximum tokens to generate for each prompt.
        use_count_answer: Whether to use count_answer to extract label from response.
        task_id: Task ID (default: None). If task_id == 8, will apply special cleaning.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = [None] * len(prompts)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(annotate_ascend, p, max_tokens, use_count_answer, task_id): i for i, p in enumerate(prompts)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing prompt {idx}: {e}")
                results[idx] = (None, None)  # 返回 (None, None) 而不是 None，以便解包
    
    return results
