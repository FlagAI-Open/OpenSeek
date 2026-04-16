import re
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

""" Here is an example of implementation of Long-Context Data Annotation. """


_tfidf_vectorizer = None

def get_tfidf_vectorizer():




    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        _tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True
        )
    return _tfidf_vectorizer

def compute_tfidf_similarity(texts: list[str]) -> np.ndarray:





    vectorizer = get_tfidf_vectorizer()
    try:

        tfidf_matrix = vectorizer.fit_transform(texts)
    except:

        tfidf_matrix = vectorizer.transform(texts)
    

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    return similarities

def select_examples_by_similarity(all_examples: list[dict], text2annotate: str, top_k: int = None) -> list[dict]:












    if not all_examples:
        return []
    
    if top_k is None:
        top_k = len(all_examples)
    

    all_texts = [text2annotate] + [ex['input'] for ex in all_examples]
    

    similarities = compute_tfidf_similarity(all_texts)
    

    example_scores = []
    for i, example in enumerate(all_examples):
        input_text = example['input']
        similarity = similarities[i]
        input_length = len(input_text)
        example_scores.append({
            'example': example,
            'index': i,
            'similarity': float(similarity),
            'input_length': input_length
        })
    

    example_scores.sort(key=lambda x: x['similarity'], reverse=True)
    selected = example_scores[:top_k]
    

    selected.sort(key=lambda x: x['input_length'])
    
    return [item['example'] for item in selected]


def select_examples_by_diversity(all_examples: list[dict], text2annotate: str, n_clusters: int = None, top_k: int = None) -> list[dict]:














    if not all_examples:
        return []
    
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_examples = len(all_examples)
    if top_k is None:
        top_k = n_examples
    

    if n_clusters is None:
        n_clusters = min(int(np.sqrt(n_examples)), 10)
    n_clusters = min(n_clusters, n_examples)
    

    vectorizer = get_tfidf_vectorizer()
    all_texts = [ex['input'] for ex in all_examples]
    

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    cluster_centers = kmeans.cluster_centers_
    

    selected_examples = []
    for cluster_id in range(n_clusters):

        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        

        cluster_vectors = tfidf_matrix[cluster_indices].toarray()
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        

        closest_idx_in_cluster = np.argmin(distances)
        original_idx = cluster_indices[closest_idx_in_cluster]
        selected_examples.append(all_examples[original_idx])
    

    selected_examples.sort(key=lambda x: len(x['input']))
    

    return selected_examples[:top_k]


def compute_example_quality_score(example: dict) -> float:












    input_text = example.get('input', '')
    output_text = example.get('output', '')
    if isinstance(output_text, list):
        output_text = output_text[0] if output_text else ''
    

    if not output_text or not output_text.strip():
        return 0.0
    
    score = 1.0
    

    input_len = len(input_text)
    if input_len < 5:
        score *= 0.5
    elif input_len > 500:
        score *= 0.7
    elif 10 <= input_len <= 200:
        score *= 1.0
    else:
        score *= 0.9
    

    output_len = len(output_text)
    if output_len > 100:
        score *= 0.6
    elif output_len > 50:
        score *= 0.8
    elif 1 <= output_len <= 50:
        score *= 1.0
    

    special_chars = sum(1 for c in input_text if not c.isalnum() and c not in ' \t\n')
    special_ratio = special_chars / max(input_len, 1)
    if special_ratio > 0.5:
        score *= 0.7
    

    digit_ratio = sum(1 for c in input_text if c.isdigit()) / max(input_len, 1)

    
    return max(0.0, min(1.0, score))


def select_examples_by_quality(all_examples: list[dict], text2annotate: str, quality_threshold: float = 0.5, top_k: int = None) -> list[dict]:














    if not all_examples:
        return []
    
    if top_k is None:
        top_k = len(all_examples)
    

    example_scores = []
    for i, example in enumerate(all_examples):
        quality_score = compute_example_quality_score(example)
        input_length = len(example.get('input', ''))
        example_scores.append({
            'example': example,
            'index': i,
            'quality_score': quality_score,
            'input_length': input_length
        })
    

    filtered = [item for item in example_scores if item['quality_score'] >= quality_threshold]
    

    if len(filtered) < top_k * 0.3:
        filtered = sorted(example_scores, key=lambda x: x['quality_score'], reverse=True)
        filtered = filtered[:int(len(example_scores) * 0.7)]
    

    filtered.sort(key=lambda x: x['quality_score'], reverse=True)
    selected = filtered[:top_k]
    

    selected.sort(key=lambda x: x['input_length'])
    
    return [item['example'] for item in selected]


def get_task_specific_guidance(task_id: int) -> str:














    guidance_map = {

        1: ("符号推理指导：计算列表中任意两个整数之间的最小绝对差值。\n"
            "步骤：\n"
            "1. 仔细阅读输入的整数列表\n"
            "2. 计算每对整数之间的绝对差值\n"
            "3. 找出最小的差值\n"
            "4. 输出单个整数作为结果\n\n"
            "注意：\n"
            "- 输出必须是整数\n"
            "- 注意负数的处理\n"
            "- 0也是有效的差值"),
        2: ("符号推理指导：统计输入文本中名词和动词的数量。\n"
            "步骤：\n"
            "1. 仔细阅读输入文本\n"
            "2. 识别所有名词（人名、地名、物体等）\n"
            "3. 识别所有动词（动作词、状态词等）\n"
            "4. 输出格式：名词数量,动词数量\n\n"
            "注意：\n"
            "- 按照示例的格式输出\n"
            "- 区分词性要准确\n"
            "- 数字用逗号分隔"),
        3: ("符号推理指导：根据Collatz猜想计算序列的步数。\n"
            "步骤：\n"
            "1. 读取起始正整数n\n"
            "2. 如果n是偶数，n = n/2；如果n是奇数，n = 3n + 1\n"
            "3. 重复步骤2，直到n等于1\n"
            "4. 计算总共执行的步数\n\n"
            "注意：\n"
            "- 起始数n必须是正整数\n"
            "- 步数是指从n到1需要的迭代次数\n"
            "- 确保计算准确"),
        4: ("符号推理指导：将两个字符串按顺序拼接。\n"
            "步骤：\n"
            "1. 读取两个输入字符串\n"
            "2. 按顺序将第一个字符串和第二个字符串拼接\n"
            "3. 输出拼接后的结果\n\n"
            "注意：\n"
            "- 保持字符串的原始顺序\n"
            "- 不要添加额外的空格或字符\n"
            "- 注意区分大小写"),
        

        5: ("情感分析指导：判断推文是否表达了悲伤情绪。\n"
            "步骤：\n"
            "1. 仔细阅读推文内容\n"
            "2. 分析语言表达（词语、语气、标点）\n"
            "3. 考虑emoji和特殊符号的含义\n"
            "4. 判断是否表达了悲伤情绪\n\n"
            "注意：\n"
            "- 悲伤情绪可能是直接的，也可能是含蓄的\n"
            "- 注意反讽和幽默\n"
            "- 关注情感相关的关键词\n"
            "- 考虑emoji的情感色彩\n"
            "- 输出必须是Good Review或Bad Review之一"),
        

        6: ("自然语言推理指导：判断前提和假设的关系。\n"
            "关系类型：\n"
            "- Entailment（蕴含）：假设在前提下必然成立\n"
            "- Neutral（中立）：假设与前提无关或不确定\n"
            "- Contradiction（矛盾）：假设与前提矛盾\n\n"
            "步骤：\n"
            "1. 仔细阅读前提和假设\n"
            "2. 分析两者的逻辑关系\n"
            "3. 判断关系类型\n\n"
            "注意：\n"
            "- 输出必须是Entailment、Neutral或Contradiction之一\n"
            "- 仔细分析语言中的细微差别\n"
            "- 考虑上下文信息\n"
            "- 不要添加额外的推理"),
        

        7: ("开放问答指导：根据类别和线索生成答案。\n"
            "步骤：\n"
            "1. 仔细阅读类别和线索\n"
            "2. 理解线索中提供的信息\n"
            "3. 根据线索推断答案\n"
            "4. 生成简洁准确的答案\n\n"
            "注意：\n"
            "- 答案应该是简短的名称或术语\n"
            "- 不要包含额外的解释\n"
            "- 注意线索中的关键词\n"
            "- 保持答案格式一致\n"
            "- 使用正确的拼写和大小写"),
        

        8: ("代码生成指导：根据输入生成Triton内核代码。\n"
            "步骤：\n"
            "1. 仔细阅读输入要求和约束\n"
            "2. 理解需要的张量操作\n"
            "3. 设计符合Triton规范的内核\n"
            "4. 确保代码的正确性和效率\n\n"
            "注意：\n"
            "- 遵循Triton编程规范\n"
            "- 确保张量维度正确\n"
            "- 处理边界情况\n"
            "- 优化内存访问\n"
            "- 使用适当的类型和精度\n"
            "- 保持代码可读性"),
    }
    
    return guidance_map.get(task_id, "")

def build_prompt(task_description: str, text2annotate: str, task_id: int = None) -> str:







    task_guidance = ""
    if task_id is not None:
        task_guidance = get_task_specific_guidance(task_id)
    
    guidance_section = ""
    if task_guidance:
        guidance_section = (
            "### Task-Specific Guidance\n"
            f"{task_guidance}\n\n"
        )
    
    prompt = (
        "/no_think\n"
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, and ensure the final annotation result is enclosed in <label> tags.\n\n"
        "### Core Task\n"
        f"Task: {task_description}\n\n"
        f"{guidance_section}"
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
        "4. Follow the task-specific guidance if provided.\n"
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

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str, is_code_generation: bool = False, use_diversity: bool = True, use_similarity: bool = False, use_task_aware: bool = False, task_id: int = None, use_quality_filter: bool = False, quality_threshold: float = 0.5, use_cot: bool = False) -> str:


























    tokenizer = AutoTokenizer.from_pretrained("/root/flagos/Qwen3-4B", trust_remote_code=True)
    


    if use_task_aware and task_id is not None:
        task_config = get_task_aware_config(task_id)
        target_length = task_config['target_length']
        max_examples = task_config['max_examples']
    else:

        target_length = 8192
        max_examples = None
    

    fixed_prompt = (
        f"Task: {task_description}\n\n"
        "Examples:\n"
        "[[EXAMPLES]]\n\n"
        f"Input: {text2annotate}\n"
        "Output: "
    )
    fixed_tokens = len(tokenizer.encode(fixed_prompt, add_special_tokens=False))
    

    available_tokens_for_examples = target_length - fixed_tokens
    
    if available_tokens_for_examples <= 0:
        print(f"警告：任务描述和测试样本输入已超过上下文长度限制（{fixed_tokens} > {target_length} tokens）")
        return ""
    

    if len(all_examples) > 0:
        if use_quality_filter:

            selected_examples = select_examples_by_quality(all_examples, text2annotate, quality_threshold=quality_threshold)
        elif use_diversity:

            selected_examples = select_examples_by_diversity(all_examples, text2annotate)
        elif use_similarity:

            selected_examples = select_examples_by_similarity(all_examples, text2annotate)
        else:

            selected_examples = all_examples
    else:
        selected_examples = []
    

    if max_examples is not None and len(selected_examples) > max_examples:
        selected_examples = selected_examples[:max_examples]
    
    examples_str, token_num = "", 0

    for i, example in enumerate(selected_examples):
        try:
            input_text = example['input']
            output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
            
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens
            

            if use_cot and task_id is not None and 1 <= task_id <= 4:
                example_str = format_example_with_cot(example, task_id, is_code_generation)
            else:

                if is_code_generation:
                    format_tokens = len(tokenizer.encode(f"Input: \nOutput: \n\n", add_special_tokens=False))
                    example_str = f"Input: {input_text}\nOutput: {output_text}\n\n"
                else:
                    format_tokens = len(tokenizer.encode(f"# <label> </label>\n", add_special_tokens=False))
                    example_str = f"# {input_text} <label> {output_text} </label>\n"
            

            format_tokens = len(tokenizer.encode(example_str, add_special_tokens=False))
            
            if length + format_tokens + token_num <= available_tokens_for_examples:
                token_num += (length + format_tokens)
                examples_str += example_str
            else:
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue
    
    return examples_str


def get_task_aware_config(task_id: int) -> dict:

















    configs = {

        1: {'target_length': 7000, 'max_examples': 120, 'description': '符号推理-最接近整数'},
        2: {'target_length': 7000, 'max_examples': 120, 'description': '符号推理-名词动词计数'},
        3: {'target_length': 7000, 'max_examples': 120, 'description': '符号推理-Collatz猜想'},
        4: {'target_length': 7000, 'max_examples': 120, 'description': '符号推理-字符串拼接'},

        5: {'target_length': 6000, 'max_examples': 75, 'description': '情感分析'},

        6: {'target_length': 6000, 'max_examples': 75, 'description': 'NLI分类'},

        7: {'target_length': 6000, 'max_examples': 75, 'description': '开放问答'},

        8: {'target_length': 5000, 'max_examples': 45, 'description': '代码生成'},
    }
    
    return configs.get(task_id, {'target_length': 6000, 'max_examples': 30, 'description': '默认'})




def count_answer(text: str) -> tuple[list, dict]:






    if '<|answer|>' in text:
        parts = text.split('<|answer|>')
        if len(parts) > 1:
            text = parts[-1].split('<|/answer|>')[0] if '<|/answer|>' in parts[-1] else parts[-1]
    

    pattern = r'<label>\s*(.+?)\s*</label>'
    content_matches = re.findall(pattern, text, re.DOTALL)
    

    if not content_matches:
        pattern_unclosed = r'<label>\s*(.+)'
        unclosed_matches = re.findall(pattern_unclosed, text, re.DOTALL)
        if unclosed_matches:

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















    try:
        think_end_marker = chr(0x25b6)  # Unicode: ▶
        if think_end_marker in text:
            text = text.split(think_end_marker, 1)[1]
    except:
        pass
    



    label_pattern = r'<label>\s*(.+?)\s*</label>'
    label_matches = re.findall(label_pattern, text, re.DOTALL)
    
    if label_matches:

        if len(label_matches) > 1:

            cleaned_text = max(label_matches, key=len)
        else:

            label_content = label_matches[0].strip()


            if len(label_content) < 100:

                cleaned_text = re.sub(r'<[^>]+>', '', text)
            else:

                cleaned_text = label_content
    else:

        cleaned_text = text
    

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

        prediction = count_answer(whole_result)
    else:

        if task_id == 8:

            prediction = clean_code_generation_output(whole_result)
        else:

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
                results[idx] = (None, None)
    
    return results


def generate_cot_reasoning(task_id: int, example: dict) -> str:








    input_text = example.get('input', '')
    output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
    
    reasoning_map = {
        1: f"**步骤1：提取信息**\n输入列表: {input_text}\n\n**步骤2：应用规则**\n计算所有可能的整数对绝对差值\n\n**步骤3：验证结果**\n最小绝对差值 = {output_text}\n\n**结论**: 答案是 {output_text}",
        2: f"**步骤1：提取信息**\n输入文本: {input_text}\n\n**步骤2：应用规则**\n识别并统计所有名词和动词\n\n**步骤3：验证结果**\n名词数量 = {output_text.split(',')[0] if ',' in output_text else '?'}, 动词数量 = {output_text.split(',')[1] if ',' in output_text else '?'}\n\n**结论**: 答案是 {output_text}",
        3: f"**步骤1：提取信息**\n起始数: {input_text}\n\n**步骤2：应用规则**\n应用Collatz猜想规则：\n- 如果n是偶数，n = n/2\n- 如果n是奇数，n = 3n + 1\n\n**步骤3：验证结果**\n迭代直到n=1，共需要 {output_text} 步\n\n**结论**: 答案是 {output_text}",
        4: f"**步骤1：提取信息**\n需要拼接的字符串: {input_text}\n\n**步骤2：应用规则**\n按顺序将两个字符串拼接\n\n**步骤3：验证结果**\n拼接结果 = {output_text}\n\n**结论**: 答案是 {output_text}"
    }
    
    return reasoning_map.get(task_id, "")


def format_example_with_cot(example: dict, task_id: int, is_code_generation: bool = False) -> str:








    input_text = example['input']
    output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
    

    if 1 <= task_id <= 4:
        reasoning = generate_cot_reasoning(task_id, example)
        if is_code_generation:
            return f"Input: {input_text}\nReasoning:\n{reasoning}\nOutput: {output_text}\n\n"
        else:
            return f"# {input_text}\nReasoning:\n{reasoning}\nResult: <label> {output_text} </label>\n"
    else:

        if is_code_generation:
            return f"Input: {input_text}\nOutput: {output_text}\n\n"
        else:
            return f"# {input_text} <label> {output_text} </label>\n"


def vote_by_frequency(predictions: list, confidence_threshold: float = 0.4) -> tuple:












    if not predictions:
        return None, 0.0
    

    valid_predictions = [p for p in predictions if p is not None]
    if not valid_predictions:
        return None, 0.0
    

    counter = Counter(valid_predictions)
    

    most_common = counter.most_common(1)
    if not most_common:
        return None, 0.0
    
    answer, count = most_common[0]
    confidence = count / len(predictions)
    

    if confidence < confidence_threshold:
        return None, confidence
    
    return answer, confidence


def annotate_with_self_consistency(input_prompt: str, num_samples: int = 5, temperature_range: list = None, max_tokens: int = 256, task_id: int = None, confidence_threshold: float = 0.4) -> tuple:

















    import openai
    

    if temperature_range is None:
        temperature_range = [0.7, 0.85, 1.0, 1.1]
    

    if len(temperature_range) < num_samples:
        temperature_range = temperature_range + [temperature_range[-1]] * (num_samples - len(temperature_range))
    
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"
    
    predictions = []
    

    for i in range(num_samples):
        temperature = temperature_range[i % len(temperature_range)]
        
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
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            stream=False
        )
        
        whole_result = response.choices[0].message.content
        prediction = count_answer(whole_result)
        predictions.append(prediction)
    

    final_answer, confidence = vote_by_frequency(predictions, confidence_threshold)
    
    return final_answer, confidence, predictions