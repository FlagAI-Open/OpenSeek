import re
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

""" Here is an example of implementation of Long-Context Data Annotation. """

                        
_tfidf_vectorizer = None

def get_tfidf_vectorizer():
\
\
\
       
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
\
\
\
\
       
    vectorizer = get_tfidf_vectorizer()
    try:
                                         
        tfidf_matrix = vectorizer.fit_transform(texts)
    except:
                                 
        tfidf_matrix = vectorizer.transform(texts)
    
                                
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    return similarities

def select_examples_by_similarity(all_examples: list[dict], text2annotate: str, top_k: int = None) -> list[dict]:
\
\
\
\
\
\
\
\
\
\
\
       
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
\
\
\
\
\
\
\
\
\
\
\
\
\
       
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
\
\
\
\
\
\
\
\
\
\
\
       
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
\
\
\
\
\
\
\
\
\
\
\
\
\
       
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
\
\
\
\
\
\
\
\
\
\
\
\
\
       
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

def build_prompt(task_description: str, text2annotate: str, task_id: int = None, use_social_media_enhancement: bool = False) -> str:
\
\
\
\
\
\
\
\
\
\
\
\
       
               
    task_guidance = ""
    if task_id is not None:
        task_guidance = get_task_specific_guidance(task_id)
    
    guidance_section = ""
    if task_guidance:
        guidance_section = (
            "### Task-Specific Guidance\n"
            f"{task_guidance}\n\n"
        )
    
                                          
    social_media_section = ""
    if use_social_media_enhancement and task_id == 5:
        social_media_guidance = get_social_media_enhanced_guidance(text2annotate)
        social_media_section = (
            "### Social Media Analysis\n"
            f"{social_media_guidance}\n\n"
        )
    
    prompt = (
        "/no_think\n"
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, and ensure the final annotation result is enclosed in <label> tags.\n\n"
        "### Core Task\n"
        f"Task: {task_description}\n\n"
        f"{guidance_section}"
        f"{social_media_section}"
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
        "5. For social media text (Task 5), carefully consider the social media analysis if provided.\n"
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

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str, is_code_generation: bool = False, use_diversity: bool = True, use_similarity: bool = False, use_task_aware: bool = False, task_id: int = None, use_quality_filter: bool = False, quality_threshold: float = 0.5, use_cot: bool = False, balance_sentiment: bool = False, use_importance_weighting: bool = False, use_metalearning: bool = False) -> str:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
                           
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
    
                                        
    if balance_sentiment and task_id == 5:
        selected_examples = balance_sentiment_samples(selected_examples)
    
                                          
    if use_importance_weighting and task_id is not None and 1 <= task_id <= 4:
        selected_examples = select_examples_by_importance(selected_examples, task_id)
    
                                       
    if use_metalearning and task_id is not None:
        selected_examples = select_examples_by_metalearning(selected_examples, task_id, text2annotate)
    
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
\
\
\
\
\
\
\
\
\
\
\
\
\
       
                                      
               
                                        
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
\
\
\
\
       
                                    
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
\
\
\
\
\
\
\
\
\
\
\
       
                        
                            
                                       
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
\
\
\
\
\
\
\
\
\
\
\
\
\
       
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
\
\
\
\
\
\
\
       
    input_text = example.get('input', '')
    output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
    
    reasoning_map = {
        1: f"""**步骤1：解析输入列表**
输入列表: {input_text}
识别列表中的所有整数

**步骤2：识别所有整数对**
生成所有可能的整数对（有序或无序）

**步骤3：计算每对整数的绝对差值**
对于每一对，计算|a - b|

**步骤4：比较所有差值**
找出所有差值中的最小值

**步骤5：验证最小差值**
确保没有更小的差值被遗漏

**步骤6：确认结果**
最小绝对差值 = {output_text}

**步骤7：得出结论**
答案是 {output_text}""",
        
        2: f"""**步骤1：解析输入文本**
输入文本: {input_text}
识别文本中的所有词

**步骤2：分词处理**
将文本分割成单词

**步骤3：识别名词**
根据词性识别名词

**步骤4：识别动词**
根据词性识别动词

**步骤5：统计名词数量**
名词数量 = {output_text.split(',')[0] if ',' in output_text else '?'}

**步骤6：统计动词数量**
动词数量 = {output_text.split(',')[1] if ',' in output_text else '?'}

**步骤7：验证结果**
确认统计的准确性

**步骤8：得出结论**
答案是 {output_text}""",
        
        3: f"""**步骤1：解析起始数**
起始数: {input_text}
验证输入是正整数

**步骤2：应用Collatz猜想规则**
如果n是偶数，n = n/2
如果n是奇数，n = 3n + 1

**步骤3：迭代计算**
按照规则逐步计算，直到n=1

**步骤4：记录每一步**
追踪每个中间值

**步骤5：统计步数**
计算从起始数到1的总步数

**步骤6：验证步数**
确认步数计算的准确性

**步骤7：得出结论**
答案是 {output_text}""",
        
        4: f"""**步骤1：解析输入**
输入: {input_text}
识别需要拼接的字符串

**步骤2：识别字符串**
第一个字符串: {input_text.split(',')[0] if ',' in input_text else '?'}
第二个字符串: {input_text.split(',')[1] if ',' in input_text else '?'}

**步骤3：确定拼接顺序**
按照输入顺序拼接

**步骤4：执行拼接**
第一个字符串 + 第二个字符串

**步骤5：验证拼接结果**
确认没有遗漏或多余字符

**步骤6：检查最终结果**
拼接结果 = {output_text}

**步骤7：得出结论**
答案是 {output_text}"""
    }
    
    return reasoning_map.get(task_id, "")


def compute_example_importance(example: dict, task_id: int = None) -> float:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    input_text = example.get('input', '')
    output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
    
    importance = 0.5         
    
                                                   
    input_length = len(input_text)
    if input_length < 15:
        importance += 0.1
    elif input_length < 30:
        importance += 0.3
    else:
        importance += 0.5
    
                     
    digits = sum(1 for c in input_text if c.isdigit())
    digit_ratio = digits / max(input_length, 1)
    if digit_ratio > 0.3:
        importance += 0.3
    elif digit_ratio > 0.1:
        importance += 0.15
    
               
    special_chars = sum(1 for c in input_text if not c.isalnum() and c not in ' \t\n')
    if special_chars > 3:
        importance += 0.2
    elif special_chars > 1:
        importance += 0.1
    
              
    output_length = len(str(output_text))
    if output_length > 10:
        importance += 0.2
    elif output_length > 5:
        importance += 0.1
    
               
    if task_id is not None:
        if task_id == 1:  # Closest Integers
                          
            comma_count = input_text.count(',')
            importance += min(comma_count * 0.1, 0.3)
        elif task_id == 2:  # Count Nouns and Verbs
                     
            word_count = len(input_text.split())
            importance += min(word_count * 0.05, 0.3)
        elif task_id == 3:  # Collatz Conjecture
                      
            try:
                start_num = int(input_text)
                if start_num > 100:
                    importance += 0.3
                elif start_num > 50:
                    importance += 0.15
            except:
                pass
        elif task_id == 4:  # Concat Strings
                                   
            parts = input_text.split(',')
            if len(parts) == 2:
                len1, len2 = len(parts[0]), len(parts[1])
                balance = 1 - abs(len1 - len2) / max(len1, len2, 1)
                importance += balance * 0.2
    
                   
    importance = max(0.0, min(1.0, importance))
    
    return importance


def select_examples_by_importance(all_examples: list[dict], task_id: int = None, top_k: int = None) -> list[dict]:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    if not all_examples:
        return []
    
    if top_k is None:
        top_k = len(all_examples)
    
                  
    example_importances = []
    for i, example in enumerate(all_examples):
        importance = compute_example_importance(example, task_id)
        example_importances.append({
            'example': example,
            'index': i,
            'importance': importance,
            'input_length': len(example.get('input', ''))
        })
    
                    
    example_importances.sort(key=lambda x: x['importance'], reverse=True)
    
                
    selected = example_importances[:top_k]
    
                               
                                   
    if len(selected) >= 3:
        lengths = [item['input_length'] for item in selected]
        length_range = max(lengths) - min(lengths)
        
                                    
        if length_range < 10 and len(example_importances) > top_k:
                              
            remaining = example_importances[top_k:]
            remaining.sort(key=lambda x: x['input_length'])
            
                           
            if remaining:
                            
                shortest = min(remaining, key=lambda x: x['input_length'])
                longest = max(remaining, key=lambda x: x['input_length'])
                
                         
                if len(selected) > 2:
                    selected[-1] = shortest
                    selected[-2] = longest
    
            
    return [item['example'] for item in selected]


def format_example_with_cot(example: dict, task_id: int, is_code_generation: bool = False) -> str:
\
\
\
\
\
\
\
       
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
\
\
\
\
\
\
\
\
\
\
\
       
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
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
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


def extract_social_media_features(text: str) -> dict:
\
\
\
\
\
       
    import re
    
    features = {
        'emojis': [],
        'hashtags': [],
        'mentions': [],
        'urls': [],
        'punctuation': {},
        'length': len(text)
    }
    
                                 
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+"
    )
    features['emojis'] = emoji_pattern.findall(text)
    
               
    features['hashtags'] = re.findall(r'#\w+', text)
    
                
    features['mentions'] = re.findall(r'@\w+', text)
    
           
    features['urls'] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
            
    features['punctuation'] = {
        '!': text.count('!'),
        '?': text.count('?'),
        '...': text.count('...'),
        'ALL_CAPS': len([word for word in text.split() if word.isupper() and len(word) > 1])
    }
    
    return features


def analyze_sentiment_indicators(text: str, features: dict) -> dict:
\
\
\
\
\
\
       
    indicators = {
        'positive_words': [],
        'negative_words': [],
        'positive_emojis': [],
        'negative_emojis': [],
        'exclamations': features['punctuation'].get('!', 0),
        'questions': features['punctuation'].get('?', 0),
        'capital_emphasis': features['punctuation'].get('ALL_CAPS', 0)
    }
    
                        
    positive_words = ['happy', 'joy', 'love', 'great', 'awesome', 'excellent', 'wonderful', 'amazing', 'fantastic', 'positive', 'good', 'best', 'love', 'like', 'enjoy', 'smile', 'laugh', 'pleased', 'satisfied', 'delighted']
    negative_words = ['sad', 'depressed', 'angry', 'hate', 'terrible', 'awful', 'horrible', 'disappointed', 'upset', 'frustrated', 'annoyed', 'irritated', 'worried', 'anxious', 'stressed', 'miserable', 'unhappy', 'negative', 'bad', 'worst']
    
    text_lower = text.lower()
    
            
    for word in positive_words:
        if word in text_lower:
            indicators['positive_words'].append(word)
    
            
    for word in negative_words:
        if word in text_lower:
            indicators['negative_words'].append(word)
    
                              
    positive_emojis = ['😀', '😃', '😄', '😁', '😆', '😅', '🤣', '😂', '🙂', '😊', '😇', '🥰', '😍', '🤩', '😘', '😗', '😚', '😙', '🥲', '😋', '😛', '😜', '🤪', '😝', '🤑', '🤗', '🤭', '🤫', '🤔', '🤐', '🤨', '😐', '😑', '😶', '😏', '😒', '🙄', '😬', '🤥', '😌', '😔', '😪', '🤤', '😴', '😷', '🤒', '🤕', '🤢', '🤮', '🤧', '🥵', '🥶', '🥴', '😵', '🤯', '🤠', '🥳', '🥸', '😎', '🤓', '🧐']
    negative_emojis = ['😞', '😔', '😟', '😕', '🙁', '😣', '😖', '😫', '😩', '🥺', '😢', '😭', '😤', '😠', '😡', '🤬', '😈', '👿', '💀', '☠️', '💩', '🤡', '👹', '👺', '👻', '👽', '👾', '🤖']
    
    for emoji in features['emojis']:
        if emoji in positive_emojis:
            indicators['positive_emojis'].append(emoji)
        elif emoji in negative_emojis:
            indicators['negative_emojis'].append(emoji)
    
    return indicators


def detect_sarcasm(text: str, indicators: dict) -> float:
\
\
\
\
\
\
       
    sarcasm_score = 0.0
    
                            
    has_positive = len(indicators['positive_words']) > 0 or len(indicators['positive_emojis']) > 0
    has_negative = len(indicators['negative_words']) > 0 or len(indicators['negative_emojis']) > 0
    
    if has_positive and has_negative:
        sarcasm_score += 0.3
    
              
    if indicators['exclamations'] >= 3:
        sarcasm_score += 0.2
    
                      
    if indicators['capital_emphasis'] >= 2:
        sarcasm_score += 0.2
    
                     
    if indicators['questions'] >= 2:
        sarcasm_score += 0.15
    
             
    sarcasm_markers = ['sure', 'yeah right', 'totally', 'definitely', 'obviously', 'clearly']
    text_lower = text.lower()
    for marker in sarcasm_markers:
        if marker in text_lower:
            sarcasm_score += 0.1
    
                
    return min(sarcasm_score, 1.0)


def balance_sentiment_samples(examples: list[dict], target_ratio: float = 0.5) -> list[dict]:
\
\
\
\
\
\
       
    if not examples:
        return examples
    
            
    positive_samples = []
    negative_samples = []
    
    for example in examples:
        output = example['output'][0] if isinstance(example['output'], list) else example['output']
        if output.lower() == 'good review':
            positive_samples.append(example)
        else:
            negative_samples.append(example)
    
                           
    if not positive_samples or not negative_samples:
        print(f"警告：无法平衡情感样本 - 正样本数：{len(positive_samples)}，负样本数：{len(negative_samples)}")
        return examples
    
                  
    total_samples = len(examples)
    target_positive = int(total_samples * target_ratio)
    target_negative = total_samples - target_positive
    
            
    balanced = []
    
                  
    if len(positive_samples) < target_positive:
        needed = target_positive - len(positive_samples)
        for i in range(needed):
            balanced.append(positive_samples[i % len(positive_samples)])
    else:
        balanced = positive_samples[:target_positive]
    
                  
    if len(negative_samples) < target_negative:
        needed = target_negative - len(negative_samples)
        for i in range(needed):
            balanced.append(negative_samples[i % len(negative_samples)])
    else:
        balanced.extend(negative_samples[:target_negative])
    
    return balanced


def get_social_media_enhanced_guidance(text: str) -> str:
\
\
\
\
\
       
              
    features = extract_social_media_features(text)
    
             
    indicators = analyze_sentiment_indicators(text, features)
    
          
    sarcasm_score = detect_sarcasm(text, indicators)
    
    guidance = (
        "### 社交媒体情感分析指导\n\n"
        "**特殊元素识别**：\n"
    )
    
               
    if features['emojis']:
        guidance += f"- 检测到 {len(features['emojis'])} 个emoji：{', '.join(features['emojis'][:5])}\n"
    else:
        guidance += "- 未检测到emoji\n"
    
                 
    if features['hashtags']:
        guidance += f"- 检测到 {len(features['hashtags'])} 个hashtag：{', '.join(features['hashtags'][:5])}\n"
    else:
        guidance += "- 未检测到hashtag\n"
    
                  
    if features['mentions']:
        guidance += f"- 检测到 {len(features['mentions'])} 个@mention：{', '.join(features['mentions'][:5])}\n"
    else:
        guidance += "- 未检测到@mention\n"
    
    guidance += "\n**情感指示符**：\n"
    
            
    if indicators['positive_words']:
        guidance += f"- 积极词汇：{', '.join(indicators['positive_words'][:5])}\n"
    
            
    if indicators['negative_words']:
        guidance += f"- 消极词汇：{', '.join(indicators['negative_words'][:5])}\n"
    
               
    if indicators['positive_emojis']:
        guidance += f"- 积极emoji：{', '.join(indicators['positive_emojis'][:3])}\n"
    
    if indicators['negative_emojis']:
        guidance += f"- 消极emoji：{', '.join(indicators['negative_emojis'][:3])}\n"
    
            
    guidance += "\n**反讽检测**：\n"
    if sarcasm_score > 0.5:
        guidance += f"- ⚠️ 高度可能存在反讽（分数：{sarcasm_score:.2f}）\n"
        guidance += "- 注意：可能存在矛盾的情感表达或夸张语气\n"
    elif sarcasm_score > 0.3:
        guidance += f"- ⚠️ 可能存在反讽（分数：{sarcasm_score:.2f}）\n"
    else:
        guidance += f"- ✅ 未检测到明显反讽（分数：{sarcasm_score:.2f}）\n"
    
            
    guidance += "\n**分析建议**：\n"
    guidance += "1. 综合考虑文本内容、emoji、hashtag等多种元素\n"
    guidance += "2. 注意区分字面含义和实际情感\n"
    guidance += "3. 对于包含矛盾元素的文本，仔细分析主要情感倾向\n"
    guidance += "4. 如果反讽分数较高，需要更加谨慎地判断\n"
    
    return guidance


# ========================================
                          
# ========================================

def extract_example_features(example: dict, task_id: int, text2annotate: str = None) -> dict:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    input_text = example.get('input', '')
    output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
    
    features = {}
    
             
    input_length = len(input_text)
    output_length = len(str(output_text))
    
    features['input_length'] = input_length
    features['output_length'] = output_length
    features['total_length'] = input_length + output_length
    
          
    digits = sum(1 for c in input_text if c.isdigit())
    features['digit_density'] = digits / max(input_length, 1)
    
            
    special_chars = sum(1 for c in input_text if not c.isalnum() and c not in ' \t\n')
    features['special_char_ratio'] = special_chars / max(input_length, 1)
    
                          
    if text2annotate:
                       
        all_texts = [text2annotate, input_text]
        similarities = compute_tfidf_similarity(all_texts)
        features['similarity'] = float(similarities[0]) if len(similarities) > 0 else 0.0
    else:
        features['similarity'] = 0.0
    
             
    quality_score = compute_example_quality_score(example)
    features['quality_score'] = quality_score
    
               
    if task_id == 1:  # Closest Integers
        comma_count = input_text.count(',')
        features['comma_count'] = comma_count
        number_count = input_text.count(' ') + 1          
        features['number_count'] = number_count
        
    elif task_id == 2:  # Count Nouns and Verbs
        word_count = len(input_text.split())
        features['word_count'] = word_count
        
    elif task_id == 3:  # Collatz Conjecture
        try:
            start_num = int(input_text)
            features['start_num'] = start_num
                                        
            features['complexity'] = min(start_num / 100.0, 1.0)
        except:
            features['start_num'] = 0
            features['complexity'] = 0.0
            
    elif task_id == 4:  # Concat Strings
        parts = input_text.split(',')
        features['string_count'] = len(parts)
        if len(parts) == 2:
            len1, len2 = len(parts[0]), len(parts[1])
            features['string_balance'] = 1 - abs(len1 - len2) / max(len1, len2, 1)
        else:
            features['string_balance'] = 0.0
            
    elif task_id == 5:  # Sentiment Analysis
                        
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]"
        )
        emojis = emoji_pattern.findall(input_text)
        features['emoji_count'] = len(emojis)
        hashtag_count = input_text.count('#')
        features['hashtag_count'] = hashtag_count
        
    elif task_id == 6:  # NLI
                                             
        sep_pos = input_text.find('[SEP]')
        if sep_pos > 0:
            premise_len = sep_pos
            hypothesis_len = len(input_text) - sep_pos - 5
            features['premise_len'] = premise_len
            features['hypothesis_len'] = hypothesis_len
        else:
            features['premise_len'] = 0
            features['hypothesis_len'] = 0
            
    elif task_id == 7:  # Jeopardy QA
        features['contains_category'] = 'Category:' in input_text or 'Clue:' in input_text
        features['input_complexity'] = min(len(input_text) / 100.0, 1.0)
    
    return features


def predict_example_utility(features: dict, task_id: int, weights: dict = None) -> float:
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    if weights is None:
                      
        weights = {
                    
            'quality_score': 0.25,             
            'similarity': 0.30,                
            'input_length': 0.15,
            'output_length': 0.10,
            'digit_density': 0.05,
            'special_char_ratio': 0.05,
            
                    
            'comma_count': 0.05,
            'number_count': 0.05,
            'word_count': 0.05,
            'complexity': 0.10,
            'string_balance': 0.10,
            'emoji_count': 0.08,
            'hashtag_count': 0.08,
            'premise_len': 0.05,
            'hypothesis_len': 0.05,
            'input_complexity': 0.10
        }
    
    utility = 0.0
    total_weight = 0.0
    
            
    for feature_name, feature_value in features.items():
        if feature_name in weights and weights[feature_name] > 0:
            weight = weights[feature_name]
            
                            
            if feature_name == 'input_length':
                normalized = min(feature_value / 100.0, 1.0)
            elif feature_name == 'output_length':
                normalized = min(feature_value / 50.0, 1.0)
            elif feature_name == 'total_length':
                normalized = min(feature_value / 150.0, 1.0)
            elif feature_name == 'similarity':
                normalized = feature_value              
            elif feature_name == 'quality_score':
                normalized = feature_value              
            elif feature_name in ['digit_density', 'special_char_ratio']:
                normalized = feature_value              
            elif feature_name in ['complexity', 'string_balance']:
                normalized = feature_value              
            elif feature_name in ['emoji_count', 'hashtag_count', 'comma_count', 'number_count', 'word_count']:
                normalized = min(feature_value / 10.0, 1.0)
            elif feature_name in ['premise_len', 'hypothesis_len']:
                normalized = min(feature_value / 100.0, 1.0)
            elif feature_name == 'input_complexity':
                normalized = feature_value              
            elif feature_name == 'contains_category':
                normalized = 1.0 if feature_value else 0.0
            else:
                normalized = 0.5       
            
                  
            utility += weight * normalized
            total_weight += weight
    
                 
    if total_weight > 0:
        utility = utility / total_weight
    
    return max(0.0, min(1.0, utility))


def select_examples_by_metalearning(all_examples: list[dict], task_id: int, text2annotate: str, top_k: int = None) -> list[dict]:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    if not all_examples:
        return []
    
    if top_k is None:
        top_k = len(all_examples)
    
                  
    example_utilities = []
    for i, example in enumerate(all_examples):
        features = extract_example_features(example, task_id, text2annotate)
        utility = predict_example_utility(features, task_id)
        
        example_utilities.append({
            'example': example,
            'index': i,
            'utility': utility,
            'input_length': len(example.get('input', '')),
            'features': features
        })
    
                      
    example_utilities.sort(key=lambda x: x['utility'], reverse=True)
    
                      
    selected = example_utilities[:top_k]
    
                                  
    if len(selected) >= 3:
        lengths = [item['input_length'] for item in selected]
        length_range = max(lengths) - min(lengths)
        
                                    
        if length_range < 15 and len(example_utilities) > top_k:
            remaining = example_utilities[top_k:]
            remaining.sort(key=lambda x: x['input_length'])
            
                           
            if remaining:
                shortest = min(remaining, key=lambda x: x['input_length'])
                longest = max(remaining, key=lambda x: x['input_length'])
                
                               
                if len(selected) > 2:
                    selected[-1] = shortest
                    selected[-2] = longest
    
                             
    selected.sort(key=lambda x: x['input_length'])
    
    print(f"Task {task_id} - 选择了 {len(selected)} 个示例，平均效用: {sum(item['utility'] for item in selected) / len(selected):.3f}")
    
    return [item['example'] for item in selected]


def organize_examples_by_difficulty(examples: list[dict]) -> dict:
\
\
\
\
\
\
\
\
\
\
\
       
    difficulty_levels = {
        'easy': [],
        'medium': [],
        'hard': []
    }
    
    for example in examples:
        input_length = len(example.get('input', ''))
        
        if input_length < 30:
            difficulty_levels['easy'].append(example)
        elif input_length < 60:
            difficulty_levels['medium'].append(example)
        else:
            difficulty_levels['hard'].append(example)
    
    return difficulty_levels