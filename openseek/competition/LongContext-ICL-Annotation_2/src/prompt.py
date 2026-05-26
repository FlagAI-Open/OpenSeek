#获取问题
import json
import re
import requests
import openai
from collections import Counter
def get_input_by_id(task_file_path: str, sample_id: str) -> str:
    with open(task_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data.get("test_samples", []):
        if str(item.get("id")) == str(sample_id):
            return item.get("input", "")
    return ""

#提取答案
def count_answer(text: str, task_id: int = 1):
    """
    匹配is之后的答案内容，并统计出现次数最高的内容。
    对于Task 1-3, 5-7，主要提取特定的格式。
    """
    if not text:
        return None

    # 针对不同任务类型优化匹配正则
    if task_id == 1:
        # Task 1: Closest Integers (单个整数)
        pattern = re.compile(r'\bis\s+([-+]?\d+)', re.IGNORECASE)
    elif task_id == 2:
        # Task 2: Count Nouns/Verbs (单个整数)
        pattern = re.compile(r'(?:\bis\b|\bAnswer:)\s+(\d+)', re.IGNORECASE)
    elif task_id == 3:
        # Task 3: Collatz Conjecture (整数列表 [1, 2, 3])
        # 兼容 is/Answer/Output 以及 \boxed{[...]} 形式
        pattern = re.compile(r'(?:\bis\b|\bAnswer:|\bOutput:)?\s*(\\boxed\{)?(\[[\d\s,.-]*\])(\})?', re.IGNORECASE)
    elif task_id == 4:
        # Task 4: Concat Strings (字符串)
        # 兼容 is / should be / \boxed{...}，并优先匹配带引号的内容
        pattern = re.compile(r'(?:\bis\b|\bAnswer:|\bOutput:|\bshould\s+be\b)\s+["\']?(.*?)["\']?(?:\.|\"|\n|$)', re.IGNORECASE)

    elif task_id == 5:
        # Task 5: Tweet Sadness (Sad / Not sad)
        # 提取 is 或者 Final Answer: 或者 Output: 之后的答案
        pattern = re.compile(r'(?:\bis\b|\bFinal\s+Answer:|\bAnswer:|\bAnswer|\bOutput:)\s*[:*]*\s*["\']?(sad|not sad)["\']?', re.IGNORECASE)
    elif task_id == 6:
        # Task 6: same genre (Y / N)
        # 提取 is 或者 Answer: 或者 Output: 或者 should be 之后的答案
        pattern = re.compile(r'(?:\bis\b|\bAnswer:|\bOutput:|\bshould\s+be\b)\s+([YN])\b', re.IGNORECASE)
    elif task_id == 7:
        # Task 7: Jeopardy answer (All lower case)
        # 提取 is 或者 Answer: 或者 should be 之后的答案
        pattern = re.compile(r'(?:\bis\b|\bAnswer:|\bshould\s+be\b)\s+["\']?(.*?)["\']?(?:\.|\"|\n|$)', re.IGNORECASE)
    elif task_id == 8:
        # Task 8: Kernel Generation (Code block)
        # 优先匹配 python 代码块
        block_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
        block_match = block_pattern.search(text)
        if block_match:
            return block_match.group(1).strip()
        
        # 尝试匹配从 import 开始到 ``` 结束的内容
        import_pattern = re.compile(r'(import\s+.*?)```', re.DOTALL)
        import_match = import_pattern.search(text)
        if import_match:
            return import_match.group(1).strip()
        # 兜底匹配 involves "is"
        pattern = re.compile(r'\bis\s*(?:```python|```)?(.*)', re.DOTALL | re.IGNORECASE)
    else:
        # 通用匹配：匹配 is 之后的第一个词或到句号为止
        pattern = re.compile(r'\bis\s+([^.\n]+)', re.IGNORECASE)

    matches = pattern.findall(text)
    
    final_prediction = None
    if matches:
        new_matches = []
        for m in matches:
            boxed_match = re.search(r'\\boxed\{(.*?)\}', m)
            if boxed_match:
                new_matches.append(boxed_match.group(1))
            else:
                new_matches.append(m)
        matches = new_matches

        # 清洗空白及常见的干扰符号 [ ] ' "
        matches = [m.strip().strip("[]'\" ").strip() for m in matches if m.strip()]
        
        if task_id == 4:
            # 任务四：去掉中间空格
            matches = [m.replace(" ", "") for m in matches]

        if task_id == 5:
            # 任务五：答案首字母大写 (Sad / Not sad)
            matches = [m.capitalize() for m in matches]

        if task_id == 7:
            # 任务七：将匹配到的答案统一转为小写
            matches = [m.lower() for m in matches]

        if matches:
            # 统计出现次数最多的项
            counts = Counter(matches)
            max_count = max(counts.values())
            # 找到所有等于最大次数的候选项
            candidates = set(k for k, v in counts.items() if v == max_count)
            # 如果最大次数重复，优先使用原序列中最后出现的答案
            for m in reversed(matches) :
                if m in candidates:
                    final_prediction = m
                    break
    
    # 清理控制字符以防生成非法的 JSONL 文件 (JSONDecodeError: Invalid control character)
    if isinstance(final_prediction, str):
        # 移除 0x00-0x1F 的非标准控制字符 (char < 32)，保留 \n \t \r
        # 但要注意 \n \r 在 JSON 字符串中通常需要转义，json.dumps 会自动完成
        # 真正引起 JSONDecodeError 的通常是 \x08 (Backslash), \x0c (Formfeed) 等
        final_prediction = "".join(ch for ch in final_prediction if ord(ch) >= 32 or ch in "\n\r\t")

    return final_prediction if final_prediction is not None else "no"

def call_nvidia(prompt_text: str, max_tokens: int = 600, stage_name: str = "Nvidia") -> str:
    URL="http://0.0.0.0:2026/v1/completions"
    data = {
        "model": "/root/ascend/log/debug/Qwen3-4B",
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "repetition_penalty": 1.1,
        "top_p": 0.9,
    }

    whole_result = ""
    try:
        resp = requests.post(URL, json=data, timeout=600)
        resp_json = resp.json()
        if "choices" in resp_json:
            whole_result = resp_json["choices"][0]["text"]
        else:
            print(f"\n[{stage_name}] Error response: {resp_json}")
    except Exception as e:
        print(f"\n[{stage_name}] Exception occurred: {e}")

    print(f"\n[{stage_name}] Model returned content:\n{whole_result}\n" + "="*50)
    return whole_result

def call_ascend(prompt_text: str, stage_name: str = "Ascend") -> str:
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"

    messages = [
        {"role": "system", "content": "You are a highly capable data annotation AI."},
        {"role": "user", "content": prompt_text}
    ]
    whole_result = ""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=2000,
            stream=False,
        )
        whole_result = response.choices[0].message.content
    except Exception as e:
        print(f"\n[{stage_name}] Exception occurred: {e}")

    print(f"\n[{stage_name}] Model returned content:\n{whole_result}\n" + "="*50)
    return whole_result

def build_icl_prompt(task_id: int, task_file_path: str, input_text: str) -> str:
    """
    动态构造少样本长文本指令 (ICL Prompt)，融入 Task 文件中的定义和 Examples。
    根据任务 ID 动态设置不同的上下文长度上限。
    """
    

    with open(task_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    definition = dataset.get("Definition", [""])[0]

    # 为8个任务定义不同的描述，避免模型混淆示例
    task_type_map = {
        1: "the single integer is",
        2: "the number of nouns/verbs is",
        3: "the list of integers result is",
        4: "the concatenated string is",
        5: "the sentiment judgment is",
        6: "the genre similarity judgment (Y/N) is",
        7: "the answer in lower case is",
        8: "the implemented Triton code is",
    }
    answer_desc = task_type_map.get(task_id, "the final answer is")

    
    base_prompt = f"Task Definition:\n{definition}\n\n"

    # 根据任务 ID 定义特有的 Rules 提示词
    rules_map = {
        #1: "- Output ONLY the single integer.\n- Carefully compare the absolute difference between the numbers.",
        #2: "- Output ONLY the count as an integer.\n- Identify nouns/verbs precisely within the given context.",
        #3: "- Output ONLY the final sequence result.\n- Follow the Collatz conjecture progression strictly.",
        #4: "- Output ONLY the concatenated string.\n- Do NOT include any spaces, quotes, or box formatting in final value.",
        #5: "Task: Analyze the given tweet and classify the author's emotional state as either \"Sad\" or \"Not sad\".\nClassification Guidelines:\n1. \"Sad\" if: Pain & Suffering, Negative Circumstances (bad service, lost items, deaths, breakups), Strong Negative Judgments (\"horrific\", \"awful\", \"disappointing\"), Anger with Pain (helplessness/frustration). \n2. \"Not sad\" if: Neutral/Informative, Positive/Excited, Humor/Sarcasm, Admiration/Envy, Anger without Pain (insults, political rants without personal hurt), Generic Negativity (trivial \"hate\").\nLogic Patterns:\n- The \"Die/Dying\" Rule: \"I'm dying/dead\" is almost always \"Not sad\" (laughter).\n- The \"Hate\" Rule: \"I hate [X]\" is \"Sad\" if X is a situation affecting them (bad service), but \"Not sad\" if it's a general preference/joke.\n- The \"Disappointed\" Rule: Expressing disappointment (events/products/people) is consistently labeled \"Sad\".\n- The \"Service\" Rule: Complaints about companies (Airlines/Banks) are labeled \"Sad\".\n- The \"Politics\" Rule: Political tweets are \"Sad\" if they express fear/shame/disappointment; they are \"Not sad\" if they are purely mocking/insulting the opposition. \nEmoji Interpretation: 😭 usually indicates \"Not sad\" in humorous context.\nOutput Format: Respond with exactly: Sad or Not sad.",
        6: "Output ONLY 'Y' or 'N'. No explanation.\n- Y = both sentences clearly and naturally fit the given genre.\n- N = at least one sentence does NOT naturally fit the given genre.\n- Do NOT use imagination. Do not assume they are parts of a story unless obvious.\n- Judge based on the content and style of the sentences themselves.",
        #7: "- Output ONLY the short answer string.\n- Convert your final answer to all lowercase.",
        #8: "- Output ONLY the Triton code block starting with ```python.\n- Include all necessary library imports at the top."
    }
    
    if task_id in rules_map:
        base_prompt += f"Rules:\n{rules_map[task_id]}\n\n"

    if task_id == 4:
        # Add examples to help the model understand Task 4
        base_prompt += "Examples:\n"
        for ex in dataset.get("examples", [])[-3:]:
            base_prompt += f"Input: {ex['input']}\n"
            output_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            base_prompt += f"The input's concatenate string is {output_str}\n\n"

        target_prompt = "Now solve the target input.\n"
        target_prompt += (
            f"--- Target ---\nInput: {input_text}\n"
            f"Directly provide the char's concatenated string. Do not count characters or explain.\n"
            f"The input's concatenate string is "
        )
    elif task_id == 5:
        base_prompt += "You are an expert data annotator."  #容易导致过度分析
        base_prompt += (
        "Think the daily life's sad and not sad about input, then output the final answer clearly.\n\n"
        )

        target_prompt = "Now solve the target input.\n"
        target_prompt += (
            f"--- Target ---\nInput: {input_text}\n"
            f"Analysis: Let's solve this step-by-step based on the task definition. "
            f"Infer the answer from daily life,and finally determine that {answer_desc} "
        )
    elif task_id == 7:
        # Add examples to help the model understand Task 7
        base_prompt += "Examples:\n"
        for ex in dataset.get("examples", [])[-1:]:
            base_prompt += f"Input: {ex['input']}\n"
            output_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            base_prompt += f"The answer is {output_str.lower()}\n\n"

        target_prompt = "Now solve the target input.\n"
        target_prompt += (
            f"--- Target ---\nInput: {input_text}\n"
            f"Directly provide the answer in lower case. \n"
            f"The answer is "
        )
    elif task_id == 8:
        # Task 8: Kernel Generation (Code block)
        # Add examples to help the model understand Task 8
        base_prompt += "Examples:\n"
        for ex in dataset.get("examples", [])[-1:]:
            base_prompt += f"Input: {ex['input']}\n"
            output_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            base_prompt += f"The implemented Triton code is:\n```python\n{output_str}\n```\n\n"

        target_prompt = "Now solve the target input.\n"
        target_prompt += (
            f"--- Target ---\nInput: {input_text}\n"
            f"Please implement the complete, runnable Triton kernel code at once, including all necessary imports (torch, triton, triton.language, etc.). Do not provide any analysis or explanation.\n"
            f"```python\n"
        )
    else:
        base_prompt += "You are an expert data annotator."  #容易导致过度分析
        base_prompt += (
        "Think step-by-step in Analysis, then output the final answer clearly.\n\n"
        )

        target_prompt = "Now solve the target input.\n"
        target_prompt += (
            f"--- Target ---\nInput: {input_text}\n"
            f"Analysis: Let's solve this step-by-step based on the task definition. "
            f"First, I will analyze the input, perform the necessary calculations, and finally determine that {answer_desc} "
        )
    
    final_prompt = base_prompt + target_prompt
    return final_prompt

#总流程
def execute_annotation(task_id: int, task_file_path_or_input_text: str, input_text_or_backend: str = None, backend: str = "nvidia"):
    """
    Backward compatible entry:
    - New style: execute_annotation(task_id, task_file_path, input_text, backend)
    - Old style: execute_annotation(task_id, input_text, backend)
    """
    # New style: second arg is task_file_path and third arg is input_text
    if isinstance(task_file_path_or_input_text, str) and task_file_path_or_input_text.endswith('.json'):
        task_file_path = task_file_path_or_input_text
        input_text = input_text_or_backend
    else:
        # Old style fallback: second arg is input_text, infer task file from task_id
        task_file_path = {
            1: '../data/openseek-1_closest_integers.json',
            2: '../data/openseek-2_count_nouns_verbs.json',
            3: '../data/openseek-3_collatz_conjecture.json',
            4: '../data/openseek-4_conala_concat_strings.json',
            5: '../data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
            6: '../data/openseek-6_mnli_same_genre_classification.json',
            7: '../data/openseek-7_jeopardy_answer_generation_all.json',
            8: '../data/openseek-8_kernel_generation.json',
        }.get(task_id)
        input_text = task_file_path_or_input_text

        # In old style, third positional arg is backend.
        if input_text_or_backend in ("nvidia", "ascend"):
            backend = input_text_or_backend

    if not task_file_path:
        raise ValueError(f"Unknown task_id: {task_id}")

    prompt_text = build_icl_prompt(task_id, task_file_path, input_text)
    
# 为每个任务单独定义不同的上下文参考字符长度（根据任务复杂度和 Qwen3-4B 能力调整）
    task_context_map = {
        1: 2000,  # 数值差：适中。
        2: 1500,  # 词性计数：适中。
        3: 1800,  # 序列规律：高 ICL。
        4: 2000,  # 字符串拼接：低需求。
        5: 2000,  # 情感分析：典型分类。
        6: 1500,  # 语义匹配。
        7: 2000,  # 知识问答（Trivia）。
        8: 3000,  # Triton 代码生成：需要大量代码示例。
    }
    max_context_chars = task_context_map.get(task_id, 40000)

    # 第一轮：通过构造的长上下文大模型进行标注（保持不变）
    if backend == "nvidia":
        response = call_nvidia(prompt_text, max_tokens = max_context_chars, stage_name="First Call/Nvidia")
    elif backend == "ascend":
        response = call_ascend(prompt_text, stage_name="First Call/Ascend")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # 已经简化：仅一次调用。
    # 根据 task_id 定制化提取 "is" 之后的任务结果
    prediction = count_answer(response, task_id)
    
    return prediction if prediction else "no"

def run_annotation_by_id(task_id: int, task_file_path: str, sample_id: str, backend: str = "nvidia"):
    input_text = get_input_by_id(task_file_path, sample_id)
    if not input_text:
        return None
    return execute_annotation(task_id, task_file_path, input_text, backend)
