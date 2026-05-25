
import re
from collections import Counter
from transformers import AutoTokenizer
from config import NANOBOT_HOME

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
        "**You may complete any internal reasoning process, but your FINAL OUTPUT MUST consist solely of the annotation result wrapped in <label> tags (no other content whatsoever).**\n"
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
            example_str = f"- {input_text} <label> {output_text} </label>\n"
            examples_str += example_str
        else:
            return examples_str, i
    return examples_str

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str, args) -> str:
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

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
                example_str = f"- {input_text} <label> {output_text} </label>\n"
                examples_str += example_str
            else:
                # 超过长度限制，返回已拼接的示例和已选数量
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue
    # 遍历完所有示例且未超长度，返回完整拼接结果
    return examples_str


def select_examples_by_genre(all_examples: list[dict], task_description: str, text2annotate: str, args, n_per_genre: int = 10) -> str:
    """
        Select examples with the same genre as the input text, keep balanced Y/N ratio (50% Y / 50% N).
        all_examples:
            A list of examples, where each example is a dict with keys 'input' and 'output' (no 'length' needed).
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review"}``,
        task_description:
            The description of the annotation task which may be used for example evaluation.
        text2annotate:
            The text that needs to be annotated, which contains the genre information at the end in format "Genre: X.".
        n_per_genre:
            Number of examples to select per genre, will try to maintain 1:1 Y/N ratio.
    """
    import re

    # 第一步：从输入文本中提取genre信息
    genre_match = re.search(r'Genre:\s*([^.]+)\.', text2annotate)
    if not genre_match:
        print(f"警告：无法从输入文本中提取genre信息，将使用默认的顺序选择示例")
        return select_examples(all_examples, task_description, text2annotate, args)

    target_genre = genre_match.group(1).strip()
    print(f"提取到目标genre: {target_genre}")

    # 第二步：筛选出所有相同genre的示例，并按标签分组
    same_genre_y = []
    same_genre_n = []
    for example in all_examples:
        try:
            input_text = example['input']
            example_genre_match = re.search(r'Genre:\s*([^.]+)\.', input_text)
            if example_genre_match and example_genre_match.group(1).strip() == target_genre:
                label = example['output'][0].strip() if example.get('output') else ''
                if label == 'Y':
                    same_genre_y.append(example)
                elif label == 'N':
                    same_genre_n.append(example)
        except KeyError as e:
            print(f"警告：示例缺少键{e}，跳过该示例")
            continue

    print(f"找到{len(same_genre_y)}个Y样本，{len(same_genre_n)}个N样本 (同genre)")

    # 第三步：均衡采样Y和N，尽量保持1:1比例
    y_target = n_per_genre // 2
    n_target = n_per_genre - y_target

    # 采样Y样本，不够的话用N补
    selected_y = same_genre_y[:y_target]
    remaining = n_per_genre - len(selected_y)
    selected_n = same_genre_n[:max(n_target, remaining)]

    # 如果N也不够，就把所有能拿的都拿上
    if len(selected_y) + len(selected_n) < n_per_genre:
        selected_n += same_genre_n[len(selected_n):n_per_genre - len(selected_y)]

    # 合并样本，Y在前N在后
    selected_examples = selected_y + selected_n
    print(f"最终选择{len(selected_examples)}个示例: {len(selected_y)}个Y, {len(selected_n)}个N")

    # 第四步：按照原select_examples的逻辑，选择合适数量的示例
    # 初始化Qwen3-4B的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 最大上下文长度限制
    target_length = 8192

    examples_str, token_num = "", 0
    # 遍历筛选后的同genre示例，计算token数
    for i, example in enumerate(selected_examples):
        try:
            # 提取input和output（兼容output是列表的情况）
            input_text = example['input']
            output_text = example['output'][0]

            # 计算input+output的token数
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens

            # 校验当前示例是否能加入（总长度不超限制）
            if length + token_num <= target_length:
                # 累加总token数：示例文本长度 + 格式符号的token数
                token_num += (length + 2 + 3 + 1 + 1)
                # 拼接单个示例字符串
                example_str = f"- {input_text} <label> {output_text} </label>\n"
                examples_str += example_str
            else:
                # 超过长度限制，返回已拼接的示例
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue

    # 遍历完所有示例且未超长度，返回完整拼接结果
    return examples_str


def select_n_examples(all_examples: list[dict], task_description: str, text2annotate: str, args, n_samples: int) -> str:
    """
        Select up to n_samples examples from all_examples to fit into the target context length (适配Qwen3-4B的token计算)
        all_examples:
            A list of examples, where each example is a dict with keys 'input' and 'output' (no 'length' needed).
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review"}``,
        task_description:
            The description of the annotation task which may be used for example evaluation.
        text2annotate:
            The text that needs to be annotated which may be used for example retrieval.
        n_samples:
            Maximum number of examples to select.
    """
    # 初始化Qwen3-4B的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 最大上下文长度限制
    target_length = 8192

    examples_str, token_num, selected_count = "", 0, 0
    # 遍历所有示例，基于Qwen3-4B的tokenizer计算token数
    for i, example in enumerate(all_examples):
        if selected_count >= n_samples:
            # 已经选够指定数量的示例，返回
            return examples_str
        try:
            # 提取input和output（兼容output是列表的情况）
            input_text = example['input']
            output_text = example['output'][0]

            # 计算input+output的token数
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens

            # 校验当前示例是否能加入（总长度不超限制）
            if length + token_num <= target_length:
                # 累加总token数：示例文本长度 + 格式符号的token数
                token_num += (length + 2 + 3 + 1 + 1)
                # 拼接单个示例字符串
                example_str = f"- {input_text} <label> {output_text} </label>\n"
                examples_str += example_str
                selected_count += 1
            else:
                # 超过长度限制，返回已拼接的示例
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue
    # 遍历完所有示例且未超长度/数量限制，返回完整拼接结果
    return examples_str


def select_examples_by_tag(all_examples: list[dict], task_description: str, text2annotate: str, args, target_tag: str, max_samples: int = 10) -> str:
    """
        Select examples with the same tag as target_tag for few-shot learning
        混合策略：一半同tag保证相关性，一半其他tag保证多样性，先随机混洗再按长度过滤
        自动过滤掉useless标记为true的样本
        all_examples:
            A list of examples, where each example has 'input', 'output', and 'tag' fields
        target_tag:
            The tag to filter examples by, only examples with tag == target_tag will be selected
        max_samples:
            Maximum number of examples to select, default 10
        Returns:
            Formatted examples string
    """
    import random
    # 计算对半分的数量（向上取整）
    half = (max_samples + 1) // 2

    # 1. 筛选同tag样本，过滤掉useless样本，随机打乱后取最多一半
    same_tag_examples = [
        ex for ex in all_examples
        if ex.get('tag') == target_tag and not ex.get('useless', False)
    ]
    total_same = len([ex for ex in all_examples if ex.get('tag') == target_tag])
    useless_same = total_same - len(same_tag_examples)
    print(f"📂 找到{total_same}个标签为[{target_tag}]的样本，过滤掉{useless_same}个useless样本，剩余{len(same_tag_examples)}个可用")
    random.shuffle(same_tag_examples)
    selected_same = same_tag_examples[:half]

    # 2. 筛选其他tag样本，过滤掉useless样本，随机打乱后取剩下需要的数量（最多另一半）
    other_tag_examples = [
        ex for ex in all_examples
        if ex.get('tag') != target_tag and 'tag' in ex and not ex.get('useless', False)
    ]
    total_other = len([ex for ex in all_examples if ex.get('tag') != target_tag and 'tag' in ex])
    useless_other = total_other - len(other_tag_examples)
    print(f"📂 找到{total_other}个其他标签的样本，过滤掉{useless_other}个useless样本，剩余{len(other_tag_examples)}个可用")
    random.shuffle(other_tag_examples)
    need_other = max_samples - len(selected_same)
    selected_other = other_tag_examples[:need_other]

    # 3. 合并并随机混洗所有选中的样本
    all_selected = selected_same + selected_other
    random.shuffle(all_selected)
    total_useless = useless_same + useless_other
    print(f"🔀 混合样本：{len(selected_same)}个同tag + {len(selected_other)}个其他tag，共{len(all_selected)}个，已随机混洗，累计过滤{total_useless}个useless样本")

    # 4. 调用select_n_examples按token长度过滤，保证不超上下文限制
    return select_n_examples(all_selected, task_description, text2annotate, args, n_samples=max_samples)


def select_examples_from_cache(cache_path: str, task_description: str, text2annotate: str, args, max_samples: int = 50) -> str:
    """
        从缓存文件中加载示例，自动按最大上下文长度过滤，返回格式化的示例字符串
        cache_path:
            缓存文件路径，每行格式为 `# 输入 <label> 输出 </label>`
        max_samples:
            最多选择的示例数量，默认50
        Returns:
            格式化后的示例字符串，符合prompt拼接格式
    """
    import os
    if not os.path.exists(cache_path):
        print(f"⚠️  缓存文件{cache_path}不存在，返回空示例")
        return ""

    # 读取缓存文件的所有行
    with open(cache_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    target_length = 8192  # 和其他选择函数保持一致的最大长度限制

    examples_str = ""
    token_num = 0
    selected_count = 0

    for line in lines:
        if selected_count >= max_samples:
            break
        try:
            # 计算当前行的token数
            line_tokens = len(tokenizer.encode(line, add_special_tokens=False))
            # 加上换行符的token数（+1）
            total_line_tokens = line_tokens + 1

            if token_num + total_line_tokens <= target_length:
                examples_str += line + '\n'
                token_num += total_line_tokens
                selected_count += 1
            else:
                print(f"⚠️  示例已达最大上下文长度限制，已选{selected_count}个示例，共{token_num}tokens")
                break
        except Exception as e:
            print(f"⚠️  处理缓存行失败: {str(e)}，跳过该行")
            continue

    print(f"📂 从缓存加载了{selected_count}个示例，共{token_num}tokens")
    return examples_str.rstrip('\n')


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

def extract_answer(text: str) -> tuple[str,list, dict]:
    """
    提取字符串中<label>标签内的所有内容（字符串形式），统计出现次数最多的内容
    :param text: 包含<label>标签的原始字符串
    :return: 出现次数最多的内容列表、所有内容的频次统计字典
    """
    answer = text.split('<label>')[-1].strip() if '<label>' in text else ''

    if (len(answer)<=8) or answer[-8:] != '</label>':
        return None
    
    return answer[:-8].strip()

# ========== 新增：任务8公共检测函数，消除重复逻辑 ==========
def _generate_triton_test_case(task: str, annotate_text: str, code_candidate: str, bot) -> str:
    """统一生成严格测试用例，重试最多2次"""
    test_prompt = '''
# Task: Generate STRICT Test Case for Triton Kernel

## Kernel Purpose
This kernel implements the following task:
{task}

## Original Problem Description
{annotate_text}

## Given Triton Kernel Code
```python
{code_candidate}
```

## MANDATORY REQUIREMENTS (MUST FOLLOW ALL)
1. Generate a self-contained test case that PROVES the kernel correctly implements the above task
2. MUST write a complete PyTorch reference implementation that does exactly the same task as the kernel
3. MUST generate random small input tensors (1D <= 4096 elements, 2D <= 64x64)
4. MUST run both the Triton kernel and the PyTorch reference on the same input
5. MUST compare outputs with `torch.allclose(rtol=1e-3, atol=1e-3)` and ASSERT the result is correct
6. MUST verify output shape and dtype matches the expected result
7. DO NOT use broad `try-except` that catches all exceptions - only catch CUDA errors if GPU is not available
8. If the test passes, print "✅ ALL TESTS PASSED"
9. If the test fails, print "❌ TEST FAILED" + error details and raise AssertionError
10. Output ONLY the test code wrapped in <label> tags, NO extra explanations

## PROHIBITED ACTIONS
- ❌ Do NOT skip the reference implementation comparison
- ❌ Do NOT print success without actual assertion
- ❌ Do NOT catch and ignore assertion errors
- ❌ Do NOT use fake/mock results
'''

    # 重试最多2次
    test_code = None
    for test_attempt in range(3):
        if test_attempt > 0:
            print(f"⚠️  Test case generation failed, retrying {test_attempt}/2...")
        import asyncio
        test_result = asyncio.run(bot.run(test_prompt.format(
            task=task,
            annotate_text=annotate_text,
            code_candidate=code_candidate
        )))
        test_code = extract_answer(test_result.content)
        if test_code and len(test_code) > 10:
            break

    return test_code


def annotate_nvidia(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (nvidia GPU).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import requests
    #URL="http://127.0.0.1:2026/v1/completions"
    #
    #data = {
    #    "model": "../Qwen3-4B",
    #    "prompt": input_prompt,
    #    "max_tokens": 10_000, # max_token = 10k
    #}

    URL="http://127.0.0.1:9010/v1/chat/completions"
    data = {
    "messages": [
        {
            "content": input_prompt,
            "role": "system",
            "name": "string"
            }
        ]
    }

    try:
        resp = requests.post(URL, json=data)
        #print(resp.json())
        whole_result = resp.json()["choices"][0]["message"]['content']
        #print(whole_result)
    except Exception as e:
        whole_result = "None"
        print(e)


    prediction = count_answer(whole_result)
    return prediction

def annotate_ascend(input_prompt:str, task_id = -1)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (Huawei Ascend).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    from openai import OpenAI
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:9010/v1/",
    )
    model = "Qwen3-4B-ascend-flagos"

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": input_prompt}
    ]
    prediction = None
    retry_reason = ''
    retries = 0
    while not prediction:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=20_000,
                stream=False,
            )
            whole_result = response.choices[0].message.content
            #print(whole_result)
            prediction = extract_answer(whole_result)
        except Exception as err:
            print(err)
            prediction = None

        # Judge Condition ... 
        if prediction is None:
            retry_reason = 'Extraction failed'
        elif '</think>' in prediction:
            prediction = None
            retry_reason = 'Format Error'
        elif task_id == 8 and 'import' not in prediction:
            prediction = None
            retry_reason = 'Not a python code'
        elif task_id == 3 and  (('[' not in prediction) )  :
            prediction = None
            retry_reason = 'Format error, output should be a string of list such as [1,2,3]'

        # Retry
        if not prediction:
            print(f'Retry. {retry_reason}...')
            retries += 1
            # Last
            if retries > 2:
                return ""
    #print('-----\n'+prediction)
    return prediction


from nanobot import Nanobot

def annotate_nanobot(task: str, annotate_text: str, task_id: int = -1, examples: str = "", rules: str = "", use_tools = True) -> tuple[Nanobot, str]:
    """
        Annotate the unlabeled data using an agent
        :param examples: Reference examples string, format: multiple lines of `# input <label> output </label>`
    """
    prediction = None
    retry_reason = ''
    retries = 0
    suffix = ''
    # Build examples section, show only when not empty
    examples_prompt = f"## Reference Examples\n{examples}\n" if examples.strip() else ""
    feedback_prompt = f"## Previous Attempts and Feedback \n {suffix}" if suffix.strip() else ""
    while not prediction:
        try:
            from nanobot import Nanobot
            bot = Nanobot.from_config(f'{NANOBOT_HOME}/config.json',use_tools = use_tools)
            import asyncio
            result = asyncio.run(bot.run(f'''
# Task
{task}

{examples_prompt}

## Task Specific rules
{rules}

{suffix}

## Text to Annotate
{annotate_text}

            '''))
            prediction = extract_answer(result.content)
        except Exception as err:
            print(err)
            prediction = None

        # Judge Condition ... 
        if prediction is None:
            retry_reason = 'Extraction failed'
        elif '</think>' in prediction:
            prediction = None
            retry_reason = 'Format Error'
        elif task_id == 8 and 'import' not in prediction:
            prediction = None
            retry_reason = 'Not a python code'
        elif task_id == 3 and (('[' not in prediction))  :
            prediction = None
            retry_reason = 'Format error, output should be a string of list such as [1, 2, 3]'
        elif task_id == 2 and len(prediction) > 2:
            prediction = None
            retry_reason = 'You are supposed to output an integer'
        elif task_id == 4 and len(prediction) > 0:
            prediction = prediction.replace(' ','')

        # Retry
        if not prediction:
            print(f'Retry. {retry_reason}...')
            suffix = f'Last time failed because {retry_reason} and your output is {result.content}'
            retries += 1
            # Last
            if retries > 2:
                return bot, ""
    #print('-----\n'+prediction)
    return bot, prediction


def execute_code_safe(code: str, timeout: int = 10) -> tuple[bool, str]:
    """安全执行代码，捕获异常并返回结果"""
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(temp_file)
        )
        success = result.returncode == 0
        full_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

        # ========== 新增：检测静默错误 ==========
        # 即使exit code是0，只要输出包含错误关键词，也标记为失败
        error_keywords = [
            'Traceback', 'ERROR:', 'CUDA error', 'AssertionError',
            'RuntimeError', 'ValueError', 'IndexError', 'TypeError',
            'failed', 'exception', 'error:', 'Invalid argument',
            'out of memory', 'segmentation fault', 'core dumped'
        ]

        if success:
            for kw in error_keywords:
                if kw.lower() in full_output.lower():
                    success = False
                    full_output = f"⚠️  WARNING: Execution returned exit code 0 but found error keyword '{kw}' in output!\n\n" + full_output
                    break

        return success, full_output
    except subprocess.TimeoutExpired:
        return False, f"Execution timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Execution error: {str(e)}"
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass
def _validation_code(code_candidate):
    # 首先只验证导入和基本定义
    validation_code = f"""
import sys
import traceback
try:
    import torch
    import triton
    import triton.language as tl
    print("SUCCESS: Basic imports OK")

    # 尝试编译代码但不执行GPU部分
    import ast
    tree = ast.parse({repr(code_candidate)})
    print("SUCCESS: AST parsing OK")

    # 检查是否有triton.jit装饰器
    has_triton = False
    has_wrapper = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if hasattr(decorator, 'id') and 'triton' in decorator.id.lower():
                    has_triton = True
            if node.name and 'wrapper' in node.name.lower() or node.name and 'kernel' not in node.name.lower():
                has_wrapper = True

    if has_triton:
        print("SUCCESS: Found @triton.jit decorated kernel")
    else:
        print("WARNING: No @triton.jit kernel found")

    print("SUCCESS: Basic validation passed")
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    traceback.print_exc()
"""
    success, output = execute_code_safe(validation_code)
    return success,output


def annotate_nanobot_task8(task: str, annotate_text: str, task_id: int = 8, examples: str = "", rules: str = "",use_tools = False) -> tuple[Nanobot, str]:
    """
    Task8专用标注函数：Triton kernel代码生成，带迭代验证机制
    1. 生成代码
    2. 语法检查/编译检查
    3. 生成测试用例
    4. 执行测试用例
    5. 迭代直到成功
    """
    from nanobot import Nanobot
    import asyncio

    final_code = None
    iteration = 0
    max_iterations = 5
    feedback_history = []
    bot = None

    # 固定部分只构建一次
    examples_prompt = f"## Reference Examples\n{examples}\n" if examples.strip() else ""
    rules_prompt = f"## Task Rules\n{rules}\n" if rules.strip() else ""

    while iteration < max_iterations and final_code is None:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Task8 Code Generation - Iteration {iteration}/{max_iterations}")
        print('='*60)

        try:
            bot = Nanobot.from_config(f'{NANOBOT_HOME}/config.json',use_tools = use_tools)

            # 构建反馈历史
            feedback_section = ""
            if feedback_history:
                feedback_section = "## Previous Attempts and Feedback\n"
                for i, (code, feedback) in enumerate(feedback_history[-2:], 1):
                    feedback_section += f"\n### Attempt {i}:\n```python\n{code[:500]}...\n```\nFeedback: {feedback}\n"

            result = asyncio.run(bot.run(f'''
# Task - Triton Kernel Code Generation
{task}

## Core Requirements
1. Generate complete, runnable Python code with Triton kernels
2. Code must include both the @triton.jit kernel function and a Python wrapper
3. Include necessary imports: import torch, import triton, import triton.language as tl
4. Follow the exact style of the reference examples
5. Ensure all pointers, block sizes, and masks are correctly handled

{rules_prompt}
{examples_prompt}
{feedback_section}

## Text to Annotate
{annotate_text}

## Code Generation Guidelines
- Use proper Triton kernel decorators (@triton.jit)
- Include complete wrapper functions with proper grid setup
- Handle BLOCK_SIZE, P2, masks, and offsets correctly
- Make sure all pointer arithmetic is valid
- Include type hints and proper parameter declarations
            '''))

            code_candidate = extract_answer(result.content)

            if not code_candidate:
                feedback = "Failed to extract code from model output"
                feedback_history.append(("", feedback))
                print(f"❌ {feedback}")
                continue

            print(f"✅ Generated code ({len(code_candidate)} chars)")

            # 步骤1: 语法检查
            print("🔍 Step 1: Syntax check...")
            try:
                compile(code_candidate, '<string>', 'exec')
                print("✅ Syntax check passed")
            except SyntaxError as e:
                feedback = f"Syntax error at line {e.lineno}: {e.msg}"
                feedback_history.append((code_candidate, feedback))
                print(f"❌ {feedback}")
                continue

            # 步骤2: 代码执行验证（检查导入和基本结构）
            print("🔍 Step 2: Basic execution and import check...")
            success,output = _validation_code(code_candidate)
            if success:
                print("✅ Basic validation passed")
            else:
                feedback = f"Validation failed: {output[:500]}"
                feedback_history.append((code_candidate, feedback))
                print(f"❌ {feedback}")
                continue

            # 步骤3: 生成测试用例
            print("🔍 Step 3: Generating test cases...")
            test_code = _generate_triton_test_case(task, annotate_text, code_candidate, bot)
            if test_code and len(test_code) > 10:
                print("✅ Test case generated")

                # 合并代码并运行测试
                full_test_code = code_candidate + "\n\n# ===== TEST CODE =====\n" + test_code
                print("🔍 Step 4: Running test case...")

                test_success, test_output = execute_code_safe(full_test_code, timeout=30)

                if test_success and "✅ ALL TESTS PASSED" in test_output:
                    print(f"✅ Test passed! Output:\n{test_output[:300]}")
                    final_code = code_candidate
                    break
                else:
                    feedback = f"Test failed: {test_output[:500]}"
                    feedback_history.append((code_candidate, feedback))
                    print(f"❌ {feedback}")
                    continue
            else:
                print("❌ Could not generate valid test case after 3 attempts")
                feedback = "Failed to generate test case"
                feedback_history.append((code_candidate, feedback))
                continue

        except Exception as e:
            feedback = f"Exception during generation: {str(e)}"
            feedback_history.append(("", feedback))
            print(f"❌ {feedback}")
            continue

    if final_code:
        print(f"\n✅ SUCCESS! Generated valid code after {iteration} iterations")
        return bot, final_code
    else:
        print(f"\n❌ FAILED! Could not generate valid code after {max_iterations} iterations")
        # 返回最后一次的候选代码（如果有）
        if feedback_history:
            last_code = feedback_history[-1][0]
            if last_code:
                return bot, last_code
        return bot, ""
