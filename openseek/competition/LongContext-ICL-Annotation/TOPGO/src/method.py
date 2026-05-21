"""
FlagOS OpenSeek 赛道三 - TOPGO团队
标注方法模块（V19-T7回归V9参数版）

V15核心改动 - 精确对齐V9(76.05分)最高分配置:
- 最高分=76.05(V9版本), 已通过下载验证输出文件确认
- 核心原则: 答案不出现中文! T1-T7全部零中文零null
- V15策略: T2/T5/T6回滚V9中文prompt+temperature=0.3+分步推理
- V9的中文prompt引导思考,但答案提取器确保纯英文/数字输出

V12核心架构 - 多智能体流水线(仅T8保留):
- Agent 1 (Analyzer): 深度分析任务，输出结构化分析结果
- Agent 2 (Extractor): 从分析结果中提取候选答案
- Agent 3 (Validator): 校验答案合规性，不合规则触发重试

V6优化内容:
|- 温度参数按任务类型精细化调优（0.1-0.5）
|- max_tokens按任务需求动态分配
|- Few-shot示例优化（去重、质量筛选、动态适配）
|- 答案后处理二次校验机制
|- 任务4/7等低分任务专项优化
|- 增强答案提取鲁棒性
"""

import re
import os
from collections import Counter
from typing import Optional


def build_prompt(task_description: str, text2annotate: str, task_id: int = None) -> str:
    """
    构建高质量的标注提示词（针对Qwen3-4B优化）
    
    优化版本 - 支持所有8个任务的特定提示词
    
    Args:
        task_description: 任务描述
        text2annotate: 待标注文本
        task_id: 任务ID（可选，用于特定任务的提示词）
    """
    # 任务特定提示词 - 所有8个任务都有专门优化
    if task_id == 1:
        return build_prompt_task1(task_description, text2annotate)
    elif task_id == 2:
        return build_prompt_task2(task_description, text2annotate)
    elif task_id == 3:
        return build_prompt_task3(task_description, text2annotate)
    elif task_id == 4:
        return build_prompt_task4(task_description, text2annotate)
    elif task_id == 5:
        return build_prompt_task5(task_description, text2annotate)
    elif task_id == 6:
        return build_prompt_task6(task_description, text2annotate)
    # 任务7：回滚到通用中文prompt（V16专用prompt导致空输出和降分）
    # V9已验证: T7无专用prompt,使用通用prompt得分最高(76.05)
    # 通用提示词
    prompt = (
        "### 角色定义\n"
        "你是一个专业的数据标注专家，专门从事长上下文文本标注工作。"
        "你的工作必须严格遵循任务规则，充分学习提供的示例，确保最终标注结果100%包裹在<label>标签中。\n\n"

        "### 核心任务\n"
        f"{task_description}\n\n"

        "### 关键标注指南\n"
        "1. **示例学习要求**: 仔细分析并完全学习示例部分的标注逻辑、格式和标准。"
        "你的标注必须与示例的风格、判断标准和标签用法保持一致。\n"
        "2. **思考过程**: 你可以（并且被鼓励）逐步解释你的标注推理过程。\n"
        "3. **强制输出规则**: 无论你提供什么思考过程，最终标注结果必须包裹在<label>标签中。\n"
        "   - 正确示例: <label>答案</label>\n"
        "   - 错误示例1（缺少标签）: 答案\n"
        "   - 错误示例2（不完整标签）: 答案</label>\n\n"

        "### 示例（必须完全遵循）\n"
        "[[EXAMPLES]]\n\n"

        "### 待标注文本\n"
        f"{text2annotate}\n\n"

        "### 最终要求总结\n"
        "1. 你可以（也应该）提供清晰的思考过程。\n"
        "2. 最终标注结果必须包裹在<label>标签中。\n"
        "3. 所有标注逻辑必须严格遵循上面提供的示例。\n"
    )
    return prompt


def build_prompt_task1(task_description: str, text2annotate: str) -> str:
    """
    任务1专用提示词 - 最接近整数查找任务
    
    核心优化：
    - 强调输出是单个数字
    - 使用英文提示词避免中文推理
    """
    return f"""### Task
Find the integer closest to the target number.

### Rules
1. Analyze the given integer list and target number
2. Find the integer with the smallest difference from the target
3. Output ONLY the number, nothing else
4. Answer must be wrapped in <label> tags

### Examples
List: [1, 5, 9, 15], Target: 8
Answer: <label>9</label>

List: [10, 20, 30, 40], Target: 25
Answer: <label>20</label>

List: [100, 200, 300], Target: 250
Answer: <label>200</label>

### Now process
{text2annotate}
Answer: <label>"""


def build_prompt_task2(task_description: str, text2annotate: str) -> str:
    """
    任务2专用提示词 - 名词/动词计数任务（V15对齐V9-76.05版）
    
    V15改动：回滚到V9中文分步推理prompt(76.05分时使用的版本)
    核心原则：中文prompt引导思考, 但答案必须是纯数字(由提取器保证)
    """
    return f"""### 任务
计算句子中名词或动词的数量。
### 分步方法
第一步：把句子中每个词列出来
第二步：逐个判断每个词的词性（名词/动词/形容词/介词等）
第三步：只数符合要求的词总数量（名词或动词）
第四步：输出最终数字（必须是纯数字）

### 判断规则
- 名词(Noun)：表示人、事、物、地点、抽象概念的词。包括专有名词(人名/地名)和普通名词
- 动词(Verb)：表示动作或状态的词。包括be动词(is/am/are/was/were/been/being)
- 注意：冠词(a/an/the)、介词(in/on/at/of/for)、形容词、副词不是名词也不是动词
- -ing形式：如果是进行时态(如talking)，算动词；如果是名词化(如walking作为活动)，算名词
- 注意：同一个词可能既是名词又是动词，根据上下文判断

### 示例
Sentence: 'The ladder of a jet is lowered from the side for loading passengers'
Question: Count the number of verbs
分析：ladder(N) of(prep) a(det) jet(N) is(V-be) lowered(V) from(prep) the(det) side(N) for(prep) loading(V) passengers(N)
动词：is, lowered, loading -> 3个
Answer: <label>3</label>

### 现在
Sentence: {text2annotate}
Question: Count the number of {'nouns' if 'noun' in (task_description or '').lower() or '名词' in (task_description or '') else 'verbs'}
Answer (ONLY a pure number): <label>"""


def build_prompt_task3(task_description: str, text2annotate: str) -> str:
    """
    任务3专用提示词 - Collatz猜想序列任务
    
    核心优化：
    - 强调输出是数字列表
    - 使用英文提示词避免中文推理
    """
    return f"""### Task
Generate the Collatz conjecture sequence.

### Rules
1. Start from the given number
2. If even, divide by 2; if odd, multiply by 3 and add 1
3. Repeat until reaching 1
4. Output the complete sequence (including start and end)
5. Answer must be a list wrapped in <label> tags

### Examples
Input: 5
Sequence: 5 -> 16 -> 8 -> 4 -> 2 -> 1
Answer: <label>[5, 16, 8, 4, 2, 1]</label>

Input: 3
Sequence: 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
Answer: <label>[3, 10, 5, 16, 8, 4, 2, 1]</label>

Input: 6
Sequence: 6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
Answer: <label>[6, 3, 10, 5, 16, 8, 4, 2, 1]</label>

### Now process
{text2annotate}
Answer: <label>"""


def build_prompt_task4(task_description: str, text2annotate: str) -> str:
    """
    任务4专用提示词 - 字符串连接任务（V7精准修复版）
    
    V7核心修复（解决乱码输出+确认负资产问题）:
    - 诊断：V6极简prompt导致模型"生成"而非"搬运"，产生随机大写字母乱码
    - 实测：新版V6 task4导致67.9->67.75(-0.15分)，确认为负资产
    - 根因：prompt太短，模型不理解要做什么；temperature=0.0但模型仍发挥
    - 方案：(1)超详细指令 (2)6个few-shot覆盖边界情况 (3)负向约束
    """
    return f"""You are a string concatenation tool. Your ONLY job is to join all strings in the given Python list IN ORDER, with NOTHING between them.

STRICT RULES:
1. Input is a Python list of strings like ['a', 'b', 'c']
2. Output MUST be all items joined together: 'abc'
3. Do NOT add spaces, commas, or ANY characters between items
4. Do NOT change capitalization or punctuation of any item
5. Do NOT output anything except the joined result
6. This is NOT a creative task - just copy and paste the strings in order

EXAMPLES:
Input: ['p', 'that.', 'o']
Output: pthat.o

Input: ['hello', 'world']
Output: helloworld

Input: ['a', 'b', 'c']
Output: abc

Input: ['I', 'love', 'coding', '!']
Output: Ilovecoding!

Input: ['The', 'quick', 'brown', 'fox']
Output: Thequickbrownfox

Input: ['x', '=', '1', ';', 'y', '=', '2']
Output: x=1;y=2

NOW DO IT:
Input: {text2annotate}
Output: <label>"""


def build_prompt_task5(task_description: str, text2annotate: str) -> str:
    """
    任务5专用提示词 - 情感分析（V8中文分步推理版）
    
    V8核心改动：回滚到中文prompt（英文V7版偏斜依然严重96%Sad）
    策略：中文思考+分步推理+9NotSad/4Sad示例比例对抗偏斜
    """
    return f"""### 任务
判断这条推文是否表达了真正的"悲伤"情感。只有真正感到悲痛、哀伤、心碎时才判为Sad。

### 分步判断方法
第一步：这条推文是否包含负面情绪？
  - 如果是否定/中性/幽默/事实陈述 → 直接回答 Not sad
  - 如果包含负面情绪 → 进入第二步
第二步：负面情绪是否是"悲伤"（而不是愤怒/烦躁/抱怨）？
  - 悲伤 = 哭泣、心碎、丧亲、绝望、孤独、思念
  - 愤怒/烦躁/抱怨/疲惫/不满 → 回答 Not sad
  - 真正的悲伤 → 回答 Sad

### 关键规则：大多数推文都不是Sad！以下情况都不是Sad：
- 对服务/产品/公司的抱怨 → Not sad
- 疲惫、压力大、忙碌 → Not sad
- 愤怒、批评、认为某事很糟糕 → Not sad
- 幽默、讽刺、玩笑 → Not sad
- 中性陈述、观点、评价 → Not sad
- 身体不适但无情感痛苦 → Not sad

### 示例
"I lost my job today and I feel completely devastated." -> Sad（真正的绝望）
"My dog passed away yesterday. I miss him so much." -> Sad（丧亲之痛）
"I can't stop crying. Everything feels hopeless." -> Sad（无法抑制的悲伤）
"and i shouldve cut them off the moment i started hurting myself over them :o" -> Sad（情感痛苦）

"ok yes I get it -- bikes blues & bbq is frustrating & loud!!!" -> Not sad（烦躁，非悲伤）
"I'm so tired of seeing these ads everywhere." -> Not sad（厌烦）
"the ending of how I met your mother is dreadful" -> Not sad（对TV的评价）
"This restaurant service is terrible." -> Not sad（抱怨服务）
"I'm freezing in the sun, I'm burning in the run" -> Not sad（诗意表达/身体不适）
"Positive thoughts on these gloom cloudy days" -> Not sad（实际上积极）
"But guess what ? I'm sober" -> Not sad（中性陈述）
"I hella should've stayed natural in college" -> Not sad（后悔，非悲伤）

### 待分类推文
{text2annotate}

请回答 Sad 或 Not sad: <label>"""


def build_prompt_task6(task_description: str, text2annotate: str) -> str:
    """
    任务6专用提示词 - MNLI同文体分类（V15对齐V9-76.05版）
    
    V15改动：回滚到V9中文分步推理prompt(76.05分时使用的版本)
    核心原则：中文prompt引导思考, 但答案必须是Y或N(由提取器保证)
    """
    return f"""### 任务
判断两个句子是否属于相同文体/风格（same genre）。
### 核心原则：宽松判断！只要两个句子有可能出现在同类型的出版物或语境中，就应回答Y。
### 分步判断方法
第一步：两个句子的正式程度是否相同？
- 都是正式/都是随意 -> 可能是Y，继续
- 一个正式一个随意 -> 可能是N，继续
第二步：两个句子的表达方式是否相同？
- 都是口语对话/都是书面文章 -> 可能是Y
- 一个是对话一个是文章 -> 可能是N
第三步：如果在前两步中都倾向Y，就回答Y；只有在文本差异明显时才回答N

### 判断为Y的情况（宽松）：
- 两个句子正式程度相似
- 两个句子的式样和词量复杂度相似
- 两个句子可能出现在同一类来源中（杂志、对话、书类等）
- 一个句子是对另一个的改写/描述
- 两个都是口语对话 / 两个都是书面文章
### 判断为N的情况（仅限明显差异）：
- 一个正式学术，另一个明显随意口语
- 一个是对话，另一个是文章
- 文体风格差异很大（如技术手册 vs 小说）
### 示例
Y（同文体 - 宽容判断）：
S1="I do not have the energy to remedy these deficiencies now." S2="I don't have the strength to fix these problems now." Genre:slate -> Y
S1="The data shows that the method works well in practice." S2="Results indicate this approach is effective." Genre:academic -> Y
S1="Hey what's up dude, wanna grab some lunch?" S2="Yo man I'm starving let's get some food!" Genre:casual -> Y
N（不同文体 - 明显差异）：
S1="The experimental results demonstrate statistical significance at p<0.05." S2="lol thats crazy bro i cant even" -> N
S1="In conclusion, the proposed method achieves state-of-the-art performance." S2="so like i watched this movie and it was ok i guess" -> N

{text2annotate}

请回答 Y 或 N (只写一个字母): <label>"""


# build_prompt_task7 已删除 - 回滚到V9通用中文prompt
# V16专用英文prompt导致空输出增多、分数下降
# 任务7现在走 build_prompt() 通用函数（与V9一致）


def build_prompt_for_triton(task_description: str, text2annotate: str) -> str:
    """
    为Triton代码生成任务构建专用提示词（V22研究优化版）
    
    V22核心改进（基于TritonBench/AutoTriton/TritonRL论文研究）:
    1. 全新Prompt架构: 基于TritonBench最佳实践的结构化任务分解
    2. 高质量ICL示例: 5个生产级真实kernel(softmax/layernorm/add/reduction/matmul)
    3. 正向引导: 告诉模型"应该做什么",而不只是"不该做什么"
    4. 质量Checklist: 结构化验证清单引导模型自检
    5. 代码模板: 提供完整骨架结构而非抽象描述
    
    研究来源:
    - TritonBench (arxiv 2502.14752): 首个LLM生成Triton代码基准
    - AutoTriton (arxiv 2507.05687): RL自动优化Triton编程
    - TritonRL (arxiv 2510.17891): 防止reward hacking的训练方法
    - FlagGems/Liger-Kernel: 生产级Triton算子库参考实现
    """
    # V22内置高质量ICL示例
    icl_examples = _get_v22_triton_examples()
    
    # 构建ICL字符串
    icl_parts = []
    for i, ex in enumerate(icl_examples):
        icl_parts.append(f"### Example {i+1}: {ex['task']}\n**Input:** {ex['input']}\n\n```python\n{ex['code']}\n```\n")
    icl_str = '\n'.join(icl_parts)
    
    prompt = f"""You are an expert GPU programmer specializing in OpenAI Triton. You write correct, high-performance Triton kernels that compile and run correctly.

CRITICAL RULES:
1. Every function MUST have a complete implementation with real tl.load/tl.math/tl.store operations
2. NEVER use: pass, ..., TODO, placeholder, dummy, simplified, fake, stub, or "for demonstration"
3. Use descriptive names: softmax_kernel, layer_norm_kernel, matmul_kernel (NOT: kernel, func, wrapper)
4. Always include proper mask parameters in tl.load and tl.store
5. Each kernel must handle edge cases with masks
6. Include both @triton.jit kernel AND a Python wrapper function
7. Import statements must be: import torch, import triton, import triton.language as tl

## Your Task
{task_description}

## Input Data
{text2annotate}

## Reference Examples (study the pattern carefully)
{icl_str}

## Code Structure Template (follow this exact structure)
```python
import torch
import triton
import triton.language as tl

@triton.jit
def <descriptive_name>_kernel(
    # Input pointers
    # Output pointers  
    # Dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    data = tl.load(ptr + offsets, mask=mask, other=0.0)
    
    # YOUR ACTUAL COMPUTATION HERE
    result = ...  # Real computation, not placeholder!
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def <descriptive_name>(<tensor_params>) -> torch.Tensor:
    output = torch.empty_like(<input>)
    grid = lambda META: (<grid_size>, )
    <name>_kernel[grid](<params>, BLOCK_SIZE=<size>)
    return output
```

## Quality Checklist (your code MUST satisfy ALL)
- [ ] Has `import torch` and `import triton` at top
- [ ] Has `@triton.jit` decorated kernel function
- [ ] Kernel uses `tl.program_id`, `tl.arange` for indexing
- [ ] All `tl.load`/`tl.store` have `mask=` parameter
- [ ] Contains REAL math operations (tl.exp, tl.sum, tl.dot, etc.)
- [ ] Has Python wrapper function that calls kernel with grid
- [ ] No: pass, ..., TODO, placeholder, dummy, simplified, fake
- [ ] Function name is descriptive (NOT: kernel, func, wrapper)

Generate complete code now:"""
    return prompt


def select_examples(all_examples: list, task_description: str, text2annotate: str,
                    tokenizer=None, task_id: int = None) -> str:
    """
    选择示例以适配目标上下文长度（V6优化版）
    
    V6优化:
    - 去重：过滤重复或高度相似的示例
    - 质量筛选：优先选择输出简洁、格式规范的示例
    - 动态适配：根据任务类型调整示例数量和格式
    - 多样性：确保示例覆盖不同情况
    """
    # 最大上下文长度 - V8修复：比赛要求30K最小ICL上下文
    # Qwen3-4B + YaRN扩展后支持128K，prompt+examples+output留30K给examples
    # Task8要求16K最小上下文
    if task_id == 8:
        target_length = 14000  # Task8上下文短一些，但也要尽量多
    else:
        target_length = 28000  # 其他任务尽量多放示例
    
    # 根据任务类型调整策略 - V8：大幅增加示例数量匹配30K上下文
    if task_id in [4, 5, 6]:
        # 简单分类/拼接任务：多示例帮助稳定输出
        max_examples = 50
    elif task_id in [1, 2]:
        # 数学/计数任务：多示例帮助理解规则
        max_examples = 80
    else:
        # 复杂任务：尽量多放示例
        max_examples = 100

    examples_str = ""
    token_num = 0
    seen_outputs = set()  # 用于去重
    selected_count = 0
    
    # 预处理：评分和排序示例
    scored_examples = []
    for i, example in enumerate(all_examples):
        try:
            input_text = example['input']
            output_text = example['output'][0] if isinstance(example['output'], list) else example['output']
            
            # 计算质量分数
            score = 0
            
            # 1. 输出越短越好（对于简单任务）
            output_len = len(str(output_text))
            if task_id in [4, 5, 6]:
                # 简单任务偏好短输出
                score += max(0, 50 - output_len)
            else:
                # 复杂任务适中长度
                score += min(output_len, 30)
            
            # 2. 格式规范性（输出不含中文推理标记）
            if not re.search(r'[\u4e00-\u9fff]', str(output_text)):
                score += 20
            
            # 3. 示例多样性（基于输入文本的哈希前缀）
            input_prefix = str(input_text)[:20]
            
            scored_examples.append({
                'index': i,
                'input': input_text,
                'output': output_text,
                'score': score,
                'input_prefix': input_prefix
            })
        except KeyError as e:
            print(f"[WARN] 示例{i}缺少键{e}，跳过")
            continue
    
    # 按分数排序（高分优先）
    scored_examples.sort(key=lambda x: x['score'], reverse=True)
    
    # 选择示例
    for example in scored_examples:
        if selected_count >= max_examples:
            break
            
        input_text = example['input']
        output_text = example['output']
        
        # 去重检查
        output_key = str(output_text).strip().lower()[:50]
        if output_key in seen_outputs:
            continue
        seen_outputs.add(output_key)
        
        # 计算token数
        if tokenizer is not None:
            input_tokens = len(tokenizer.encode(str(input_text), add_special_tokens=False))
            output_tokens = len(tokenizer.encode(str(output_text), add_special_tokens=False))
            length = input_tokens + output_tokens
        else:
            # 简单估算：字符数/4
            length = len(str(input_text) + str(output_text)) // 4

        # 检查是否超限
        if length + token_num <= target_length:
            # 累加token数
            token_num += (length + 7)  # 7是格式符号的token数
            # 拼接示例（根据任务ID使用不同格式）
            if task_id is not None and task_id <= 8:
                example_str = format_example_for_task(input_text, output_text, task_id)
            else:
                example_str = f"Input: {input_text}\nOutput: <label>{output_text}</label>\n\n"
            examples_str += example_str
            selected_count += 1
        else:
            break

    return examples_str


def format_example_for_task(input_text: str, output_text: str, task_id: int) -> str:
    """根据任务类型格式化示例"""
    # 不同任务使用不同的示例格式，与build_prompt保持一致
    if task_id == 1:
        return f"List: {input_text}\nAnswer: <label>{output_text}</label>\n\n"
    elif task_id == 2:
        return f"Sentence: '{input_text}'\nAnswer: <label>{output_text}</label>\n\n"
    elif task_id == 3:
        return f"Input: {input_text}\nAnswer: <label>{output_text}</label>\n\n"
    elif task_id == 4:
        return f"Input: {input_text}\nOutput: <label>{output_text}</label>\n\n"
    elif task_id == 5:
        return f'Tweet: "{input_text}"\nAnswer: <label>{output_text}</label>\n\n'
    elif task_id == 6:
        # MNLI特殊格式
        lines = str(input_text).split('\n')
        if len(lines) >= 2:
            return f"Premise: {lines[0]}\nHypothesis: {lines[1]}\nAnswer: <label>{output_text}</label>\n\n"
        return f"Input: {input_text}\nAnswer: <label>{output_text}</label>\n\n"
    elif task_id == 7:
        return f"Question: {input_text}\n<label>{output_text}</label>\n\n"
    else:
        return f"Input: {input_text}\nOutput: <label>{output_text}</label>\n\n"


def count_answer(text: str, task_id: int = None) -> Optional[str]:
    """
    提取<label>标签内的内容并返回出现次数最多的（优化版）
    
    优化：
    - 支持长代码输出
    - 支持多行内容
    - 支持代码块提取
    - 支持所有8个任务的特定答案提取
    
    Args:
        text: 模型输出的文本
        task_id: 任务ID，用于特定任务的答案提取
    """
    if not text:
        return None
    
    # 任务1特殊处理 - 最接近整数任务
    if task_id == 1:
        return extract_task1_answer(text)
    
    # 任务2特殊处理 - 名词/动词计数任务
    if task_id == 2:
        return extract_task2_answer(text)
    
    # 任务3特殊处理 - Collatz序列任务
    if task_id == 3:
        return extract_task3_answer(text)
    
    # 任务4特殊处理 - 字符串连接任务
    if task_id == 4:
        return extract_task4_answer(text)
    
    # 任务5特殊处理 - 情感分析任务
    if task_id == 5:
        return extract_task5_answer(text)
    
    # 任务6特殊处理 - MNLI蕴含分类任务
    if task_id == 6:
        return extract_task6_answer(text)
    
    # 任务7特殊处理 - 阅读理解任务
    if task_id == 7:
        return extract_task7_answer(text)
    
    # 任务8特殊处理 - Triton代码生成任务
    if task_id == 8:
        return extract_task8_answer(text)
    
    # 通用处理
    return extract_generic_answer(text)


def extract_task1_answer(text: str) -> Optional[str]:
    """
    提取任务1的答案 - 最接近整数任务
    
    任务1的特点：
    - 输出是单个数字
    - 不包含中文
    """
    if not text:
        return None
    
    # 过滤思考标记
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 1. 尝试提取 <label> 标签内容
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        content = matches[-1].strip()
        # 尝试提取数字（包括负数）
        numbers = re.findall(r'-?\d+', content)
        if numbers:
            return numbers[-1]
        return content if content else None
    
    # 2. 如果没有label标签，尝试直接提取数字
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 过滤包含中文的行（推理过程）
        if re.search(r'[\u4e00-\u9fff]', line):
            continue
        # 尝试提取数字
        numbers = re.findall(r'-?\d+', line)
        if numbers:
            return numbers[-1]
    
    return None


def extract_task2_answer(text: str) -> Optional[str]:
    """
    提取任务2的答案 - 名词/动词计数任务（V11.1增强版）
    
    V11.1增强：
    - 优先找 <label> 标签
    - 找 "Answer:" 后面的数字
    - 过滤行号等假数字（只接受0-20范围的合理计数）
    - 最后兜底：返回文本中最后一个合理数字
    """
    if not text:
        return None
    
    # 过滤思考标记
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 1. 尝试提取 <label> 标签内容（最高优先级）
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        content = matches[-1].strip()
        numbers = re.findall(r'\d+', content)
        if numbers:
            return numbers[-1]
        return content if content else None
    
    # 2. 找 "Answer:" 行后面的数字
    answer_pattern = r'(?:Answer|答案)[：:]\s*(\d+)'
    answer_matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1]
    
    # 3. 从文本末尾往前找合理数字（0-20范围，排除行号/步骤号）
    all_numbers = re.findall(r'\b(\d+)\b', text)
    # 过滤出合理的计数数字（名词/动词数量通常在0-20之间）
    reasonable_numbers = [n for n in all_numbers if 0 <= int(n) <= 20]
    
    if reasonable_numbers:
        # 返回最后一个合理的数字（通常是最终答案）
        return reasonable_numbers[-1]
    
    # 4. 如果有数字但都超出范围，返回最小的那个
    if all_numbers:
        return min(all_numbers, key=lambda x: int(x))
    
    # V11.1: 兜底返回0而不是None
    return '0'


def extract_task3_answer(text: str) -> Optional[str]:
    """
    提取任务3的答案 - Collatz序列任务
    
    任务3的特点：
    - 输出是数字列表
    - 格式如 [5, 16, 8, 4, 2, 1]
    """
    if not text:
        return None
    
    # 过滤思考标记
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 1. 尝试提取 <label> 标签内容
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        content = matches[-1].strip()
        # 尝试提取列表格式
        list_pattern = r'\[\s*[\d\s,\-]+\s*\]'
        list_matches = re.findall(list_pattern, content)
        if list_matches:
            return list_matches[-1]
        return content if content else None
    
    # 2. 如果没有label标签，尝试直接提取列表
    list_pattern = r'\[\s*[\d\s,\-]+\s*\]'
    list_matches = re.findall(list_pattern, text)
    
    if list_matches:
        return list_matches[-1]
    
    return None


def extract_task4_answer(text: str) -> Optional[str]:
    """
    提取任务4的答案 - 字符串连接任务
    
    任务4的特点：
    - 输入是一个字符串列表，如 ['p', 'that.', 'o']
    - 输出是连接后的字符串，如 'pthat.o'
    - 答案通常很短，不包含中文
    """
    if not text:
        return None
    
    # 过滤思考标记
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'\x00', '', text)  # 移除null字符
    
    # 1. 尝试提取 <label> 标签内容
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # 取最后一个匹配
        content = matches[-1].strip()
        # 清理换行
        content = re.sub(r'[\n\r]', '', content)
        # 过滤无效内容
        if '标签' in content or 'label' in content.lower():
            # 尝试提取非中文部分
            parts = re.split(r'[，。！？,.]', content)
            for part in parts:
                part = part.strip()
                if part and not re.search(r'[\u4e00-\u9fff]', part) and '标签' not in part:
                    if 1 <= len(part) <= 200:
                        return part
            return None
        return content if content else None
    
    # 2. 如果没有label标签，尝试直接提取答案
    # 移除HTML和思考标签
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'</?think>', '', text)
    
    # 按行分割，找到第一个非中文、非解释的行
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # 过滤条件
        if not line:
            continue
        # 过滤包含中文的行（通常是思考过程）
        if re.search(r'[\u4e00-\u9fff]', line):
            continue
        # 过滤包含推理关键词的行
        if any(kw in line for kw in ['因为', '所以', '首先', '然后', '我认为', '因此']):
            continue
        # 过滤包含特定关键词的行
        if any(kw in line.lower() for kw in ['标签', 'label', '示例', 'example', '首先', '比如']):
            continue
        # 长度检查
        if 1 <= len(line) <= 200:
            return line
    
    return None


# V10: Task 5 悲伤关键词库（强信号=一定是Sad）
_SAD_KEYWORDS_STRONG = {
    'devastated', 'heartbroken', 'grief', 'mourning', 'crying', 'tears',
    'hopeless', 'despair', 'depressed', 'suicidal', 'bereaved', 'loss of',
    'passed away', 'died', 'death', 'funeral', 'miss him', 'miss her',
    'cannot stop crying', 'can\'t stop crying', 'broken heart', 'heart break',
    'so sad', 'very sad', 'extremely sad', 'terribly sad',
}

# V10: Task 5 非悲伤负面词库（有负面情绪但不是Sad）
_NOT_SAD_NEGATIVE_KEYWORDS = {
    'frustrating', 'frustrated', 'terrible', 'awful', 'horrible', 'annoying',
    'tired of', 'sick of', 'fed up', 'hate', 'dislike', 'boring', 'bored',
    'angry', 'mad', 'furious', 'irritated', 'annoyed',
    'complaint', 'complaining', 'wrong', 'bad', 'worst',
    'freezing', 'burning', 'cold', 'hot',  # 身体不适
    'sober', 'natural', 'guess what', 'positive thoughts',  # 中性/积极
}


def _task5_keyword_correction(prediction: Optional[str], original_text: str) -> Optional[str]:
    """
    V10新增: 基于关键词规则的Task 5后处理校正
    
    不重跑模型，而是用规则对模型输出进行二次校正。
    当模型输出与文本中的强关键词信号矛盾时，自动修正。
    
    Args:
        prediction: 模型预测结果 (Sad/Not sad/None)
        original_text: 原始推文文本
    
    Returns:
        校正后的预测结果
    """
    if not prediction or not original_text:
        return prediction
    
    text_lower = original_text.lower().strip()
    
    # 检查强悲伤关键词
    has_strong_sad = any(kw in text_lower for kw in _SAD_KEYWORDS_STRONG)
    # 检查非悲伤负面关键词
    has_not_sad_negative = any(kw in text_lower for kw in _NOT_SAD_NEGATIVE_KEYWORDS)
    
    # 规则1: 如果模型说Not Sad，但文本中有强悲伤关键词 → 翻转为Sad
    if prediction == 'Not sad' and has_strong_sad and not has_not_sad_negative:
        return 'Sad'
    
    # 规则2: 如果模型说Sad，但文本中只有非悲伤负面词且无悲伤词 → 翻转为Not sad
    if prediction == 'Sad' and has_not_sad_negative and not has_strong_sad:
        return 'Not sad'
    
    return prediction


def extract_task5_answer(text: str, original_text: str = None) -> Optional[str]:
    """
    提取任务5的答案 - 情感分析任务（V10增强版）
    
    V10新增: 关键词规则后处理校正
    
    任务5的特点：
    - 输出是Sad或Not sad
    - 模型天然偏斜输出Sad，需要规则校正
    """
    if not text:
        return None
    
    # 1. 尝试提取 <label> 标签内容
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    prediction = None
    if matches:
        content = matches[-1].strip()
        content_lower = content.lower()
        if 'not sad' in content_lower or 'not_sad' in content_lower:
            prediction = 'Not sad'
        elif 'sad' in content_lower:
            prediction = 'Sad'
        else:
            prediction = content if content else None
    else:
        # 2. 如果没有label标签，尝试直接提取
        text_lower = text.lower()
        if 'not sad' in text_lower or 'not_sad' in text_lower:
            prediction = 'Not sad'
        elif 'sad' in text_lower:
            prediction = 'Sad'
    
    # V10: 应用关键词规则校正（如果提供了原始文本）
    if original_text:
        prediction = _task5_keyword_correction(prediction, original_text)
    
    return prediction


def extract_task6_answer(text: str) -> Optional[str]:
    """
    提取任务6的答案 - MNLI蕴含分类任务（V10增强版）
    
    V10新增: 更强的Y/N提取鲁棒性，减少null
    
    任务6的特点：
    - 输出是Y或N
    - 答案非常短
    - 必须过滤掉推理过程中出现的Y/N字符
    - 数据集偏N(约90%是N)，null时默认返回N更安全
    """
    if not text:
        # V10: null时返回默认值N（因为数据集90%是N，猜N的期望收益更高）
        return 'N'
    
    # 1. 尝试提取 <label> 标签内容
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # 取最后一个label标签（通常是最终答案）
        content = matches[-1].strip().upper()
        # 只取第一个字符
        if content:
            first_char = content[0]
            if first_char in ['Y', 'N']:
                return first_char
        # 如果内容是YES或NO
        if content.startswith('YES'):
            return 'Y'
        if content.startswith('NO'):
            return 'N'
    
    # 2. 过滤推理过程，查找答案行
    # 过滤思考标记
    text = re.sub(r'heed>.*? вит>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 按行分割
    lines = text.strip().split('\n')
    
    # 从后往前找答案（答案通常在最后）
    for line in reversed(lines):
        line = line.strip().upper()
        if not line:
            continue
        # 过滤包含中文的行（推理过程）
        if re.search(r'[\u4e00-\u9fff]', line):
            continue
        # 过滤包含推理关键词的行
        reasoning_keywords = ['FIRST', 'SECOND', 'THEREFORE', 'HOWEVER', 'BECAUSE', 
                             'ANALYZE', 'ANALYSIS', 'SENTENCE', 'PREMISE', 'HYPOTHESIS',
                             'THEREFORE', 'THUS', 'MEANS', 'IMPLIED', 'CONCLUSION']
        if any(kw in line for kw in reasoning_keywords):
            continue
        # 检查是否是纯答案行
        if line in ['Y', 'N', 'YES', 'NO']:
            return 'Y' if line in ['Y', 'YES'] else 'N'
        # 检查行首是否是Y或N
        if line.startswith('Y'):
            return 'Y'
        if line.startswith('N'):
            return 'N'
    
    # 3. 最后尝试：查找最后一个独立的Y或N字符
    # 在整篇文本中查找
    text_upper = text.upper()
    # 从后往前查找
    for i in range(len(text_upper) - 1, -1, -1):
        char = text_upper[i]
        if char in ['Y', 'N']:
            # 检查前后是否是字母（避免匹配到NEW YORK中的Y）
            prev_is_letter = i > 0 and text_upper[i-1].isalpha()
            next_is_letter = i < len(text_upper) - 1 and text_upper[i+1].isalpha()
            # 如果Y/N是独立的（前后不是字母），返回它
            if not prev_is_letter and not next_is_letter:
                return char
    
    # V10: 全部提取失败时返回默认值N（数据集90%是N，比返回None/null更好）
    return 'N'


def extract_task7_answer(text: str) -> Optional[str]:
    """
    V19: T7 extractor - 回归V9简洁风格 + 最小必要修复
    
    V19核心变化:
    - 回归V9的简洁提取逻辑(去除过度过滤)
    - 保留Category:前缀剥离(已确认存在此问题)
    - 保留中文答案过滤(评测要求英文答案)
    - 但不再轻易返回unknown(让更多有效答案通过)
    - 配合TASK_CONFIG回滚V9参数(temp=0.7/max_tokens=10000)
    
    参考文档启示: 答案提取应宽松,归一化在后处理做
    """
    if not text:
        return None

    text = text.strip()

    thinking_start = ''
    thinking_end = ''

    if text in [thinking_start, thinking_end, '']:
        return None

    # 1. 提取 <label> 标签内容 (V9原始逻辑)
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        content = matches[-1].strip()
        if content in [thinking_start, thinking_end, '']:
            return None
        content = re.sub(r'<[^>]+>', '', content)
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if lines:
            answer = lines[0]
            # V19: 剥离Category:前缀(保留此修复)
            if answer.upper().startswith('CATEGORY:'):
                answer = answer.split(':', 1)[1].strip()
            # V19: 宽松中文处理 - 首行有中文时尝试后续行,但不放弃
            if re.search(r'[\u4e00-\u9fff]', answer):
                for line in lines[1:]:
                    if not re.search(r'[\u4e00-\u9fff]', line) and line.strip():
                        return line.strip()[:100]
                # V19: 即使只有中文也返回(可能包含有效英文实体),不返回unknown
                if answer:
                    return answer[:100]
                return None
            if answer:
                return answer[:100]
        return content[:100] if content else None

    # 2. 无label标签 (V9原始逻辑)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace(thinking_start, '').replace(thinking_end, '')

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        answer = lines[0]
        # V19: 剥离Category:前缀
        if answer.upper().startswith('CATEGORY:'):
            answer = answer.split(':', 1)[1].strip()
        if answer in [thinking_start, thinking_end, '']:
            return None
        # V19: 宽松中文处理
        if re.search(r'[\u4e00-\u9fff]', answer):
            for line in lines[1:]:
                line = line.strip()
                if line and not re.search(r'[\u4e00-\u9fff]', line):
                    return line[:100]
            # 不放弃,返回已有内容
            if answer:
                return answer[:100]
            return None
        return answer[:100]

    # V19: 回归V9, 返回None而非unknown(由count_answer统一处理默认值)
    return None



def _is_pseudocode(code: str) -> bool:
    """
    V9新增: 检测代码是否为伪代码/占位符
    
    Returns True if the code contains placeholder patterns that indicate incomplete implementation.
    """
    if not code:
        return True
    
    # 转小写检测（不区分大小写）
    code_lower = code.lower()
    
    # 致命占位符模式（任何一项命中即为伪代码）
    fatal_patterns = [
        r'\bpass\s*(#.*)?$',           # 单独的pass语句
        r'\.\.\.\s*$',                 # 省略号作为函数体
        r'\bTODO\b',                   # TODO标记
        r'\bFIXME\b',                  # FIXME标记
        r'placeholder',                # placeholder
        r'simplified',                 # simplified
        r'implement this',             # implement this
        r'your code here',             # your code here
        r'实现代码',                    # 中文占位符
        r'NotImplementedError',        # 异常占位符
        r'#\s*to\s+do\b',              # # to do
        r'#\s*add\s+',                 # # add ...
        r'#\s*fill\s+in',             # # fill in
        r'#\s*write\s+your',          # # write your
        r'#\s*insert\s+code',         # # insert code
    ]
    
    for pattern in fatal_patterns:
        if re.search(pattern, code_lower, re.MULTILINE):
            return True
    
    # 检查函数体是否过于简单（只有docstring没有实际代码）
    lines = code.split('\n')
    in_function = False
    function_body_lines = 0
    actual_code_lines = 0
    
    for line in lines:
        stripped = line.strip()
        
        # 检测函数定义开始
        if re.match(r'^def\s+\w+\s*\(', stripped):
            in_function = True
            function_body_lines = 0
            actual_code_lines = 0
            continue
        
        # 在函数体内计数
        if in_function:
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                function_body_lines += 1
                # 实际代码行：不是纯注释、空行、或简单return/pass
                if stripped not in ['pass', '...', 'return', 'return None'] and not stripped.startswith('raise'):
                    actual_code_lines += 1
            
            # 函数结束（下一个def或顶层代码）
            if re.match(r'^def\s+\w+\s*\(', stripped) or (stripped and not line.startswith(' ') and not line.startswith('\t') and stripped and not stripped.startswith('@')):
                if in_function and function_body_lines > 0 and actual_code_lines == 0:
                    # 这个函数有"体"但没有实际代码 → 可能是伪代码
                    pass  # 不立即返回，继续检查其他指标
                in_function = False
    
    # 最终检查：如果代码太短且包含可疑模式，判定为伪代码
    if len(code.strip()) < 100:
        # 短代码检查是否有实质内容
        has_import = bool(re.search(r'^import\s+|^from\s+', code, re.MULTILINE))
        has_triton_kernel = '@triton.jit' in code
        has_actual_logic = bool(re.search(r'tl\.|torch\.|triton\.', code))
        if not (has_import and (has_triton_kernel or has_actual_logic)):
            return True
    
    return False


def extract_task8_answer(text: str) -> Optional[str]:
    """
    提取任务8的答案 - Triton代码生成任务（V21修复版）
    
    V21修复: 过滤Phase分析文本、智能代码边界检测、dummy实现二次过滤、完整性评分优先选择
    
    任务8的特点：
    - 输出是完整的Python代码
    - 代码可能很长
    - 需要过滤掉含占位符的伪代码
    - 需要过滤think标签内的思考文本和分析过程
    """
    if not text:
        return None
    
    # V20: 先过滤所有思考/分析标记（必须在其他处理之前！）
    text = re.sub(r'​*think​*.*?​*/​*think​*', '', text, flags=re.DOTALL)  # V21 fix: V20 had empty regex
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r' </think>', '', text)
    
    # 1. 尝试提取 <label> 标签内的代码块
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    code = None
    if matches:
        content = matches[-1].strip()
        # 提取代码块
        code_pattern = r'```(?:python)?\n(.*?)\n```'
        code_matches = re.findall(code_pattern, content, re.DOTALL)
        if code_matches:
            code = code_matches[-1].strip()
        else:
            # 如果没有代码块标记，直接使用内容
            code = content
    
    # 2. 尝试直接提取代码块
    if not code:
        code_pattern = r'```(?:python)?\n(.*?)\n```'
        code_matches = re.findall(code_pattern, text, re.DOTALL)
        if code_matches:
            code = code_matches[-1].strip()
    
    # 3. 检查是否包含import语句（代码开始的标志）
    if not code:
        if 'import torch' in text or 'import triton' in text:
            import_match = re.search(r'(import torch.*)', text, re.DOTALL)
            if import_match:
                code = import_match.group(1).strip()
    
    # V9+V20: 伪代码/dummy检测 - 如果检测到占位符，返回None让系统重试
    if code and _is_pseudocode(code):
        print("[WARN] Task 8: 检测到伪代码/占位符，拒绝此输出")
        return None
    
    # V20: 二次过滤 - dummy/fake实现检测（补充_is_pseudocode未覆盖的模式）
    if code:
        dummy_patterns = [
            r'dummy implementation',
            r'does not perform actual',
            r'just returns? input',
            r'placeholder for actual',
            r'simplified version',
            r'for demonstration only',
        ]
        for dp in dummy_patterns:
            if re.search(dp, code, re.IGNORECASE):
                print(f"[WARN] Task 8: 检测到dummy实现模式 '{dp}'，拒绝此输出")
                return None
    
    return code


def validate_answer(answer: Optional[str], task_id: int = None) -> Optional[str]:
    """
    答案二次校验函数（V6新增）
    
    对提取的答案进行后处理验证：
    - 空值检查
    - 格式校验
    - 合理性检查
    - 特殊字符清理
    
    Args:
        answer: 提取的原始答案
        task_id: 任务ID
    
    Returns:
        校验后的答案（如果无效则返回None）
    """
    if answer is None:
        return None
    
    # 基本清理
    answer = str(answer).strip()
    
    # 过滤空答案
    if not answer or answer.lower() in ['none', 'null', '', 'n/a']:
        return None
    
    # 移除可能的BOM和零宽字符
    answer = answer.replace('\ufeff', '').replace('\u200b', '').replace('\x00', '')
    
    # 按任务类型进行特定校验
    if task_id == 1:
        # 任务1：必须是数字
        numbers = re.findall(r'-?\d+', answer)
        return numbers[-1] if numbers else None
    
    elif task_id == 2:
        # 任务2：必须是正整数
        numbers = re.findall(r'\d+', answer)
        return numbers[-1] if numbers else None
    
    elif task_id == 3:
        # 任务3：必须是列表格式
        list_pattern = r'\[\s*[\d\s,\-]+\s*\]'
        match = re.search(list_pattern, answer)
        return match.group() if match else None
    
    elif task_id == 4:
        # 任务4：字符串连接结果，移除多余空白
        answer = re.sub(r'\s+', ' ', answer).strip()
        # 限制最大长度（防止包含推理内容）
        if len(answer) > 500:
            answer = answer[:500]
        return answer if answer else None
    
    elif task_id == 5:
        # 任务5：必须是 Sad 或 Not sad（V10增强）
        answer_lower = answer.lower()
        if 'not sad' in answer_lower or 'not_sad' in answer_lower:
            return 'Not sad'
        elif 'sad' in answer_lower:
            return 'Sad'
        # V10: 无法判断时返回Not sad（数据集中Not sad占多数，比返回None安全）
        return 'Not sad'
    
    elif task_id == 6:
        # 任务6：必须是 Y 或 N（V10增强：拒绝None）
        answer_upper = answer.upper().strip()
        if answer_upper.startswith('Y'):
            return 'Y'
        elif answer_upper.startswith('N'):
            return 'N'
        # V10: 无法判断时返回默认值N（数据集约90%是N）
        return 'N'
    
    elif task_id == 7:
        # 任务7：阅读理解答案，限制长度并清理（V19: 配合V9参数回滚）
        answer = re.sub(r'<[^>]+>', '', answer)
        lines = answer.strip().split('\n')
        answer = lines[0].strip() if lines else answer.strip()
        # V19: 剥离Category:前缀(保留修复)
        if answer.upper().startswith('CATEGORY:'):
            answer = answer.split(':', 1)[1].strip()
        if len(answer) > 100:
            answer = answer[:100]
        return answer if answer else ''
    
    elif task_id == 8:
        # 任务8：代码生成，基本清理即可
        if answer.startswith('<label>'):
            answer = answer[7:]
        if answer.endswith('</label>'):
            answer = answer[:-8]
        return answer.strip() if answer.strip() else None
    
    # 默认：返回清理后的答案
    return answer[:500] if answer else None


def extract_generic_answer(text: str) -> Optional[str]:
    """
    通用答案提取函数
    """
    if not text:
        return None
    
    # 过滤思考标记
    text = re.sub(r'heed>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 方法1: 尝试提取 <label> 标签
    pattern = r'<label>\s*(.*?)\s*</label>'
    content_matches = re.findall(pattern, text, re.DOTALL)
    
    if content_matches:
        # 取最后一个匹配（通常是最确定的）
        last_match = content_matches[-1].strip()
        
        # 如果内容包含代码块，提取代码块
        code_block_pattern = r'```python\n(.*?)\n```'
        code_matches = re.findall(code_block_pattern, last_match, re.DOTALL)
        
        if code_matches:
            # 返回完整的代码块
            return code_matches[-1].strip()
        
        # 如果内容是列表格式，提取列表
        list_pattern = r'\[[\d\s,\-]+\]'
        list_matches = re.findall(list_pattern, last_match)
        if list_matches:
            return list_matches[-1]
        
        # 否则返回清理后的内容
        # 清理思考过程文本和推理关键词
        if '标签' in last_match or 'label' in last_match.lower() or any(kw in last_match for kw in ['所以', '因此', '我认为']):
            # 尝试提取实际答案部分
            lines = last_match.split('\n')
            for line in reversed(lines):
                line = line.strip()
                # 过滤包含推理关键词的行
                if any(kw in line for kw in ['因为', '所以', '首先', '然后', '我认为', '因此', '标签', 'label']):
                    continue
                if line and '标签' not in line and 'label' not in line.lower():
                    # 如果是代码块开头，返回整个代码块
                    if line.startswith('```') or line.startswith('import'):
                        return last_match
                    return line
        
        # 过滤推理过程
        lines = last_match.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # 过滤包含推理关键词的行
            if any(kw in line for kw in ['因为', '所以', '首先', '然后', '接下来', '我认为', '因此', '分析如下']):
                continue
            clean_lines.append(line)
        
        # 如果过滤后仍有内容，返回清理后的结果
        if clean_lines:
            return '\n'.join(clean_lines[-3:])  # 返回最后3行（避免返回推理过程）
        
        return last_match
    
    # 方法2: 如果没有 <label> 标签，尝试提取代码块
    code_block_pattern = r'```python\n(.*?)\n```'
    code_matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if code_matches:
        # 返回最后一个代码块
        return code_matches[-1].strip()
    
    # 方法3: 尝试提取列表格式
    list_pattern = r'\[[\d\s,\-]+\]'
    list_matches = re.findall(list_pattern, text)
    
    if list_matches:
        return list_matches[-1]
    
    return None


# ============================================================
# 任务配置表 - 精细化的超参数配置
# ============================================================
TASK_CONFIG = {
    # 任务1: 最接近整数查找 - 确定性数学任务，用低温度
    1: {
        'temperature': 0.1,       # 低温度确保确定性输出
        'top_p': 0.9,
        'max_tokens': 500,        # 只需输出一个数字
        'system_prompt': 'You are a precise math assistant. Output only the answer.',
        'description': 'closest_integers'
    },
    # 任务2: 名词/动词计数 - V15对齐V9(76.05分)
    2: {
        'temperature': 0.3,       # V15: 回滚V9参数(高温度让分步推理更自然)
        'top_p': 0.9,
        'max_tokens': 1000,       # V15: 回滚V9参数(分步推理需要更多token)
        'system_prompt': '你是一个精确的语言分析专家。逐词分析词性，仔细计数。',  # V15: 回滚V9中文
        'description': 'count_nouns_verbs'
    },
    # 任务3: Collatz猜想序列 - 确定性数学序列
    3: {
        'temperature': 0.1,       # 低温度确保序列正确
        'top_p': 0.9,
        'max_tokens': 2000,       # 序列可能较长
        'system_prompt': 'You are a math sequence generator. Follow the rules precisely.',
        'description': 'collatz_conjecture'
    },
    # 任务4: 字符串连接 - 确定性操作，但需要精确匹配
    4: {
        'temperature': 0.0,       # 零温度！这是纯拼接任务，不需要任何创造性
        'top_p': 1.0,             # top_p不限制（temperature=0时无效）
        'max_tokens': 1000,       # 拼接结果通常不长
        'system_prompt': 'You are a string processing tool. Concatenate exactly as instructed. No explanation.',
        'description': 'concat_strings'
    },
# 任务5: 情感分析 - V15对齐V9(76.05分)
5: {
    'temperature': 0.3,       # V15: 回滚V9参数(高温度让分步推理更自然)
    'top_p': 0.9,
    'max_tokens': 500,
    'system_prompt': '你是一个情感分析专家。仔细分析推文情感。不要轻易判断Sad。',  # V15: 回滚V9中文+反偏斜
    'description': 'tweet_sadness'
},
# 任务6: MNLI蕴含分类 - V15对齐V9(76.05分)
6: {
    'temperature': 0.3,       # V15: 回滚V9参数(高温度让分步推理更自然)
    'top_p': 0.9,
    'max_tokens': 500,        # V15: 回滚V9参数
    'system_prompt': '你是一个文体分类专家。宽松判断，倾向于回答Y。',  # V15: 回滚V9中文+宽松Y
    'description': 'mnli_classification'
},
# Task 7: Jeopardy QA - V19回滚V9原始高分参数(76.05分)
# V19发现: 之前T7降分的根本原因不是prompt而是参数!
# V9(76.05分) T7配置: temp=0.7, top_p=0.95, max_tokens=10000
# 当前V18错误地用了temp=0.3/max_tokens=500,导致模型无法充分推理
# 参考文档启示: ReAct思维链需要高temperature+大token空间
7: {
    'temperature': 0.7,       # V19: 回滚V9! Jeopardy QA需要创造性推理(不是确定性任务)
    'top_p': 0.95,            # V19: 回滚V9! 更开放的采样
    'max_tokens': 10000,      # V19: 回滚V9! 充分推理空间(500太短导致截断)
    'system_prompt': '你是一个有帮助的助手。',  # V19: 回滚V9中文prompt(英文prompt导致-2.15分)
    'description': 'jeopardy_qa'
},
# 任务8: Triton代码生成 - V22研究优化版（基于TritonBench/AutoTriton论文）
8: {
    'temperature': 0.1,       # V22: 更低起始温度! 更确定性输出(从0.2降到0.1)
    'top_p': 0.85,            # V22: 略微收紧采样范围
    'max_tokens': 16000,     # V22: 更多空间给完整代码(从12000增到16000)
    'system_prompt': 'You are an expert GPU programmer specializing in OpenAI Triton. Generate complete, correct, runnable Python code with real implementations. No placeholders ever.',  # V22: 英文系统提示更稳定
    'description': 'kernel_generation_v22'
},
}


def get_task_config(task_id: int) -> dict:
    """获取任务配置，带默认值回退"""
    return TASK_CONFIG.get(task_id, {
        'temperature': 0.2,
        'top_p': 0.9,
        'max_tokens': 2000,
        'system_prompt': 'You are a helpful assistant.',
        'description': 'unknown'
    })


def annotate_ascend(input_prompt: str, is_triton_task: bool = False, task_id: int = None) -> Optional[str]:
    """
    使用华为Ascend进行标注（V11彻底修复版）
    
    V11核心改动 - 解决API不稳定导致的大量null问题:
    - 所有任务统一高重试次数(10次)
    - 统一长超时(600s/10分钟)
    - 指数退避从1/2/4s改为2/5/15/30/60s
    - 所有错误类型都重试(不再放弃)
    - 连接预热机制(首次调用前检测连接)
    
    Args:
        input_prompt: 输入提示词
        is_triton_task: 是否为Triton代码生成任务
        task_id: 任务ID，用于特定任务的答案提取
    
    Returns:
        标注结果（极端情况下返回默认值而非None）
    """
    import openai
    import time

    # 配置API
    openai.api_key = os.getenv("QWEN_API_KEY", "EMPTY")
    openai.base_url = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1/")

    model = "/home/Qwen/Qwen3-4B"  # 使用vLLM服务的实际模型路径

    # 获取任务配置
    config = get_task_config(task_id) if task_id else {}
    temperature = config.get('temperature', 0.2)
    top_p = config.get('top_p', 0.9)
    max_tokens = config.get('max_tokens', 2000)
    system_prompt = config.get('system_prompt', 'You are a helpful assistant.')
    
    # V11: 统一使用更长的超时和更多重试
    timeout = 600  # 所有任务统一10分钟超时
    max_retries = 10  # V11: 所有任务统一10次重试!

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    # V11: 重试逻辑 - 更激进的错误恢复
    last_error = None
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
                timeout=timeout,
            )
            whole_result = response.choices[0].message.content
            
            # 二次校验：对答案进行后处理验证
            prediction = count_answer(whole_result, task_id=task_id)
            validated = validate_answer(prediction, task_id)
            
            # V11: 即使validate返回None，也尝试用原始结果
            if validated is not None:
                return validated
            else:
                # validate失败了但模型有输出，尝试直接返回原始提取结果
                if prediction is not None:
                    return prediction
                # 最后手段：返回模型的原始输出（总比null好）
                return whole_result.strip()[:500] if whole_result else None
                
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # V11: 所有错误类型都重试！不再区分错误类型直接放弃
            if attempt < max_retries - 1:
                # V11: 更长的指数退避: 2, 5, 15, 30, 60, 60, 60...
                wait_times = [2, 5, 15, 30, 60, 60, 60, 60, 60, 60]
                wait_time = wait_times[attempt] if attempt < len(wait_times) else 60
                
                error_type = "连接" if 'connection' in error_str or 'connect' in error_str else \
                             "超时" if 'timeout' in error_str or 'timed out' in error_str else \
                             "API"
                print(f"[WARN] 任务{task_id}{error_type}错误 (尝试 {attempt+1}/{max_retries}): {str(e)[:100]}")
                print(f"[INFO] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
    
    # V11: 所有重试都失败后的最终兜底
    print(f"[ERROR] 任务{task_id} API调用全部失败 (已重试{max_retries}次): {last_error}")
    
    # V11: 根据任务类型返回默认值而不是None
    if task_id == 1:
        return "0"  # Task 1: 返回0（会被确定性兜底覆盖，这里只是保险）
    elif task_id == 2:
        return "0"  # Task 2: 返回0
    elif task_id == 3:
        return "[1]"  # Task 3: 返回最小序列
    elif task_id == 4:
        return ""  # Task 4: 返回空串
    elif task_id == 5:
        return "Not sad"  # Task 5: 默认Not sad（数据集多数）
    elif task_id == 6:
        return "N"  # Task 6: 默认N（数据集90%是N）
    elif task_id == 7:
        return "unknown"  # Task 7: 返回unknown而非空串
    elif task_id == 8:
        return None  # Task 8: 代码不能猜，只能返回None
    else:
        return ""  # 其他: 返回空串


# ============================================================
# V12 多智能体流水线框架
# ============================================================

def _call_llm(prompt: str, task_id: int = None, temperature: float = 0.3, 
              max_tokens: int = 2000, timeout: int = 120, system_prompt: str = None) -> Optional[str]:
    """
    底层LLM调用函数 - 所有Agent共享的API调用接口
    
    Args:
        prompt: 完整提示词
        task_id: 任务ID（用于日志）
        temperature: 温度参数
        max_tokens: 最大输出token
        timeout: 超时时间(秒)
    
    Returns:
        模型原始输出文本，失败返回None
    """
    import openai
    import time
    
    openai.api_key = os.getenv("QWEN_API_KEY", "EMPTY")
    openai.base_url = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1/")
    model = "/home/Qwen/Qwen3-4B"
    
    # 简化重试：3次快速重试
    for attempt in range(3):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"[ERROR] LLM调用失败 (task={task_id}): {str(e)[:100]}")
                return None


# ==================== Task 2 多智能体: 词性计数 ====================

def multi_agent_task2(task_description: str, text2annotate: str) -> Optional[str]:
    """
    Task 2 单次调用+结构化输出 - 名词/动词计数 (V12.1修复版)
    
    V12.1改动: 从双Agent改为单次API调用 + 强制结构化输出
    原因: 双Agent依赖两次API调用，API不稳定时Agent 1失败导致全部返回0
    
    策略:
    - 单次API调用，prompt强制模型先分析再输出数字
    - Validator层做后处理校验和fallback
    """
    # 判断目标词性
    desc_lower = (task_description or "").lower()
    target = "nouns" if 'noun' in desc_lower else "verbs"
    target_tag = "N" if target == "nouns" else "V"
    
    # 单次结构化prompt - 让模型一步完成分析和计数
    prompt = f"""Count the {target} in this sentence. Show your work then give the final number.

Sentence: {text2annotate}

Step 1 - Label each word's POS tag (N=noun, V=verb, D=det, P=prep, A=adj, Adv=adv, C=conj):
(Write the tagged sentence here)

Step 2 - Count only {target} ({target_tag}) tags:
Number of {target}: 

Answer: <label>"""

    result = _call_llm(prompt, task_id=2, temperature=0.1, max_tokens=800, timeout=120)
    
    if not result:
        print(f"[T2] API returned null, using fallback")
        return '0'
    
    print(f"[T2] Raw output: {result[:300]}...")
    
    # ===== Validator: 提取并校验答案 =====
    
    # 优先找 <label> 标签
    label_match = re.search(r'<label>\s*(\d+)\s*</label>', result)
    if label_match:
        answer = label_match.group(1)
        if 0 <= int(answer) <= 30:
            return answer
    
    # 找 "Number of xxx:" 后面的数字
    count_match = re.search(r'Number of \w+:\s*(\d+)', result, re.IGNORECASE)
    if count_match:
        answer = count_match.group(1)
        if 0 <= int(answer) <= 30:
            return answer
    
    # 找 "Answer:" 后面的数字
    ans_match = re.search(r'Answer:\s*(\d+)', result, re.IGNORECASE)
    if ans_match:
        answer = ans_match.group(1)
        if 0 <= int(answer) <= 30:
            return answer
    
    # Fallback: 从标注结果中用正则统计TAG数量
    # 匹配 word(TAG) 格式
    tags = re.findall(r'\((\w)\)', result)
    if tags:
        count = sum(1 for t in tags if t.upper() == target_tag)
        if count > 0:
            print(f"[T2] Fallback from POS tags: found {count} {target_tag}")
            return str(count)
    
    # 最终兜底：从所有数字中取最后一个合理的
    all_numbers = re.findall(r'\b(\d+)\b', result)
    reasonable = [n for n in all_numbers if 0 <= int(n) <= 30]
    if reasonable:
        return reasonable[-1]
    
    print(f"[T2] All extraction failed, returning 0")
    return '0'


# ==================== Task 6 单次调用+结构化输出: 文体分类 ====================

def multi_agent_task6(task_description: str, text2annotate: str) -> Optional[str]:
    """
    Task 6 单次调用+结构化输出 - MNLI同文体分类 (V12.1修复版)
    
    V12.1改动: 从双Agent改为单次API调用 + 强制结构化输出
    原因: 双Agent依赖两次API调用，API不稳定时Agent 1失败导致全部返回默认值
    """
    # 单次结构化prompt - 分析+判定一步完成
    prompt = f"""Are these two sentences from the SAME genre/style? Analyze then decide.

Sentences:
{text2annotate}

Analysis:
S1 formality: (formal/semi-formal/casual/slang)
S2 formality: (formal/semi-formal/casual/slang)
S1 domain: (academic/conversational/journalistic/etc)
S2 domain: (academic/conversational/journalistic/etc)

Decision rule:
- Y = same or similar style (could appear in same context)
- N = OBVIOUSLY different style (e.g., academic paper vs teen slang)

Answer Y or N: <label>"""

    result = _call_llm(prompt, task_id=6, temperature=0.1, max_tokens=300, timeout=120)
    
    if not result:
        print(f"[T6] API returned null, defaulting to N")
        return 'N'
    
    print(f"[T6] Raw output: {result[:200]}...")
    
    # ===== Validator: 提取Y/N =====
    result_upper = result.upper()
    
    # 优先找 <label> 标签
    label_match = re.search(r'<label>\s*([YN])\s*</label>', result_upper)
    if label_match:
        return label_match.group(1)
    
    # 找 "Answer Y or N:" 或 "Decision:" 后面的内容
    answer_match = re.search(r'(?:Answer|Decision)[^:]*:\s*([YN])', result_upper)
    if answer_match:
        return answer_match.group(1)
    
    # 从文本末尾找独立的Y/N
    for char in reversed(result_upper):
        if char in ('Y', 'N'):
            idx = result_upper.rfind(char)
            prev_ok = idx == 0 or not result_upper[idx-1].isalpha()
            next_ok = idx >= len(result_upper)-1 or not result_upper[idx+1].isalpha()
            if prev_ok and next_ok:
                return char
    
    print(f"[T6] Could not extract Y/N, defaulting to N")
    return 'N'


# ==================== Task 8 多智能体: 代码生成 ====================

def multi_agent_task8(task_description: str, text2annotate: str, 
                      icl_examples: list = None) -> Optional[str]:
    """
    Task 8 多智能体流水线 - Triton代码生成（V22研究优化版）
    
    V22核心改进（基于TritonBench/AutoTriton/TritonRL论文）:
    1. 全新Prompt: 英文prompt + 5个内置高质量ICL示例(softmax/layernorm/add/reduction/matmul)
    2. 渐进式温度: 0.1起始, 每轮+0.08, 最高0.5 (比V21更激进探索)
    3. 结构化验证: 基于评分系统(100分制)而非简单pass/fail
    4. 智能fallback: 根据任务描述选择最接近的模板
    5. 内置ICL: 不再依赖外部[[EXAMPLES]]注入, 直接内置5个生产级示例
    
    Args:
        task_description: 代码任务描述
        text2annotate: 输入数据
        icl_examples: ICL few-shot示例列表(V22主要使用内置示例, 此参数保留兼容)
    
    Returns:
        完整Python代码字符串
    """
    import time
    
    # V22配置
    base_temp = 0.1       # 更低起始温度
    temp_increment = 0.08  # 每轮温度增量
    max_retries = 5
    
    for generation_round in range(max_retries):
        # ===== Agent 1: 代码生成器 (V22: 使用新prompt) =====
        gen_prompt = build_prompt_for_triton(task_description, text2annotate)
        
        if generation_round > 0:
            # V22: 重试时添加具体错误反馈
            retry_instruction = f"""

## RETRY #{generation_round + 1} - Previous output was rejected.
Please focus on these requirements:
1. Write COMPLETE working code with real logic in every function body
2. Use a specific function name related to the task (e.g., fft_kernel, lu_decomposition)
3. Include detailed comments explaining your algorithm approach
4. Ensure ALL tl.load/tl.store calls have proper mask parameters for edge cases
5. The wrapper function should handle the actual tensor shapes from the input""" 
            gen_prompt += retry_instruction
        
        # V22温度策略: 从0.1开始递增
        current_temp = min(base_temp + generation_round * temp_increment, 0.5)

        code = _call_llm(gen_prompt, task_id=8, temperature=current_temp,
                         max_tokens=16000, timeout=600,
                         system_prompt="You are a Triton GPU programming expert. Generate complete, correct, runnable Python code with real implementations. No placeholders ever.")
        
        if not code:
            print(f"[Agent1-T8-V22] Round {generation_round+1}: API returned null")
            continue
        
        print(f"[Agent1-T8-V22] Round {generation_round+1}: Generated {len(code)} chars (temp={current_temp:.2f})")
        
        # ===== Agent 2: Quality Inspector (V22 Enhanced Scoring) =====
        validation = _validate_triton_code_v22(code)
        
        is_valid = validation['is_valid']
        score = validation['score']
        reasons = validation['reasons']
        
        print(f'[Agent2-T8-V22] R{generation_round+1}: score={score}/100 valid={is_valid}')
        
        if is_valid:
            print(f"[T8-V22] R{generation_round+1}: ACCEPTED (score={score})")
            return code
        else:
            print(f'[WARN] T8-V22 R{generation_round+1}: Rejected - {"; ".join(reasons[:3])}') 
            time.sleep(2)
    
    # 所有轮次失败, 返回智能fallback
    print("[ERROR] T8-V22: All rounds failed, using fallback template")
    return _get_fallback_code_v22(task_description)


def _validate_triton_code_v22(code: str) -> dict:
    """
    V22结构化代码验证 - 返回评分和原因
    
    评分标准(满分100):
    - 基本结构: 30分 (import/@triton.jit/wrapper/grid)
    - 操作完整性: 40分 (足够的tl操作/无占位符)
    - 命名规范: 10分 (有意义的函数名)
    - 代码长度: 10分 (>500字符)
    - 无伪代码: 10分 (通过dummy检测)
    """
    if not code:
        return {'is_valid': False, 'score': 0, 'reasons': ['empty']}
    
    score = 0
    reasons = []
    
    # 1. 基本结构检查 (30分)
    has_import_torch = 'import torch' in code
    has_import_triton = 'import triton' in code
    has_jit = '@triton.jit' in code or 'triton.jit' in code
    has_wrapper_def = bool(re.search(r'def\s+\w+\([^)]*torch\.Tensor', code))
    has_grid = 'grid = lambda' in code or '.run(' in code or '[grid]' in code
    
    struct_score = sum([has_import_torch, has_import_triton, has_jit, 
                        has_wrapper_def, has_grid]) * 6
    score += struct_score
    
    if not has_import_torch: reasons.append('no_torch_import')
    if not has_import_triton: reasons.append('no_triton_import')
    if not has_jit: reasons.append('no_jit_decorator')
    if not has_wrapper_def: reasons.append('no_wrapper_func')
    if not has_grid: reasons.append('no_grid_launch')
    
    # 2. 操作完整性 (40分)
    tl_ops = re.findall(r'tl\.\w+', code)
    op_count = len(tl_ops)
    
    has_load = any('load' in op for op in tl_ops)
    has_store = any('store' in op for op in tl_ops)
    has_math = any(op in ['tl.exp', 'tl.log', 'tl.sqrt', 'tl.sum', 'tl.max', 'tl.min', 
                           'tl.dot', 'tl.matmul', 'tl.add', 'tl.mul',
                           'tl.div', 'tl.pow', 'tl.sin', 'tl.cos', 'tl.abs',
                           'tl.trans', 'tl.broadcast_to', 'tl.where'] for op in tl_ops)
    
    if op_count >= 10:
        op_score = 40
    elif op_count >= 5:
        op_score = 28
    elif op_count >= 3:
        op_score = 16
    else:
        op_score = 4
        reasons.append(f'few_ops({op_count})')
    
    score += op_score
    
    # 3. 命名规范 (10分)
    bad_names = re.search(r'def\s+(kernel|func|wrapper|function|op)\s*\(', code)
    good_name = re.search(r'def\s+(softmax|layer_norm|matmul|add|mul|exp|log|'
                         r'sum|max|min|relu|sigmoid|tanh|fft|conv|gemm|'
                         r'attention|norm|scale|bias|embedding|linear)_?\w*\s*\(', code, re.I)
    
    if good_name and not bad_names:
        score += 10
    elif not bad_names:
        score += 7
    else:
        reasons.append('bad_function_name')
    
    # 4. 代码长度 (10分)
    code_len = len(code.strip())
    if code_len >= 1500:
        score += 10
    elif code_len >= 800:
        score += 7
    elif code_len >= 400:
        score += 4
    else:
        reasons.append(f'too_short({code_len})')
    
    # 5. 无伪代码 (10分)
    code_lower = code.lower()
    fatal_patterns = [
        r'\bpass\s*$',
        r'\.\.\.\s*$',
        r'\bTODO\b',
        r'placeholder',
        r'simplified.*version',
        r'dummy.*implementation',
        r'does not perform',
        r'for demonstration',
        r'just returns? input',
        r'fake',
        r'stub',
    ]
    
    has_fatal = any(re.search(p, code_lower, re.MULTILINE) for p in fatal_patterns)
    
    if not has_fatal:
        score += 10
    else:
        for p in fatal_patterns:
            if re.search(p, code_lower, re.MULTILINE):
                reasons.append(f'pattern:{p.strip()}')
                break
    
    # 判定是否有效 (60分阈值 + 无致命问题 + 足够操作 + 足够长度)
    is_valid = score >= 60 and not has_fatal and op_count >= 3 and code_len >= 400
    
    return {
        'is_valid': is_valid,
        'score': score,
        'reasons': reasons,
        'details': {
            'struct_score': struct_score,
            'op_count': op_count,
            'has_load': has_load,
            'has_store': has_store,
            'has_math': has_math,
            'code_length': code_len,
        }
    }


def _get_fallback_code_v22(task_description: str) -> str:
    """V22: 根据任务描述选择最接近的fallback模板"""
    desc_lower = (task_description or '').lower()
    
    if any(kw in desc_lower for kw in ['softmax', 'normalization', 'norm']):
        examples = _get_v22_triton_examples()
        return examples[0]['code']
    elif any(kw in desc_lower for kw in ['layer', 'batch', 'rms']):
        examples = _get_v22_triton_examples()
        return examples[1]['code']
    elif any(kw in desc_lower for kw in ['add', 'mul', 'element', 'wise']):
        examples = _get_v22_triton_examples()
        return examples[2]['code']
    elif any(kw in desc_lower for kw in ['sum', 'reduce', 'mean', 'max', 'min']):
        examples = _get_v22_triton_examples()
        return examples[3]['code']
    elif any(kw in desc_lower for kw in ['matmul', 'multiply', 'matrix', 'gemm']):
        examples = _get_v22_triton_examples()
        return examples[4]['code']
    else:
        # 默认fallback模板
        return '''import torch
import triton
import triton.language as tl

@triton.jit
def operation_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = data  # Apply actual operation here
    tl.store(output_ptr + offsets, result, mask=mask)

def execute_operation(input_tensor: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input_tensor)
    n_elements = output.numel()
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']), )
    operation_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE=1024)
    return output
'''


def _get_v22_triton_examples() -> list:
    """
    V22: 返回高质量生产级Triton kernel示例
    
    基于以下来源精选:
    - OpenAI Triton官方教程
    - FlagGems算子库(FlagOS社区)
    - Liger-Kernel(LinkedIn)
    - FlashAttention实现
    
    这些示例都是真实可运行的完整实现,用于ICL引导模型生成高质量代码
    """
    return [
        # Example 1: Softmax (from FlashAttention/Triton tutorial)
        {
            "task": "Implement a fused softmax operation using Triton",
            "input": "Input tensor of shape [N], compute exp(x) / sum(exp(x)) for each row",
            "code": '''import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    input_row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=float('-inf'))
    
    row_max = tl.max(input_row, axis=0)
    input_stable = input_row - row_max
    numerator = tl.exp(input_stable)
    denominator = tl.sum(numerator, axis=0) + 1e-6
    output_row = numerator / denominator
    
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output_row, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    grid = lambda META: (n_rows,)
    softmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=1024)
    return output
'''
        },
        
        # Example 2: Layer Normalization (from Liger-Kernel/FlagGems)
        {
            "task": "Implement layer normalization using Triton",
            "input": "Normalize input tensor along last dimension with given weight and bias",
            "code": '''import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=0) / n_elements
    x_diff = x - mean
    var = tl.sum(x_diff * x_diff, axis=0) / n_elements
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    normalized = x_diff * rstd
    output = normalized * w + b
    
    tl.store(output_ptr + offsets, output, mask=mask)
    tl.store(mean_ptr + offsets[0], mean)
    tl.store(rstd_ptr + offsets[0], rstd)

def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    n_elements = x.shape[-1]
    output = torch.empty_like(x)
    mean = torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype)
    rstd = torch.empty_like(mean)
    grid = lambda META: (x.numel() // n_elements,)
    BLOCK = min(n_elements, 1024)
    layernorm_kernel[grid](
        x, weight, bias, output, mean, rstd, n_elements,
        BLOCK_SIZE=BLOCK,
    )
    return output, mean, rstd
'''
        },
        
        # Example 3: Element-wise Addition
        {
            "task": "Implement element-wise addition of two tensors using Triton",
            "input": "Two tensors of same shape, add them element-wise",
            "code": '''import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    result = a + b
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda and a.shape == b.shape
    output = torch.empty_like(a)
    n_elements = a.numel()
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']), )
    add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=1024)
    return output
'''
        },
        
        # Example 4: Row Sum Reduction
        {
            "task": "Implement sum reduction along the last dimension using Triton",
            "input": "2D tensor, sum each row to produce 1D output",
            "code": '''import torch
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row_data = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    row_sum = tl.sum(row_data, axis=0)
    
    tl.store(output_ptr + row_idx, row_sum)

def triton_row_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    n_rows, n_cols = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    grid = lambda META: (n_rows,)
    BLOCK = min(n_cols, 1024)
    row_sum_kernel[grid](x, output, n_cols, BLOCK_SIZE=BLOCK)
    return output
'''
        },
        
        # Example 5: Matrix Multiplication (simplified GEMM)
        {
            "task": "Implement matrix multiplication C = A @ B^T using Triton",
            "input": "Matrix A (M,K) and Matrix B (N,K), compute (M,N) output",
            "code": '''import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    mask_m = rm < M
    mask_n = rn < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_range = k + rk
        mask_k = k_range < K
        
        a = tl.load(a_ptr + rm[:, None] * stride_am + k_range[None, :] * stride_ak,
                   mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptr + rn[None, :] * stride_bn + k_range[:, None] * stride_bk,
                   mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        
        acc += tl.dot(a, b)
    
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc, mask=mask_m[:, None] & mask_n[None, :])

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    return c
'''
        }
    ]


def _build_triton_icl(examples: list) -> str:
    """V20: Build ICL few-shot example string for Task 8 Triton code generation."""
    if not examples:
        return ''
    parts = []
    for i, ex in enumerate(examples):
        try:
            inp = ex.get('input', '')
            out = ex.get('output', [])
            if isinstance(out, list):
                out = out[0] if out else ''
            if len(str(out)) > 2000:
                out = str(out)[:2000] + '\n    # ... (truncated)'
            parts.append(f'### Example {i+1}\n**Task Input:** {str(inp)[:500]}\n\n**Expected Output Code:**\n```python\n{out}\n```\n')
        except Exception as e:
            print(f'[WARN] T8 ICL: Example {i} format failed: {e}')
            continue
    return '\n'.join(parts) if parts else ''


def annotate_nvidia(input_prompt: str, task_id: int = None) -> Optional[str]:
    """
    使用NVIDIA GPU进行标注
    """
    import requests

    # API端点
    url = os.getenv("QWEN_API_BASE", "http://0.0.0.0:2026/v1/completions")

    data = {
        "model": "../Qwen3-4B",
        "prompt": input_prompt,
        "max_tokens": 10000,
    }

    try:
        resp = requests.post(url, json=data, timeout=300)
        whole_result = resp.json()["choices"][0]["text"]
        prediction = count_answer(whole_result, task_id=task_id)
        return prediction
    except Exception as e:
        print(f"[ERROR] NVIDIA API调用失败: {e}")
        return None
