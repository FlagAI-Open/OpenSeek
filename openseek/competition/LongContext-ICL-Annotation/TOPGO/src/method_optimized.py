"""
FlagOS OpenSeek 赛道三 - TOPGO团队
标注方法模块（优化版 v2.0）

重大改进:
1. 任务特定的提示词模板
2. 改进的答案提取逻辑
3. 后处理管道
4. 答案验证机制
"""

import re
import os
from collections import Counter
from typing import Optional, List


# ============ 提示词构建 ============

def build_prompt(task_description: str, text2annotate: str, task_id: int = None) -> str:
    """
    构建高质量的标注提示词（优化版）
    
    根据任务ID使用特定的提示词模板
    """
    if task_id == 1:
        return build_prompt_task1(task_description, text2annotate)
    elif task_id == 2:
        return build_prompt_task2(task_description, text2annotate)
    elif task_id == 4:
        return build_prompt_task4(task_description, text2annotate)
    elif task_id == 5:
        return build_prompt_task5(task_description, text2annotate)
    elif task_id == 6:
        return build_prompt_task6(task_description, text2annotate)
    elif task_id == 7:
        return build_prompt_task7(task_description, text2annotate)
    elif task_id == 8:
        return build_prompt_for_triton(task_description, text2annotate)
    else:
        return build_prompt_generic(task_description, text2annotate)


def build_prompt_generic(task_description: str, text2annotate: str) -> str:
    """通用提示词模板"""
    prompt = (
        "### 任务\n"
        f"{task_description}\n\n"
        "### 规则\n"
        "1. 仔细分析提供的示例\n"
        "2. 只输出最终答案，不要解释\n"
        "3. 将答案包裹在<label>标签中\n\n"
        "### 示例\n"
        "[[EXAMPLES]]\n\n"
        "### 待标注\n"
        f"{text2annotate}\n\n"
        "### 输出\n"
        "<label>你的答案</label>"
    )
    return prompt


def build_prompt_task1(task_description: str, text2annotate: str) -> str:
    """
    任务1专用提示词 - 最小绝对差（确定性数学计算）
    
    关键改进：
    1. 强调这是纯数学计算任务
    2. 明确输出格式要求
    3. 简洁直接的指令
    """
    return f"""这是一个数学计算任务。

任务：找出列表中两个整数之间的最小绝对差。

规则：
- 只输出数字答案，不要解释
- 如果列表中所有数字都相同，答案为0
- 直接给出最小差值（整数）

示例：
输入: [1, 5, 3, 9]
分析：|1-5|=4, |1-3|=2, |1-9|=8, |5-3|=2, |5-9|=4, |3-9|=6
最小差值是2
答案：2

输入: [10, 10, 10]
分析：所有数字相同，差值为0
答案：0

现在计算：
{text2annotate}

答案（只输出数字）：<label>"""


def build_prompt_task2(task_description: str, text2annotate: str) -> str:
    """
    任务2专用提示词 - 统计名词和动词数量
    
    关键改进：
    1. 明确输出格式
    2. 强调只输出数字
    3. 简洁指令
    """
    return f"""统计句子中名词和动词的总数量。

规则：
- 只输出数字，不要解释
- 名词：人、地点、事物、概念的名称
- 动词：表示动作或状态的词
- 统计名词+动词的总数

示例：
"The cat sat on the mat."
名词：cat, mat (2个)
动词：sat (1个)
总数：3
答案：3

"The boy runs quickly to school."
名词：boy, school (2个)
动词：runs (1个)
总数：3
答案：3

现在统计：
{text2annotate}

答案（只输出数字）：<label>"""


def build_prompt_task4(task_description: str, text2annotate: str) -> str:
    """
    任务4专用提示词 - 字符串连接
    
    关键改进：
    1. 更明确的指令
    2. 清晰的示例
    3. 强调只输出答案
    """
    return f"""将以下字符串列表连接成一个字符串。

规则：
- 直接将所有字符串连接在一起
- 不要添加空格或其他字符
- 只输出连接结果，不要解释

示例：
输入: ['p', 'that.', 'o']
输出: pthat.o

输入: ['hello', 'world']
输出: helloworld

输入: ['a', 'b', 'c']
输出: abc

现在处理：
输入: {text2annotate}
输出："""


def build_prompt_task7(task_description: str, text2annotate: str) -> str:
    """
    任务7专用提示词 - Jeopardy问答（优化版）
    
    关键改进：
    1. 强调从文中找答案
    2. 简洁答案（1-3个词）
    3. 禁止解释和思考过程
    4. 直接输出答案，不输出任何解释
    """
    return f"""根据给定的内容，回答问题。

规则：
- 只从提供的内容中提取答案
- 答案要极其简洁（1-3个词）
- 不要解释，不要思考过程
- 直接输出答案，不要输出任何额外文字

示例：
内容: "Paris is the capital of France."
问题: What is the capital of France?
答案: Paris

现在回答：
{text2annotate}
答案："""


def build_prompt_task5(task_description: str, text2annotate: str) -> str:
    """
    任务5专用提示词 - 情感分析（强抗偏斜版）
    
    优化策略：
    1. 明确Sad定义，收紧判断标准
    2. 详细列举NOT Sad的情况
    3. 平衡示例（1:1 Sad/Not Sad）
    4. 添加"常见错误"警告
    """
    return f"""判断这条推文是否表达真正的情感悲伤（SADNESS）。

=== SAD的严格定义 ===
只有满足以下条件才能判为Sad：
- 因失去亲人、宠物、恋人而悲伤、哀悼
- 抑郁、绝望、深深的痛苦
- 哭泣、心碎、无法抑制的悲伤
- 极度思念某人、感到孤独被遗弃

=== NOT SAD的详细列举（这些都判Not sad）===
- 对服务、产品、公司的抱怨投诉
- 讽刺、吐槽、调侃、幽默
- 愤怒、批评、指责
- 中性陈述、客观观察、问题
- 体育、媒体、新闻的观点评论
- 疲惫、压力、忙碌
- 身体不适但无情感痛苦
- "@某人"的争论或闲聊
- 对某事感到"terrible/dreadful"但实际是夸张表达

=== 示例（平衡1:1）===

Sad示例：
"我失去了工作，感觉完全崩溃了" -> Sad
"我的狗昨天去世了，我好想它" -> Sad
"我无法停止哭泣，一切都感觉没有希望" -> Sad

Not Sad示例：
"这家餐厅的服务太差了" -> Not sad
"你的体育观点真可笑" -> Not sad
"联合航空在纽瓦克需要更多自助值机亭" -> Not sad
"草增长模拟器被冒犯了" -> Not sad（幽默）
"刚完成期末考试，感觉轻松了！" -> Not sad

=== 待分类推文 ===
"{text2annotate}"

答案（只能是"Sad"或"Not sad"）：<label>"""


def build_prompt_task6(task_description: str, text2annotate: str) -> str:
    """
    任务6专用提示词 - MNLI同文体分类（强抗偏斜版）
    
    优化策略：
    1. 强调宽松判断Y
    2. 明确什么算"same genre"
    3. 平衡示例（1:1 Y/N）
    """
    return f"""判断两个句子是否属于相同的写作风格或体裁。

=== 判断标准 ===

判Y（相同体裁）的情况：
- 两者正式程度相似（都正式或都随意）
- 句子结构和词汇复杂度相似
- 可能出现在同一类型出版物中
- 一个是对另一个的改写
- 两者都是对话/都是书面语

判N（不同体裁）的情况：
- 一个正式/学术，另一个明显非正式
- 一个是口语对话，另一个是书面文章
- 主题完全不同
- 写作风格差异巨大

=== 示例（平衡1:1）===

Y示例：
"我不具备现在弥补这些缺陷的精力" / "我没有力气现在解决这些问题" -> Y（都正式）
"我觉得这对他们来说很可怕，看到角色互换" / "他们一次又一次地决斗" -> Y（都对话）
"这是一种完全合法化的出路" / "这是最实用的解决方案" -> Y（都官僚）

N示例：
"然而，如果为了使患者参与治疗而获取它..." / "根据专栏作家，恐惧的生态学..." -> N（医学vs生态）
"你有没有...我假设你没有在军队待过" / "乔恩说我们没有其他办法了" -> N（陈述vs回应）

=== 输入 ===
{text2annotate}

答案Y或N：<label>"""


def build_prompt_for_triton(task_description: str, text2annotate: str) -> str:
    """
    为Triton代码生成任务构建专用提示词（强抗占位符版）
    
    关键改进：
    1. 明确禁止任何占位符
    2. 要求完整可执行代码
    3. 强调必须处理所有边界情况
    """
    prompt = f"""你是一个专业的Triton GPU编程专家。根据自然语言描述生成完整的Triton kernel代码。

### 绝对禁止（违反将导致答案无效）
- 禁止使用"simplified"、"placeholder"、"TODO"、"FIXME"等占位符
- 禁止使用"..."或"pass"作为函数体
- 禁止省略任何代码部分
- 禁止使用"此处应有..."等提示

### 关键要求
1. 生成完整的、可执行的Python代码（不是示例）
2. 包含所有必要的import语句
3. 使用 @triton.jit 装饰kernel函数
4. 提供完整的Python wrapper函数
5. 处理所有边界条件（mask处理）
6. 所有变量必须初始化，所有函数必须有实现

### 标准代码结构

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # 执行计算
    output = data
    tl.store(output_ptr + offsets, output, mask=mask)

def wrapper_function(input_tensor: torch.Tensor):
    output = torch.empty_like(input_tensor)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel_name[grid](input_tensor, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 任务描述
{task_description}

### 输入
{text2annotate}

### 输出要求
将完整的Python代码包裹在<label>标签中，不要包含任何解释：

<label>
```python
# 你的代码
```
</label>

请生成代码："""
    return prompt


# ============ 示例选择 ============

def select_examples(all_examples: list, task_description: str, text2annotate: str,
                    tokenizer=None, task_id: int = None) -> str:
    """
    选择高质量示例（优化版）
    
    改进：
    1. 筛选有效示例
    2. 任务特定的过滤
    3. 限制示例数量
    """
    # 筛选高质量示例
    filtered_examples = filter_high_quality_examples(all_examples, task_id)
    
    # 限制示例数量（避免上下文过长）
    max_examples = 10 if task_id in [4, 7] else 20
    selected_examples = filtered_examples[:max_examples]
    
    # 构建示例字符串
    examples_str = ""
    for ex in selected_examples:
        try:
            input_text = ex.get('input', '')
            output = ex.get('output', [])
            output_text = output[0] if isinstance(output, list) else output
            
            # 任务特定的格式化
            if task_id == 4:
                examples_str += f"输入: {input_text}\n输出: {output_text}\n\n"
            elif task_id == 7:
                examples_str += f"问题: {input_text}\n答案: {output_text}\n\n"
            else:
                examples_str += f"# {input_text} <label> {output_text} </label>\n"
        except:
            continue
    
    return examples_str


def filter_high_quality_examples(examples: list, task_id: int = None) -> list:
    """
    筛选高质量示例
    
    策略：
    1. 检查输入输出格式
    2. 过滤异常长度
    3. 任务特定的过滤
    """
    filtered = []
    
    for ex in examples:
        try:
            input_text = ex.get('input', '')
            output = ex.get('output', [])
            output_text = output[0] if isinstance(output, list) else output
            
            # 基本检查
            if not input_text or not output_text:
                continue
            
            # 任务特定检查
            if task_id == 4:
                # 任务4输出应该没有空格，且长度适中
                if len(output_text) > 100:
                    continue
                # 过滤包含中文的输出（可能是错误）
                if re.search(r'[\u4e00-\u9fff]', output_text):
                    continue
                    
            elif task_id == 7:
                # 任务7输出应该较短
                if len(output_text) > 50:
                    continue
                # 过滤思考标记
                if output_text.strip() in ['</think>', '<think>']:
                    continue
            
            filtered.append(ex)
        except:
            continue
    
    return filtered


# ============ 答案提取 ============

def count_answer(text: str, task_id: int = None) -> Optional[str]:
    """
    提取答案（优化版）
    
    根据任务ID使用特定的提取策略
    """
    if not text:
        return None
    
    # 任务特定提取
    if task_id == 1:
        return extract_task1_answer(text)
    elif task_id == 2:
        return extract_task2_answer(text)
    elif task_id == 4:
        return extract_task4_answer(text)
    elif task_id == 6:
        return extract_task6_answer(text)  # MNLI Y/N分类
    elif task_id == 7:
        return extract_task7_answer(text)
    elif task_id == 8:
        return extract_task8_answer(text)
    
    # 通用提取
    return extract_generic_answer(text)


def extract_task1_answer(text: str) -> Optional[str]:
    """
    提取任务1的答案 - 最小绝对差
    
    关键改进：
    1. 提取数字答案
    2. 过滤所有非数字内容
    """
    if not text:
        return None
    
    # 移除所有标签
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    
    # 尝试提取<label>标签内的内容
    pattern = r'<label>\s*(\d+)\s*</label>'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    
    # 尝试提取纯数字
    numbers = re.findall(r'\d+', text)
    if numbers:
        # 返回最后一个找到的数字（通常在答案位置）
        return numbers[-1]
    
    return None


def extract_task2_answer(text: str) -> Optional[str]:
    """
    提取任务2的答案 - 统计名词动词数量
    
    关键改进：
    1. 提取数字答案
    2. 过滤所有非数字内容
    3. 优先提取<label>标签内的内容
    4. 如果没有标签，尝试提取独立的数字答案
    """
    if not text:
        return None
    
    # 移除思考标记
    text = re.sub(r'</think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 移除所有HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    
    # 尝试提取<label>标签内的内容
    pattern = r'<label>\s*(\d+)\s*</label>'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    
    # 按行分割，从后往前找纯数字行（答案通常在最后）
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # 过滤包含中文的行
        if re.search(r'[\u4e00-\u9fff]', line):
            continue
        # 过滤推理关键词
        reasoning_keywords = ['first', 'second', 'therefore', 'however', 'because',
                             'analyze', 'analysis', 'sentence', 'premise', 'hypothesis',
                             'thus', 'means', 'implied', 'conclusion', 'answer', '答案是']
        if any(kw in line.lower() for kw in reasoning_keywords):
            continue
        # 检查是否是纯数字
        if re.match(r'^\d+$', line):
            return line
    
    # 尝试提取纯数字（最后兜底）
    numbers = re.findall(r'\d+', text)
    if numbers:
        # 返回最后一个找到的数字
        return numbers[-1]
    
    return None


def extract_task4_answer(text: str) -> Optional[str]:
    """
    提取任务4的答案 - 字符串连接
    
    关键改进：
    1. 移除所有标签
    2. 过滤思考过程（中文）
    3. 提取最后一行有效内容
    """
    if not text:
        return None
    
    # 移除所有标签
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    
    # 分割成行
    lines = text.split('\n')
    
    # 从后向前查找有效答案
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        
        # 过滤包含中文的行（通常是思考过程）
        if re.search(r'[\u4e00-\u9fff]', line):
            continue
        
        # 过滤关键词
        if any(kw in line.lower() for kw in ['标签', 'label', '示例', 'example', '输入', 'output']):
            continue
        
        # 过滤过短的结果
        if len(line) < 1:
            continue
        
        return line
    
    return None


def extract_task6_answer(text: str) -> Optional[str]:
    """
    提取任务6的答案 - MNLI蕴含分类任务
    
    任务6的特点：
    - 输出是Y或N
    - 答案非常短
    - 必须过滤掉推理过程中出现的Y/N字符
    """
    if not text:
        return None
    
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
    text = re.sub(r'heed>.*? </think>', '', text, flags=re.DOTALL)
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
        # 检查是否包含Y或N（但不是作为推理的一部分）
        if ' Y ' in line or line.startswith('Y ') or line.endswith(' Y') or line == 'Y':
            return 'Y'
        if ' N ' in line or line.startswith('N ') or line.endswith(' N') or line == 'N':
            return 'N'
    
    return None


def extract_task7_answer(text: str) -> Optional[str]:
    """
    提取任务7的答案 - 阅读理解
    
    关键改进：
    1. 过滤思考标记
    2. 移除HTML标签
    3. 限制长度
    """
    if not text:
        return None
    
    text = text.strip()
    
    # 过滤思考标记
    if text.strip() in ['</think>', '<think>', '</label>', '<label>', '']:
        return None
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 取第一行
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    answer = lines[0]
    
    # 过滤过短的答案
    if len(answer) < 2:
        return None
    
    # 限制长度
    return answer[:100]


def extract_task8_answer(text: str) -> Optional[str]:
    """
    提取任务8的答案 - Triton代码
    
    提取代码块内容
    """
    if not text:
        return None
    
    # 尝试提取<label>标签内的内容
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        content = matches[-1].strip()
        
        # 提取代码块
        code_pattern = r'```python\n(.*?)\n```'
        code_matches = re.findall(code_pattern, content, re.DOTALL)
        
        if code_matches:
            return code_matches[-1].strip()
        
        return content
    
    # 直接提取代码块
    code_pattern = r'```python\n(.*?)\n```'
    code_matches = re.findall(code_pattern, text, re.DOTALL)
    
    if code_matches:
        return code_matches[-1].strip()
    
    return text.strip()


def extract_generic_answer(text: str) -> Optional[str]:
    """通用答案提取"""
    if not text:
        return None
    
    # 尝试提取<label>标签
    pattern = r'<label>\s*(.*?)\s*</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    # 取第一行
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[0] if lines else None


# ============ 答案验证 ============

def validate_prediction(prediction: str, task_id: int) -> tuple[bool, Optional[str]]:
    """
    验证预测结果
    
    返回: (是否有效, 修正后的预测)
    """
    if not prediction:
        return False, None
    
    if task_id == 1:
        return validate_task1_prediction(prediction)
    elif task_id == 2:
        return validate_task2_prediction(prediction)
    elif task_id == 4:
        return validate_task4_prediction(prediction)
    elif task_id == 7:
        return validate_task7_prediction(prediction)
    
    return True, prediction


def validate_task1_prediction(prediction: str) -> tuple[bool, Optional[str]]:
    """验证任务1预测 - 必须是整数"""
    if not prediction:
        return False, None
    
    prediction = prediction.strip()
    
    # 必须是纯数字
    if not re.match(r'^\d+$', prediction):
        return False, None
    
    return True, prediction


def validate_task2_prediction(prediction: str) -> tuple[bool, Optional[str]]:
    """验证任务2预测 - 必须是整数"""
    if not prediction:
        return False, None
    
    prediction = prediction.strip()
    
    # 必须是纯数字
    if not re.match(r'^\d+$', prediction):
        return False, None
    
    return True, prediction


def validate_task4_prediction(prediction: str) -> tuple[bool, Optional[str]]:
    """验证任务4预测"""
    # 过滤包含中文的预测
    if re.search(r'[\u4e00-\u9fff]', prediction):
        return False, None
    
    # 过滤思考过程关键词
    invalid_keywords = ['标签', 'label', '示例', 'example', '输入', '首先', '我需要']
    for kw in invalid_keywords:
        if kw in prediction.lower():
            return False, None
    
    # 清理并返回
    cleaned = prediction.strip()
    return True, cleaned


def validate_task7_prediction(prediction: str) -> tuple[bool, Optional[str]]:
    """验证任务7预测"""
    # 过滤思考标记
    if prediction.strip() in ['</think>', '<think>', '</label>', '<label>']:
        return False, None
    
    # 过滤过短的答案
    if len(prediction.strip()) < 2:
        return False, None
    
    return True, prediction.strip()


# ============ 标注函数 ============

def annotate_ascend(input_prompt: str, task_id: int = None, is_triton_task: bool = False) -> Optional[str]:
    """
    使用华为Ascend进行标注（优化版）
    
    Args:
        input_prompt: 输入提示词
        task_id: 任务ID（用于选择提取策略）
        is_triton_task: 是否为Triton代码生成任务
    
    Returns:
        标注结果
    """
    import openai
    import time

    # 配置API
    openai.api_key = os.getenv("QWEN_API_KEY", "EMPTY")
    openai.base_url = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1/")

    model = "/home/Qwen/Qwen3-4B"

    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。只输出答案，不要解释。"},
        {"role": "user", "content": input_prompt}
    ]

    # 根据任务类型调整参数
    if is_triton_task or task_id == 8:
        max_tokens = 16000  # 降低到16000，避免服务器超时
        temperature = 0.3
        max_retries = 3  # 任务8增加重试次数
    elif task_id == 1:
        max_tokens = 100   # 任务1输出极短（单个数字）
        temperature = 0.0  # 完全确定性
        max_retries = 1
    elif task_id == 2:
        max_tokens = 100   # 任务2输出极短（单个数字）
        temperature = 0.0  # 完全确定性
        max_retries = 1
    elif task_id == 4:
        max_tokens = 1000  # 任务4输出很短
        temperature = 0.1  # 更确定性
        max_retries = 1
    elif task_id == 7:
        max_tokens = 500   # 任务7输出很短
        temperature = 0.1
        max_retries = 1
    else:
        max_tokens = 10000
        temperature = 0.7
        max_retries = 2

    # 重试逻辑
    last_error = None
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                stream=False,
                timeout=300,  # 5分钟超时
            )
            
            whole_result = response.choices[0].message.content
            
            # 使用任务特定的提取逻辑
            prediction = count_answer(whole_result, task_id=task_id)
            
            # 验证预测结果
            is_valid, prediction = validate_prediction(prediction, task_id or (8 if is_triton_task else 0))
            
            return prediction if is_valid else None
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # 判断错误类型
            if 'connection' in error_str or '连接' in error_str:
                print(f"[WARN] 任务{task_id}连接错误 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避: 1, 2, 4秒
                    print(f"[INFO] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
            elif 'timeout' in error_str or '超时' in error_str:
                print(f"[WARN] 任务{task_id}超时 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[INFO] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
            else:
                # 其他错误，直接返回失败
                print(f"[ERROR] Ascend API调用失败: {e}")
                return None
    
    # 所有重试都失败
    print(f"[ERROR] Ascend API调用失败 (已重试{max_retries}次): {last_error}")
    return None


def annotate_nvidia(input_prompt: str, task_id: int = None) -> Optional[str]:
    """
    使用NVIDIA GPU进行标注（优化版）
    """
    import requests

    # API端点
    url = os.getenv("QWEN_API_BASE", "http://0.0.0.0:2026/v1/completions")

    # 根据任务调整参数
    if task_id == 1:
        max_tokens = 100
        temperature = 0.0
    elif task_id == 2:
        max_tokens = 100
        temperature = 0.0
    elif task_id == 4:
        max_tokens = 1000
        temperature = 0.1
    elif task_id == 7:
        max_tokens = 500
        temperature = 0.1
    else:
        max_tokens = 10000
        temperature = 0.7

    data = {
        "model": "../Qwen3-4B",
        "prompt": input_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, json=data, timeout=300)
        whole_result = resp.json()["choices"][0]["text"]
        
        # 使用任务特定的提取逻辑
        prediction = count_answer(whole_result, task_id=task_id)
        
        # 验证预测结果
        is_valid, prediction = validate_prediction(prediction, task_id or 0)
        
        return prediction if is_valid else None
        
    except Exception as e:
        print(f"[ERROR] NVIDIA API调用失败: {e}")
        return None


# ============ Self-Consistency ============

def annotate_with_consistency(input_prompt: str, task_id: int = None, 
                               num_samples: int = 3) -> Optional[str]:
    """
    使用Self-Consistency提升答案质量
    
    多次采样，选择最一致的答案
    
    Args:
        input_prompt: 输入提示词
        task_id: 任务ID
        num_samples: 采样次数
    
    Returns:
        最一致的答案
    """
    predictions = []
    
    # 多次调用
    for i in range(num_samples):
        try:
            # 使用稍高的temperature进行采样
            import openai
            
            openai.api_key = os.getenv("QWEN_API_KEY", "EMPTY")
            openai.base_url = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1/")
            
            model = "/home/Qwen/Qwen3-4B"
            
            messages = [
                {"role": "system", "content": "你是一个有帮助的助手。只输出答案，不要解释。"},
                {"role": "user", "content": input_prompt}
            ]
            
            # 使用稍高的temperature进行多样性采样
            temp = 0.3 + (i * 0.1)  # 0.3, 0.4, 0.5
            
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                top_p=0.95,
                max_tokens=1000,
                stream=False,
            )
            
            whole_result = response.choices[0].message.content
            pred = count_answer(whole_result, task_id=task_id)
            
            if pred:
                predictions.append(pred)
                
        except Exception as e:
            print(f"[WARN] Self-Consistency采样{i}失败: {e}")
            continue
    
    if not predictions:
        return None
    
    if len(predictions) == 1:
        return predictions[0]
    
    # 选择最常见的答案
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0]
    
    print(f"[INFO] Self-Consistency: {len(predictions)}个样本，最一致答案出现{most_common[1]}次")
    
    return most_common[0]


# 保持向后兼容
build_prompt_for_triton = build_prompt_for_triton
