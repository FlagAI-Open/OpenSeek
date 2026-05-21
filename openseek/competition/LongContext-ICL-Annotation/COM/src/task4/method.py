"""method.py — Task 4 (conala_concat_strings) 方法核心

模块职责：
- ``generate_cot_task4``：程序化 CoT scratchpad 生成器（带空格逐步拼接）
- ``Task4ExampleSelector``：列表长度匹配分桶 + reverse_select + 双边界 token 校验
- ``build_task4_prompt``：5 条 Critical Rules + 带空格示例 + 严格输出格式
- ``postprocess_task4``：智能去空格（模型带空格输出，代码端去空格）

设计要点：
- 模型 tokenizer 对 token 间空格倾向较强；不与之对抗，让模型始终带空格拼接，
  代码后处理去除元素间空格（``postprocess_task4``）。
- ICL 示例选择以输入字符串列表长度匹配为优先级（同长 → 邻 ±1 → 其他），
  组内按 input 长度短→长，整体 priority 2 → 1 → 0（recency bias）。
- ICL token 严格 ≥ 30K（赛题硬约束），同时 ≤ 31K（留 1K 余量）。
"""

import ast
import logging
from collections import defaultdict
from typing import Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================
# 1. CoT scratchpad 生成器（运行时程序化生成；离线 generate_cot.py 也复用）
# ============================================================

def generate_cot_task4(input_text: str, output_text: str) -> str:
    """带空格逐步拼接的 CoT。

    思路：模型 tokenizer 倾向于在 token 间插入空格，与其对抗不如顺势而为。
    示例始终带空格拼接（tokenizer 友好），``<label>`` 输出带空格版本，
    最终由代码后处理去空格（见 ``postprocess_task4``）。

    格式（示例 ``['get', 'B', 'have']``）::

        [1] "get" → get
        [2] + "B" → get B
        [3] + "have" → get B have
        <label>get B have</label>
    """
    try:
        strings = ast.literal_eval(input_text.strip())
        if not strings:
            return f"<label>{output_text}</label>"
        if len(strings) == 1:
            return (f'[1] "{strings[0]}" → {strings[0]}\n'
                    f'<label>{strings[0]}</label>')
        steps = []
        spaced = str(strings[0])
        steps.append(f'[1] "{spaced}" → {spaced}')
        for idx, s in enumerate(strings[1:], start=2):
            s_str = str(s)
            spaced = spaced + " " + s_str
            steps.append(f'[{idx}] + "{s_str}" → {spaced}')
        cot = "\n".join(steps)
        # <label> 输出带空格版本，由后处理去空格
        return f"{cot}\n<label>{spaced}</label>"
    except Exception:
        return f"<label>{output_text}</label>"


# ============================================================
# 2. Task 4 示例选择器（列表长度匹配 + 渐进排列 + 双边界 token 校验）
# ============================================================

class Task4ExampleSelector:
    """Task 4 专用示例选择器。

    分桶策略：
    - priority=0：input 列表长度与 test 输入相同
    - priority=1：相邻长度 ±1
    - priority=2：其他
    组内按 input 长度短→长（渐进展示）；整体顺序 priority 2 → 1 → 0，
    末尾贴最相关同长度示例（recency bias）。

    Token 截断：``reverse_select=True`` 从末尾开始选择，优先保留高优先级示例。

    支持 think 字段：示例若有 think，按 ``# input\\n{think}\\n{cot}`` 拼接；
    否则按 ``# input {cot}`` 拼接。
    """

    def __init__(self, tokenizer: AutoTokenizer, max_context_tokens: int = 31000,
                 min_context_tokens: int = 30000, use_think: bool = True):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.min_context_tokens = min_context_tokens
        self.use_think = use_think
        self._cot_sample_logged = False

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _log_token_budget(self, n_examples: int, total_tokens: int, mode: str):
        msg = (f"选择 {n_examples} 个示例, {total_tokens} tokens "
               f"(区间 [{self.min_context_tokens}, {self.max_context_tokens}], {mode})")
        if total_tokens < self.min_context_tokens:
            logger.warning(f"[ICL 长度不足] {msg} — 低于赛题硬约束下限 {self.min_context_tokens}")
        else:
            logger.info(msg)

    def _truncate_by_tokens(self, example_lines: list[str],
                            reverse_select: bool = True) -> tuple[str, int]:
        """精确 token 截断；reverse_select=True 时从末尾开始选择，
        优先保留 recency bias 排列的高优先级示例。
        """
        if reverse_select:
            selected = []
            total_tokens = 0
            for line in reversed(example_lines):
                line_tokens = self._count_tokens(line)
                if total_tokens + line_tokens > self.max_context_tokens:
                    break
                selected.append(line)
                total_tokens += line_tokens
            selected.reverse()
            self._log_token_budget(len(selected), total_tokens, 'reverse_select')
            return "".join(selected), total_tokens
        else:
            result = ""
            total_tokens = 0
            count = 0
            for line in example_lines:
                line_tokens = self._count_tokens(line)
                if total_tokens + line_tokens > self.max_context_tokens:
                    break
                result += line
                total_tokens += line_tokens
                count += 1
            self._log_token_budget(count, total_tokens, 'forward_select')
            return result, total_tokens

    def select(self, all_examples: list[dict], test_input: str) -> str:
        """列表长度匹配：同长度 + 相邻长度 + 其他"""
        # 1. 解析 test 输入的列表长度
        try:
            test_list = ast.literal_eval(test_input.strip())
            test_len = len(test_list) if isinstance(test_list, list) else 0
        except Exception:
            test_len = 0

        # 2. 按列表长度分桶
        buckets = defaultdict(list)
        for ex in all_examples:
            try:
                ex_list = ast.literal_eval(ex['input'].strip())
                ex_len = len(ex_list) if isinstance(ex_list, list) else 0
            except Exception:
                ex_len = 0
            buckets[ex_len].append(ex)

        # 3. 按优先级组合候选
        candidates = []
        same_len = buckets.get(test_len, [])
        candidates.extend([(ex, 0) for ex in same_len])  # 同长度
        for adj in [test_len - 1, test_len + 1]:         # 相邻 ±1
            adj_exs = buckets.get(adj, [])
            candidates.extend([(ex, 1) for ex in adj_exs])
        for l, exs in sorted(buckets.items()):           # 其他
            if l not in (test_len, test_len - 1, test_len + 1):
                candidates.extend([(ex, 2) for ex in exs])

        # 4. 生成 CoT + 拼接示例行
        example_lines = []
        for ex, priority in candidates:
            output_text = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            # cot 优先用样本自带（离线生成），否则运行时程序化生成保底
            cot = ex.get('cot') if ex.get('cot') else generate_cot_task4(
                ex['input'], output_text)
            think = ex.get('think', '').strip() if self.use_think else ''
            if think:
                line = f"# {ex['input']}\n{think}\n{cot}\n"
            else:
                line = f"# {ex['input']} {cot}\n"
            example_lines.append((line, priority, len(ex['input'])))

        # 5. 打印前几条 CoT 示例供人工核验（仅首次）
        if example_lines and not self._cot_sample_logged:
            self._cot_sample_logged = True
            logger.info(f"[Task4 CoT 示例样本] 共 {len(example_lines)} 条，前 3 条：")
            for idx, (line, pri, _) in enumerate(example_lines[:3]):
                logger.info(f"  [{idx+1}] (priority={pri}) {line.strip()[:200]}")

        # 6. 按 priority 分组，每组内按 input 长度短→长
        by_priority = defaultdict(list)
        for line, pri, input_len in example_lines:
            by_priority[pri].append((line, input_len))

        # 7. 排列顺序：其他(2) → 相邻(1) → 同长度(0)，每组内短→长
        ordered_lines = []
        for pri in [2, 1, 0]:
            group = by_priority.get(pri, [])
            group.sort(key=lambda x: x[1])
            ordered_lines.extend([line for line, _ in group])

        # 8. reverse_select=True：token 预算紧张时优先保留末尾的同长度示例
        return self._truncate_by_tokens(ordered_lines, reverse_select=True)[0]


# ============================================================
# 3. Task 4 专用 Prompt 构建
# ============================================================

def build_task4_prompt(task_description: str, text2annotate: str,
                       examples_str: str) -> str:
    """Task 4 (conala_concat_strings) 专用 Prompt。

    关键设计：
    1. 5 条显式 Critical Rules（不跳元素 / 保留大小写 / 不修改字符 /
       保留标点 / 每个相邻元素间正好一个空格）
    2. 示例格式与 CoT 一致：每步带空格拼接
    3. 模型输出带空格版本，代码后处理去空格（tokenizer 友好策略）
    """
    return (
        "### Task\n"
        f"{task_description}\n\n"

        "### Critical Rules\n"
        "1. Include ALL elements in order \u2014 NO skipping any element.\n"
        "2. Preserve the EXACT case of every character \u2014 'k' stays 'k', NOT 'K'.\n"
        "3. Copy each element character-by-character \u2014 do NOT modify any element.\n"
        "4. Preserve ALL punctuation exactly as given.\n"
        "5. In your output, place EXACTLY ONE space between consecutive elements "
        "(same format as the examples above). The downstream postprocessor will "
        "remove those separator spaces automatically \u2014 do NOT try to pre-remove "
        "them yourself.\n\n"

        "### Examples\n"
        f"{examples_str}\n\n"

        "Now annotate the following input.\n"
        "Write each step with SPACES between elements (same format as the examples above).\n"
        "Then wrap the LAST step's result in <label> tags (keep the spaces).\n"
        "Use the exact tag form <label>...</label> (angle brackets, lowercase, no other variants).\n"
        f"# {text2annotate}\n"
    )


# ============================================================
# 4. 后处理：智能去空格（带空格拼接策略的核心）
# ============================================================

def postprocess_task4(prediction: str, input_text: str) -> str:
    """Task 4 后处理：智能去除模型输出中的空格。

    策略：模型带空格拼接（tokenizer 友好），由代码去空格。
    处理两种情况：
    1. 元素本身均不含空格 → 直接 ``.replace(' ', '')``
    2. 有元素本身含空格 → 按元素顺序匹配，只去掉元素间空格，保留元素内部空格
    """
    if not prediction or ' ' not in prediction:
        return prediction  # 无空格，直接返回

    try:
        elements = [str(e) for e in ast.literal_eval(input_text.strip())]
    except Exception:
        # 解析失败，保守地直接去空格
        return prediction.replace(' ', '')

    has_internal_spaces = any(' ' in e for e in elements)

    if not has_internal_spaces:
        # 简单情况：所有空格都是 tokenizer 伪影，安全去除
        return prediction.replace(' ', '')

    # 复杂情况：有元素含空格，需要按顺序匹配元素
    # 在 prediction 中逐个查找元素，跳过元素间的空格
    result_parts = []
    pos = 0
    for elem in elements:
        idx = prediction.find(elem, pos)
        if idx >= 0:
            result_parts.append(elem)
            pos = idx + len(elem)
        else:
            # 元素未找到，fallback 到直接去空格
            logger.warning(f"postprocess_task4: 元素 '{elem}' 未在预测中找到, "
                           f"fallback 去全部空格")
            return prediction.replace(' ', '')

    return ''.join(result_parts)
