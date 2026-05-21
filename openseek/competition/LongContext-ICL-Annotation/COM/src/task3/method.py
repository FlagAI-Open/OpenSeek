"""
method.py — Task 3 (collatz_conjecture) 方法模块

包含 Task 3 全部专用逻辑：
- ``generate_cot_task3``     : 运行时程序化生成 CoT scratchpad（逐元素 even/odd → result）
- ``Task3ExampleSelector``   : 列表长度匹配分桶 + 渐进排列 + 含 think 的示例选择器
- ``build_task3_prompt``     : Task 3 专用 Prompt（强制逐元素展开 + 严格输出格式）

通用 LLM 调用、答案抽取等能力位于 ``COM/src/common/llm_client.py``。
"""

import ast
import logging
from collections import defaultdict
from typing import Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================
# 1. CoT Scratchpad 生成器（运行时程序化生成）
# ============================================================

def generate_cot_task3(input_text: str, output_text: str) -> str:
    """Task 3 (collatz_conjecture): 逐元素奇偶判断的程序化 CoT。

    格式：
    - 每个元素独立一行 "n even → n//2" 或 "n odd → 3n+1"
    - 用 "; " 分隔步骤（紧凑格式避免 token 过度膨胀）
    - 最终 ``<label>`` 原样输出 ground truth 列表字符串
    """
    try:
        nums = ast.literal_eval(input_text.strip())
        steps = []
        for n in nums:
            n = int(n)
            if n % 2 == 0:
                steps.append(f"{n} even → {n // 2}")
            else:
                steps.append(f"{n} odd → {n * 3 + 1}")
        return "; ".join(steps) + f"\n<label>{output_text}</label>"
    except Exception:
        return f"<label>{output_text}</label>"


# ============================================================
# 2. Task 3 示例选择器（列表长度匹配 + 渐进排列）
# ============================================================

class Task3ExampleSelector:
    """
    Task 3 专用示例选择器。

    流程：
    1. 按列表长度分桶：同长度 (priority=0) / 相邻长度 ±1 (priority=1) / 其他 (priority=2)
    2. 组内按 input 长度短→长（渐进排列）
    3. 整体排列 priority 2 → 1 → 0，末尾贴最相关的同长度示例（recency bias）
    4. token 截断采用 ``reverse_select=True``：从末尾开始累加，
       优先保留 recency bias 排列的高优先级示例

    示例格式（有 think）：
        # {input}
        {think}
        {cot}

    示例格式（仅 cot）：
        # {input} {cot}
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

    def _log_token_budget(self, n_examples: int, total_tokens: int, mode: str) -> None:
        """统一日志：输出 token 数与 [min, max] 区间，低于下限时升级为 WARNING。"""
        msg = (f"选择 {n_examples} 个示例, {total_tokens} tokens "
               f"(区间 [{self.min_context_tokens}, {self.max_context_tokens}], {mode})")
        if total_tokens < self.min_context_tokens:
            logger.warning(f"[ICL 长度不足] {msg} — 低于赛题硬约束下限 {self.min_context_tokens}")
        else:
            logger.info(msg)

    def _truncate_by_tokens(self, example_lines: list,
                            reverse_select: bool = True):
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
            result = "".join(selected)
            self._log_token_budget(len(selected), total_tokens, mode='reverse_select')
            return result, total_tokens
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
            self._log_token_budget(count, total_tokens, mode='forward_select')
            return result, total_tokens

    def select(self, all_examples: list, test_input: str) -> str:
        """列表长度匹配：同长度 / 相邻长度 / 其他长度三档优先级"""
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
        # 同长度 priority=0 (最相关)
        same_len = buckets.get(test_len, [])
        candidates.extend([(ex, 0) for ex in same_len])
        # 相邻长度 priority=1
        for adj in [test_len - 1, test_len + 1]:
            adj_exs = buckets.get(adj, [])
            candidates.extend([(ex, 1) for ex in adj_exs])
        # 其他长度 priority=2
        for length, exs in sorted(buckets.items()):
            if length not in (test_len, test_len - 1, test_len + 1):
                candidates.extend([(ex, 2) for ex in exs])

        # 4. 生成 CoT + 拼接示例行
        example_lines = []
        for ex, priority in candidates:
            output_text = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            # cot 优先用样本自带（离线生成），否则运行时程序化生成保底
            cot = ex.get('cot') if ex.get('cot') else generate_cot_task3(
                ex['input'], output_text)
            think = ex.get('think', '').strip() if self.use_think else ''
            if think:
                # 带 think：# input\n{think}\n{cot}
                line = f"# {ex['input']}\n{think}\n{cot}\n"
            else:
                # 仅 cot：# input + cot
                line = f"# {ex['input']} {cot}\n"
            example_lines.append((line, priority, len(ex['input'])))

        # 5. 打印前几条 CoT 示例供人工核验（仅首次）
        if example_lines and not self._cot_sample_logged:
            self._cot_sample_logged = True
            logger.info(f"[Task3 CoT 示例样本] 共 {len(example_lines)} 条，前 3 条：")
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
            group.sort(key=lambda x: x[1])  # 短→长
            ordered_lines.extend([line for line, _ in group])

        # 8. reverse_select=True：token 预算紧张时优先保留末尾的同长度示例
        return self._truncate_by_tokens(ordered_lines, reverse_select=True)[0]


# ============================================================
# 3. Task 3 Prompt 构建（与 generate_cot.py 规则完全一致）
# ============================================================

def build_task3_prompt(task_description: str, text2annotate: str,
                       examples_str: str) -> str:
    """Task 3 (collatz_conjecture) Prompt — 强制逐元素展开 + 严格输出格式。

    设计要点：
    1. 严格 Output Format + 元素逐行展开 ``n even/odd → result``
    2. 显式禁止跳元素 / 多次应用 / markdown / 自我反思
    3. 最后输出 ``<label>[列表]</label>``
    """
    return (
        "### Task\n"
        f"{task_description}\n\n"

        "### Output Format (MUST follow exactly)\n"
        "n1 even/odd \u2192 result1\n"
        "n2 even/odd \u2192 result2\n"
        "... (one line per element, keep the ORIGINAL order of the input)\n"
        "<label>[result1, result2, ..., resultN]</label>\n\n"

        "### Rules\n"
        "1. For EACH element n in the input list (keep the ORIGINAL order):\n"
        "   - If n is even \u2192 n // 2\n"
        "   - If n is odd  \u2192 3 * n + 1\n"
        "2. You MUST produce exactly N lines (one per input element, "
        "where N = length of the input list). Do NOT skip any element.\n"
        "3. Collect the per-element results into a Python-style list inside <label> tags. "
        "The output list MUST have the SAME length as the input list.\n"
        "4. Do NOT apply the Collatz step multiple times. Apply it exactly ONCE per element.\n"
        "5. Output ONLY in the format above. NO markdown, NO explanation, "
        "NO 'Step 1', NO 'Wait', NO self-correction.\n\n"

        "### Examples\n"
        f"{examples_str}\n\n"

        "Now annotate the following input. "
        "Show each element's parity and result step by step.\n"
        f"# {text2annotate}\n"
    )
