"""
method.py — Task 1 (closest_integers) 方法模块

包含 Task 1 全部专用逻辑：
- ``compute_task1_answer``  : 程序化求解器（最小相邻差），用于离线 CoT 数据生成时校验示例答案
- ``generate_cot_task1``    : 运行时程序化 CoT 生成器（保底用，离线已生成）
- ``Task1ExampleSelector``  : 列表长度匹配 + 渐进排列示例选择器（支持 think 字段）
- ``build_task1_prompt``    : Task 1 专用 Prompt（强制逐对展开 |a[i+1]-a[i]| = diff）

通用 LLM 调用、答案抽取等能力位于 ``COM/src/common/llm_client.py``。
"""

import ast
import logging
from collections import defaultdict
from typing import Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================
# 1. 程序化求解器（用于离线 CoT 数据生成时校验 ICL 示例答案）
# ============================================================

def compute_task1_answer(input_text: str) -> Optional[str]:
    """Task 1 (closest_integers): 程序化计算正确答案。

    Task 1 是纯算法任务（排序 → 相邻差 → 取 min），答案 100% 确定性可算。
    可用于覆盖模型的错误预测（排序幻觉、自比较、算术错误等）。
    """
    try:
        nums = ast.literal_eval(input_text.strip())
        if not isinstance(nums, list) or len(nums) < 2:
            return None
        sorted_nums = sorted(nums)
        diffs = [abs(sorted_nums[i + 1] - sorted_nums[i])
                 for i in range(len(sorted_nums) - 1)]
        return str(min(diffs))
    except Exception:
        return None


# ============================================================
# 2. CoT Scratchpad 生成器（运行时程序化生成）
# ============================================================

def generate_cot_task1(input_text: str, output_text: str) -> str:
    """Task 1 (closest_integers): Sorted → 逐对 diff 计算 → Min → label

    关键格式：
    - Sorted 以 \\n 开头，确保独占一行（不与 # [input] 挤在同一行）
    - 加 (N) 元素计数，帮模型 track 元素数量一致性
    - 每个 diff 展开实际数字，避免"猜数字"
    - 负数用括号包裹：|-30 - (-96)| = 66
    """
    try:
        nums = ast.literal_eval(input_text.strip())
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        lines = [f"\nSorted ({n}): {sorted_nums}"]
        diffs = []
        for i in range(n - 1):
            a, b = sorted_nums[i], sorted_nums[i + 1]
            d = abs(b - a)
            diffs.append(d)
            lines.append(f"|{b} - ({a})| = {d}" if a < 0 else f"|{b} - {a}| = {d}")
        min_d = min(diffs) if diffs else 0
        lines.append(f"Min = {min_d}")
        lines.append(f"<label>{output_text}</label>")
        return "\n".join(lines)
    except Exception:
        return f"<label>{output_text}</label>"


# ============================================================
# 3. Task 1 示例选择器（列表长度匹配 + 渐进排列）
# ============================================================

class Task1ExampleSelector:
    """
    Task 1 专用示例选择器：
    - 按列表长度分桶：同长度 60% + 相邻长度 30% + 其他 10%
    - 内部排序：按 input 长度短→长（渐进排列）
    - 整体排列：priority 2 (其他) → 1 (相邻) → 0 (同长度)，末尾放最相关
    - Token 截断：reverse_select=True，优先保留末尾的高优先级示例
    - 支持 think 字段：若示例自带 think，按 "# input\n{think}\n{cot}" 格式拼接
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

    def _log_token_budget(self, n_examples: int, total_tokens: int, mode: str) -> None:
        """统一日志：输出 token 数与 [min, max] 区间，低于下限时升级为 WARNING。"""
        msg = (f"选择 {n_examples} 个示例, {total_tokens} tokens "
               f"(区间 [{self.min_context_tokens}, {self.max_context_tokens}], {mode})")
        if total_tokens < self.min_context_tokens:
            logger.warning(f"[ICL 长度不足] {msg} — 低于赛题硬约束下限 {self.min_context_tokens}")
        else:
            logger.info(msg)

    def select(self, all_examples: list[dict], test_input: str) -> str:
        """列表长度匹配：同长度 60% + 相邻长度 30% + 其他 10%"""
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
        # 同长度 priority=0 (highest)
        same_len = buckets.get(test_len, [])
        candidates.extend([(ex, 0) for ex in same_len])
        # 相邻长度 priority=1
        for adj in [test_len - 1, test_len + 1]:
            adj_exs = buckets.get(adj, [])
            candidates.extend([(ex, 1) for ex in adj_exs])
        # 其他长度 priority=2
        for l, exs in sorted(buckets.items()):
            if l not in (test_len, test_len - 1, test_len + 1):
                candidates.extend([(ex, 2) for ex in exs])

        # 4. 生成 CoT + 拼接示例行
        example_lines = []
        for ex, priority in candidates:
            output_text = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            # cot 优先用样本自带（离线生成），否则运行时程序化生成保底
            cot = ex.get('cot') if ex.get('cot') else generate_cot_task1(
                ex['input'], output_text)
            think = ex.get('think', '').strip() if self.use_think else ''
            if think:
                # 带 think：# input\n{think}\n{cot}
                # cot 以 \n 开头（generate_cot_task1 格式），与 think 之间用 \n 衔接
                cot_norm = cot if cot.startswith('\n') else '\n' + cot
                line = f"# {ex['input']}\n{think}{cot_norm}\n"
            else:
                # 无 think 字段时的格式：# input + cot
                line = f"# {ex['input']} {cot}\n"
            example_lines.append((line, priority, len(ex['input'])))

        # 5. 打印前几条 CoT 示例供人工核验（仅首次）
        if example_lines and not self._cot_sample_logged:
            self._cot_sample_logged = True
            logger.info(f"[Task1 CoT 示例样本] 共 {len(example_lines)} 条，前 3 条：")
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
# 4. Task 1 专用 Prompt 构建
# ============================================================

def build_task1_prompt(task_description: str, text2annotate: str,
                       examples_str: str) -> str:
    """Task 1 (closest_integers) 专用 Prompt — 逐步展开 diff 计算。

    关键设计：
    1. 强制逐对展开 |a[i+1] - a[i]| = diff 的计算过程
    2. 避免模型直接"猜" Diffs 列表（压缩格式导致算术错误）
    3. 最后取 min 并输出 <label>
    """
    return (
        "### Task\n"
        f"{task_description}\n\n"

        "### Output Format (MUST follow exactly)\n"
        "Sorted (N): [sorted list in ascending order]\n"
        "|a[1] - a[0]| = diff\n"
        "|a[2] - a[1]| = diff\n"
        "... (one line per consecutive pair, starting from the SECOND element)\n"
        "Min = minimum_difference\n"
        "<label>minimum_difference</label>\n\n"

        "### Rules\n"
        "1. Sort the input list in ascending order. N = number of elements.\n"
        "2. The sorted list MUST contain the EXACT SAME elements as the input "
        "(same count, same values). Verify: input has N numbers, sorted has N numbers.\n"
        "3. Compute |a[i+1] - a[i]| for EACH consecutive pair. "
        "Start from the FIRST pair (second element minus first element). "
        "You should have exactly N-1 difference lines.\n"
        "4. The answer is the MINIMUM of all computed differences.\n"
        "5. Output ONLY in the format above. NO markdown, NO explanation, "
        "NO 'Step 1', NO 'Wait', NO self-correction.\n\n"

        "### Examples\n"
        f"{examples_str}\n\n"

        "Now annotate the following input. "
        "Show each diff computation step by step.\n"
        f"# {text2annotate}\n"
    )
