"""method.py — Task 7 自包含模块

Task 7: openseek-7_jeopardy_answer_generation
  输入：Category + Clue + ICL 示例池
  输出：1-5 词小写英文答案

模块组成：
  * _BM25：基于词袋的相似度检索
  * Task7ExampleSelector：三策略示例选择器
      A: Category 三级语义匹配（精确 > 关键词重叠 > 其他）+ BM25 + 四层 recency bias
      B: 纯 Clue BM25（跨 Category，知识更多样）
      C: Category 分区 + 同 zone 随机打散（seed=42）
  * build_task7_prompt：3 种 Prompt 模板（A 直接 / B Plan / C ReAct）
  * build_verifier_prompt：验证器接收 2-3 候选选最优
  * build_task7_retry_prompt：全剔重调用（带排除段）
  * candidate_in_clue_trap：候选作整词出现在 Clue 中即剔除
  * parse_category_letter_constraint / candidate_satisfies_letter：
        Category 引号字母硬约束
  * postprocess_task7：小写 + NFD 去变音 + 去尾标点（保留前导冠词）
  * validate_prediction / rescue_from_thinking：答案校验与抢救
"""

from __future__ import annotations

import logging
import math
import random as _random
import re
import unicodedata
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 推理参数
# ============================================================

TASK7_MAX_TOKENS = 4096          # 思考模式需要充足 token 空间
TASK7_TIMEOUT = 600              # 单次请求超时
TASK7_ENABLE_THINKING = True     # 思考模式
TASK7_REP_PENALTY = 1.1          # 抑制思考复读
TASK7_TEMPERATURE = 0.0          # 确定性输出
TASK7_MAX_INPUT_LENGTH = 31000   # ICL prompt token 上限（赛题硬约束 ≥ 30K，留 1K 余量）
TASK7_MIN_INPUT_LENGTH = 30000   # ICL prompt token 下限（赛题硬约束 ≥ 30K，必须达标）

# 验证器调用参数
TASK7_VERIFIER_MAX_TOKENS = 2048

# /no_think 降级重试参数
TASK7_NOTHINK_MAX_TOKENS = 512
TASK7_NOTHINK_TIMEOUT = 300

# 三策略 ensemble 配置
TASK7_STRATEGIES = ('A', 'B', 'C')

# Category 语义匹配停用词
CATEGORY_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'of', 'in', 'for', 'to', 'and', 'it', 'is',
    'on', 'at', 'by', 'or', 'as', 'no', 'not', 'do', 'are', 'was',
    'be', 'my', 'me', 'we', 'us', 'you', 'your', 'its', 'his', 'her',
    'our', 'their', 'this', 'that', 'with', 'from', 'up', 'out',
    'about', 'what', 'who', 'how', 'when', 'where', 'why', 'which',
    's', 't', 're', 've', 'll', 'd', 'em',
})


# ============================================================
# BM25 检索模型
# ============================================================

class _BM25:
    """BM25 检索模型，用于 Task 7 输入文本相似度计算。"""

    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(doc.split()) for doc in corpus) / self.N if self.N > 0 else 0
        self.df: dict = {}
        self.tf: list = []
        self.doc_len: list = []

        for doc in corpus:
            words = doc.lower().split()
            self.doc_len.append(len(words))
            tf_dict: dict = {}
            for w in words:
                tf_dict[w] = tf_dict.get(w, 0) + 1
            self.tf.append(tf_dict)
            for w in set(words):
                self.df[w] = self.df.get(w, 0) + 1

    def score(self, query: str, doc_idx: int) -> float:
        query_words = query.lower().split()
        score = 0.0
        doc_tf = self.tf[doc_idx]
        dl = self.doc_len[doc_idx]
        for word in query_words:
            if word not in doc_tf:
                continue
            tf = doc_tf[word]
            df = self.df.get(word, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def scores(self, query: str) -> np.ndarray:
        return np.array([self.score(query, i) for i in range(self.N)])


# ============================================================
# 示例行格式
# ============================================================

def _format_example_line(ex: dict) -> str:
    """格式化单个示例为字符串行：``# {input} <label>{output}</label>``"""
    input_text = ex['input']
    output_text = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
    return f"# {input_text} <label>{output_text}</label>\n"


# ============================================================
# 三策略示例选择器
# ============================================================

class Task7ExampleSelector:
    """Task 7 三策略示例选择器。

    A: Category 三级语义匹配（精确 > 关键词重叠 > 其他）+ BM25 + 四层 recency bias
    B: 纯 BM25 按 Clue 检索（跨 Category，知识更多样）
    C: BM25 按 Category 检索 + 同 zone 随机打散（固定 seed=42）
    """

    def __init__(self, tokenizer,
                 max_context_tokens: int = TASK7_MAX_INPUT_LENGTH,
                 min_context_tokens: int = TASK7_MIN_INPUT_LENGTH):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.min_context_tokens = min_context_tokens
        self._bm25: Optional[_BM25] = None
        self._examples_cache = None

    # ---------- 文本辅助 ----------

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _init_bm25(self, examples: list):
        if self._examples_cache is examples and self._bm25 is not None:
            return
        corpus = [ex['input'] for ex in examples]
        self._bm25 = _BM25(corpus)
        self._examples_cache = examples
        logger.info(f"[Task7] BM25 构建完成: N={len(corpus)}, avgdl={self._bm25.avgdl:.1f}")

    @staticmethod
    def _compute_bm25_scores(examples: list, query: str) -> np.ndarray:
        corpus = [ex['input'] for ex in examples]
        bm25 = _BM25(corpus)
        return bm25.scores(query)

    # ---------- token 截断 ----------

    def _truncate_by_tokens(self, example_lines: list,
                            reverse_select: bool = True) -> tuple[str, int]:
        """Token 截断。

        reverse_select=True：从末尾开始选（优先保留高优先级 / 高相似度示例），
        然后恢复原始顺序拼接，保证最相关示例紧贴 test。
        """
        if reverse_select:
            selected: list = []
            total_tokens = 0
            for line in reversed(example_lines):
                line_tokens = self._count_tokens(line)
                if total_tokens + line_tokens > self.max_context_tokens:
                    break
                selected.append(line)
                total_tokens += line_tokens
            selected.reverse()
            return "".join(selected), total_tokens

        result = ""
        total_tokens = 0
        for line in example_lines:
            line_tokens = self._count_tokens(line)
            if total_tokens + line_tokens > self.max_context_tokens:
                break
            result += line
            total_tokens += line_tokens
        return result, total_tokens

    def _log_token_budget(self, n_examples: int, total_tokens: int,
                          strategy: str):
        compliance_tag = "✅" if total_tokens >= self.min_context_tokens else "⚠️<30K"
        msg = (f"[Task7] strategy={strategy} 截断后: {n_examples} 条 ({total_tokens} tokens "
               f"{compliance_tag}, 区间 [{self.min_context_tokens}, "
               f"{self.max_context_tokens}])")
        if total_tokens < self.min_context_tokens:
            logger.warning(f"[ICL 长度不足] {msg}")
        else:
            logger.info(msg)

    # ---------- Category 三级分类 ----------

    @staticmethod
    def _extract_category(input_text: str) -> str:
        m = re.search(r'Category:\s*(.+?)(?:\n|$)', input_text)
        return m.group(1).strip() if m else ""

    def _split_by_category(self, all_examples: list, test_input: str
                           ) -> tuple[list, list, list]:
        """按 Category 三级分类：精确匹配 / 关键词重叠 / 其他。

        Returns:
            (exact_match, fuzzy_examples, other) 三个 example 列表，
            其中 fuzzy_examples 已按重叠词数升序（少→多）。
        """
        test_category = self._extract_category(test_input)
        test_cat_words = (
            set(re.findall(r'[a-zA-Z]+', test_category.lower()))
            - CATEGORY_STOP_WORDS
        )

        exact_match: list = []
        fuzzy_match: list = []
        other: list = []

        for ex in all_examples:
            ex_category = self._extract_category(ex['input'])
            if ex_category.lower() == test_category.lower() and test_category:
                exact_match.append(ex)
            elif test_cat_words:
                ex_cat_words = (
                    set(re.findall(r'[a-zA-Z]+', ex_category.lower()))
                    - CATEGORY_STOP_WORDS
                )
                overlap = test_cat_words & ex_cat_words
                if overlap and len(overlap) >= 1:
                    fuzzy_match.append((ex, len(overlap)))
                else:
                    other.append(ex)
            else:
                other.append(ex)

        fuzzy_match.sort(key=lambda x: x[1])
        fuzzy_examples = [ex for ex, _ in fuzzy_match]

        logger.info(f"[Task7] Category='{test_category}': "
                    f"精确={len(exact_match)} 模糊={len(fuzzy_examples)} 其他={len(other)}")
        return exact_match, fuzzy_examples, other

    # ---------- 策略 A：Category 三级 + BM25 + 四层 recency ----------

    def _select_strategy_a(self, all_examples: list, test_input: str) -> str:
        exact_match, fuzzy_examples, other = self._split_by_category(
            all_examples, test_input)

        # Zone 1: 其他 Category — BM25 排序（低→高）
        if other:
            self._init_bm25(other)
            other_sims = self._bm25.scores(test_input)
            other_ranked = sorted(range(len(other)),
                                  key=lambda i: other_sims[i])
        else:
            other_ranked = []

        # Zone 2: fuzzy_match — 已按重叠词数排序（少→多）；多条时再按 BM25 升序
        if len(fuzzy_examples) > 1:
            fuzzy_sims = self._compute_bm25_scores(fuzzy_examples, test_input)
            fuzzy_ranked = sorted(range(len(fuzzy_examples)),
                                  key=lambda i: fuzzy_sims[i])
        else:
            fuzzy_ranked = list(range(len(fuzzy_examples)))

        # Zone 3: 精确匹配 — BM25 升序（最相似紧贴 test）
        if len(exact_match) > 1:
            exact_sims = self._compute_bm25_scores(exact_match, test_input)
            exact_ranked = sorted(range(len(exact_match)),
                                  key=lambda i: exact_sims[i])
        else:
            exact_ranked = list(range(len(exact_match)))

        example_lines: list = []
        for idx in other_ranked:
            example_lines.append(_format_example_line(other[idx]))
        for idx in fuzzy_ranked:
            example_lines.append(_format_example_line(fuzzy_examples[idx]))
        for idx in exact_ranked:
            example_lines.append(_format_example_line(exact_match[idx]))

        truncated, total_tokens = self._truncate_by_tokens(
            example_lines, reverse_select=True)
        n_lines = truncated.count('\n# ') + (1 if truncated else 0)
        self._log_token_budget(n_lines, total_tokens, 'A')
        return truncated

    # ---------- 策略 B：纯 Clue BM25 全量排序 ----------

    def _select_strategy_b(self, all_examples: list, test_input: str) -> str:
        self._init_bm25(all_examples)
        sims = self._bm25.scores(test_input)
        ranked = sorted(range(len(all_examples)), key=lambda i: sims[i])
        example_lines = [_format_example_line(all_examples[i]) for i in ranked]
        truncated, total_tokens = self._truncate_by_tokens(
            example_lines, reverse_select=True)
        n_lines = truncated.count('\n# ') + (1 if truncated else 0)
        self._log_token_budget(n_lines, total_tokens, 'B')
        return truncated

    # ---------- 策略 C：Category 三级 + 同 zone 随机打散 ----------

    def _select_strategy_c(self, all_examples: list, test_input: str) -> str:
        _random.seed(42)
        exact_match, fuzzy_examples, other = self._split_by_category(
            all_examples, test_input)

        if other:
            self._init_bm25(other)
            other_sims = self._bm25.scores(test_input)
            other_ranked = sorted(range(len(other)),
                                  key=lambda i: other_sims[i])
        else:
            other_ranked = []
        if len(fuzzy_examples) > 1:
            fuzzy_sims = self._compute_bm25_scores(fuzzy_examples, test_input)
            fuzzy_ranked = sorted(range(len(fuzzy_examples)),
                                  key=lambda i: fuzzy_sims[i])
        else:
            fuzzy_ranked = list(range(len(fuzzy_examples)))
        if len(exact_match) > 1:
            exact_sims = self._compute_bm25_scores(exact_match, test_input)
            exact_ranked = sorted(range(len(exact_match)),
                                  key=lambda i: exact_sims[i])
        else:
            exact_ranked = list(range(len(exact_match)))

        other_lines = [_format_example_line(other[i]) for i in other_ranked]
        fuzzy_lines = [_format_example_line(fuzzy_examples[i]) for i in fuzzy_ranked]
        exact_lines = [_format_example_line(exact_match[i]) for i in exact_ranked]

        # 同 zone 随机打散
        _random.shuffle(other_lines)
        _random.shuffle(fuzzy_lines)
        _random.shuffle(exact_lines)

        example_lines = other_lines + fuzzy_lines + exact_lines
        truncated, total_tokens = self._truncate_by_tokens(
            example_lines, reverse_select=True)
        n_lines = truncated.count('\n# ') + (1 if truncated else 0)
        self._log_token_budget(n_lines, total_tokens, 'C')
        return truncated

    # ---------- 主入口 ----------

    def select_by_strategy(self, strategy: str, all_examples: list,
                           test_input: str) -> str:
        """按策略 A/B/C 选择示例并返回拼接好的 examples_str。"""
        if strategy == 'A':
            return self._select_strategy_a(all_examples, test_input)
        if strategy == 'B':
            return self._select_strategy_b(all_examples, test_input)
        if strategy == 'C':
            return self._select_strategy_c(all_examples, test_input)
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# Prompt 构建（三种 variant：A 直接 / B Plan / C ReAct）
# ============================================================

_TASK7_ROLE_BLOCK = (
    "### Role\n"
    "You are an expert Jeopardy contestant with vast knowledge "
    "across history, science, literature, geography, pop culture, "
    "and wordplay.\n\n"
)


def _task7_rules_block(variant: str, category: str) -> str:
    cat_quoted = f'"{category}"'
    if variant == 'C':
        return (
            "### Game & Rules\n"
            "You are playing the Jeopardy! game show. In Jeopardy, you are given "
            "a Category and a Clue, and you must provide the correct answer.\n\n"
            "1. The Category is a CRITICAL constraint \u2014 your answer MUST satisfy it.\n"
            f"2. Current Category: {cat_quoted}\n"
            "3. If the Category contains quoted text (e.g. '\"H\" NAMES', "
            "'\"AW\" SHUCKS'), the answer MUST start with or contain that quoted "
            "letter/word. Verify this BEFORE finalizing.\n"
            "4. Use BOTH the provided examples AND your own world knowledge.\n"
            "5. Answers are typically 1-5 words, lowercase, as CONCISE as possible.\n\n"
        )
    return (
        "### Game & Rules\n"
        "You are playing the Jeopardy! game show. In Jeopardy, you are given "
        "a Category and a Clue, and you must provide the answer (a word or phrase).\n\n"
        "1. The Category is a CRITICAL constraint \u2014 your answer MUST satisfy it.\n"
        f"2. Current Category: {cat_quoted} \u2014 think carefully about what "
        "type of answer this demands.\n"
        "3. If the Category contains quoted text (e.g. '\"H\" NAMES'), "
        "the answer MUST start with or contain that quoted letter/word.\n"
        "4. Use BOTH the provided examples AND your own world knowledge.\n"
        "5. Answers must be lowercase, 1-5 words, matching example style.\n\n"
    )


def _task7_instruction(variant: str, category: str) -> str:
    if variant == 'B':
        return (
            "Use the PLAN approach:\n"
            f"  PLAN:\n"
            f"    P1: What TYPE of answer does Category \"{category}\" demand? "
            "(person / place / thing / phrase / letter pattern)\n"
            "    P2: What are the 2-3 most diagnostic facts in the Clue?\n"
            "    P3: What knowledge domain is this question in?\n"
            "    P4: What are 1-2 candidate answers that fit the Category AND match the Clue facts?\n"
            "  EXECUTE: Follow your plan \u2014 evaluate each candidate, eliminate those "
            "that violate the Category constraint or Clue facts, and select the best one.\n"
            "Then output ONLY your final answer wrapped in <label> tags."
        )
    if variant == 'C':
        return (
            "Use the ReAct approach (Reason \u2192 Act \u2192 Observe, repeat until confident):\n"
            f"  Thought 1: What does Category \"{category}\" constrain? "
            "What type of answer is expected?\n"
            "  Act 1: Retrieve relevant knowledge about the key entity/event in the Clue.\n"
            "  Observation 1: What do I know? Which facts match the Clue's description?\n"
            "  Thought 2: Does my candidate satisfy the Category constraint? "
            "Is there a more precise or concise form?\n"
            "  Act 2: Verify \u2014 does this answer fit ALL constraints (Category + Clue facts)?\n"
            "  Observation 2: Confirm or refine.\n"
            "Then output ONLY your final answer wrapped in <label> tags."
        )
    return (
        "Think carefully and thoroughly about the Category constraint and all "
        "key facts in the Clue. Consider multiple possibilities before settling "
        "on the most accurate answer. "
        "Then output ONLY your final answer wrapped in <label> tags."
    )


def build_task7_prompt(task_description: str, text2annotate: str,
                       examples_str: str,
                       variant: str = 'A') -> str:
    """Task 7 Jeopardy 知识问答 Prompt（A 直接 / B Plan / C ReAct）。"""
    cat_match = re.search(r'Category:\s*(.+?)(?:\n|$)', text2annotate)
    category = cat_match.group(1).strip() if cat_match else "Unknown"

    return (
        _TASK7_ROLE_BLOCK
        + f"### Task\n{task_description}\n\n"
        + _task7_rules_block(variant, category)
        + f"### Examples\n{examples_str}\n\n"
        + f"### Input\n{text2annotate}\n\n"
        + _task7_instruction(variant, category)
    )


def build_verifier_prompt(category: str, clue_text: str,
                          candidates: list,
                          examples_str: str = '') -> str:
    """Task 7 验证器 Prompt：从 2-3 个去重候选中选最优答案。

    Args:
        category: Jeopardy Category 文本。
        clue_text: Jeopardy Clue 文本。
        candidates: 去重后的候选列表（2-3 个）。
        examples_str: ICL 示例文本块（与主路同构，30K-31K tokens）。
            作为同类问答风格的参考上下文，与赛题 30K ICL 硬约束保持一致；
            模型仍按 Evaluation Rules 在候选中裁决，不要求基于示例改写候选。
    """
    labels = ['A', 'B', 'C']
    n = len(candidates)
    candidates_str = "\n".join(f"{labels[i]}: {c}" for i, c in enumerate(candidates))
    count_desc = f"{n} candidate answer{'s' if n > 1 else ''}"
    examples_block = (
        f"### Examples\n{examples_str}\n\n" if examples_str else ""
    )
    return (
        "### Role\n"
        "You are a Jeopardy expert judge with encyclopedic knowledge.\n"
        f"Your job is to evaluate {count_desc} and determine "
        "the most correct one.\n\n"

        "### Evaluation Rules\n"
        "1. The Category is a CRITICAL constraint \u2014 the correct answer MUST satisfy it.\n"
        "2. If the Category contains quoted text (e.g. '\"H\" NAMES', '\"AW\" SHUCKS'), "
        "the answer MUST start with or contain that quoted letter/word \u2014 "
        "eliminate any candidate that violates this.\n"
        "3. Cross-check each candidate against the key facts in the Clue.\n"
        "4. Prefer the most CONCISE correct form \u2014 Jeopardy answers are typically "
        "1-5 words. Do NOT include titles (Lord, Sir, Dr., University of ...) or extra "
        "qualifiers unless the Clue explicitly requires them. "
        "A shorter answer that is factually correct beats a longer over-specified one.\n"
        "5. BE FAIR AND IMPARTIAL: evaluate each candidate solely on its factual accuracy "
        "against the Category and Clue \u2014 let the facts decide, not the order listed.\n\n"

        + examples_block +

        f"### Question\n"
        f"Category: {category}\n"
        f"Clue: {clue_text}\n\n"

        f"### Candidate Answers\n"
        f"{candidates_str}\n\n"

        "Briefly reason which candidate best satisfies both the Category constraint "
        "and the Clue\u2019s key facts. "
        "Output ONLY the exact answer text (the actual word/phrase, not just 'A'/'B'/'C') "
        "wrapped in <label> tags."
    )


def build_task7_retry_prompt(task_description: str, text2annotate: str,
                             examples_str: str,
                             excluded: list,
                             exclude_reason: str) -> str:
    """重调用 Prompt：在 build_task7_prompt(A) 基础上插入排除段。

    exclude_reason: ``clue_trap`` / ``category_violation``
    """
    cat_match = re.search(r'Category:\s*(.+?)(?:\n|$)', text2annotate)
    category = cat_match.group(1).strip() if cat_match else "Unknown"

    rules = (
        "### Game & Rules\n"
        "You are playing the Jeopardy! game show. Given a Category and a Clue, "
        "you must provide the correct answer (a word or phrase).\n\n"
        "1. The Category is a CRITICAL constraint — your answer MUST satisfy it.\n"
        f"2. Current Category: \"{category}\" — analyze what it demands.\n"
        "3. If the Category contains quoted text (e.g. '\"H\" NAMES'), "
        "the answer MUST start with that quoted letter/word.\n"
        "4. The answer MUST NOT be any word/phrase that already appears in the Clue.\n"
        "5. Use BOTH the provided examples AND your own world knowledge.\n"
        "6. Answers are lowercase, 1-5 words, matching example style.\n\n"
    )

    excluded_clean = [e for e in excluded if e]
    if excluded_clean:
        excluded_lines = "\n".join(f"  - {e}" for e in excluded_clean)
    else:
        excluded_lines = "  (none)"

    reason_map = {
        'clue_trap': (
            "These answers were REJECTED because they appear verbatim in the Clue "
            "itself (which is not allowed — the answer must be a fresh term)."
        ),
        'category_violation': (
            "These answers were REJECTED because they VIOLATE the Category "
            "constraint (e.g. they do not start with the required letter/word "
            "specified by the Category)."
        ),
    }
    reason_text = reason_map.get(exclude_reason, "These answers were rejected.")

    exclusion_sec = (
        "### Previously Rejected Answers (do NOT repeat any of these)\n"
        f"{excluded_lines}\n\n"
        f"Reason: {reason_text}\n"
        "Provide a DIFFERENT answer that satisfies BOTH the Category constraint "
        "AND the Clue's facts.\n\n"
    )

    instruction = (
        "Think carefully about the Category constraint and the key facts in the "
        "Clue. Explicitly avoid the rejected answers above. "
        "Then output ONLY your final answer wrapped in <label> tags."
    )

    return (
        _TASK7_ROLE_BLOCK
        + f"### Task\n{task_description}\n\n"
        + rules
        + f"### Examples\n{examples_str}\n\n"
        + f"### Input\n{text2annotate}\n\n"
        + exclusion_sec
        + instruction
    )


# ============================================================
# 硬过滤：ClueTrap / Category 字母约束
# ============================================================

def candidate_in_clue_trap(candidate: str, input_text: str) -> bool:
    """候选是否作为整词/整短语出现在 Clue 中。

    规则：
    - 只检查 Clue 段（不检查 Category，避免误伤）
    - 候选长度 >= 3 字符（避免 ``of`` / ``the`` 等触发）
    - word-boundary 整短语匹配，大小写不敏感
    """
    if not candidate or not input_text:
        return False
    cand = candidate.strip().strip('\'"`').lower()
    if len(cand) < 3:
        return False
    m = re.search(r'Clue\s*:\s*(.+)$', input_text, re.DOTALL | re.IGNORECASE)
    clue = m.group(1) if m else input_text
    clue_low = clue.lower()
    escaped = re.escape(cand)
    escaped = re.sub(r'\\\ ', r'\\s+', escaped)
    pattern = r'(?<![A-Za-z0-9])' + escaped + r'(?![A-Za-z0-9])'
    return re.search(pattern, clue_low) is not None


def parse_category_letter_constraint(category: str) -> Optional[str]:
    """解析 Category 中的引号字母/短语约束。

    典型模式：``"H" NAMES`` / ``"AW" SHUCKS`` / ``"S" WORDS``
    返回小写的引号内字符串（长度 1-5）；无法识别返回 None。

    严格条件：
    - 引号内必须是字母串（1-5 字符）
    - 仅抓取 Category 中首个合法引号字段
    """
    if not category:
        return None
    m = re.search(r'["\u201c\u201d\']([A-Za-z]{1,5})["\u201c\u201d\']', category)
    if m:
        return m.group(1).lower()
    return None


def candidate_satisfies_letter(candidate: str, letter: str) -> bool:
    """候选是否满足首词以 ``letter`` 开头（忽略 a/an/the 前导冠词）。

    - letter 长度 1：候选首个 alphabetic 字符必须等于 letter
    - letter 长度 >=2：候选必须以 letter 整串开头
    - 允许前导 ``a `` / ``an `` / ``the `` 被忽略
    """
    if not candidate or not letter:
        return True
    cand = candidate.strip().lower()
    for art in ('the ', 'an ', 'a '):
        if cand.startswith(art):
            cand = cand[len(art):].strip()
            break
    if not cand:
        return False
    letter = letter.lower()
    if len(letter) == 1:
        for ch in cand:
            if ch.isalpha():
                return ch == letter
        return False
    return cand.startswith(letter)


# ============================================================
# 答案后处理 / 校验 / 抢救
# ============================================================

def postprocess_task7(prediction: str) -> str:
    """Task 7 答案归一化。

    Jeopardy 答案格式：全小写，1-5 词，通常是专有名词/简洁概念。
    - NFD 分解后去掉组合变音符（galápagos → galapagos）
    - 去除尾部标点
    - 不去除前导冠词（GT 中部分答案含冠词如 ``the plague``）
    """
    if not prediction:
        return prediction
    result = prediction.strip().lower()
    result = unicodedata.normalize('NFD', result)
    result = ''.join(c for c in result if unicodedata.category(c) != 'Mn')
    result = result.rstrip('.,;:!?')
    return result.strip()


def validate_prediction(prediction: Optional[str]) -> Optional[str]:
    """校验 Task 7 generation 类型 prediction 的合理性。"""
    if prediction is None:
        return None
    prediction = prediction.strip()
    if not prediction:
        return None
    if prediction in ('...', '…', '..', '…………', 'N/A', 'n/a', 'None', 'none', 'null'):
        return None
    if all(c in '.…' for c in prediction):
        return None
    if len(prediction) > 500:
        logger.warning(f"[Task7] 答案过长 ({len(prediction)} chars): {prediction[:100]}...")
        return None
    return prediction


def rescue_from_thinking(raw: Optional[str]) -> Optional[str]:
    """从思考内容中抢救答案。

    当 thinking 模式 content 为空但 prediction 已经走过 fallback 拼装后，
    再次匹配 ``the answer is X`` 等口语化结尾，作为最后兜底。
    保持小写，要求 1-5 词且非单字符。
    """
    if not raw:
        return None
    m = re.search(
        r'(?:the answer is|answer would be|answer should be|answer:)\s*["\']?([^"\'\.\n,]+)',
        raw, re.IGNORECASE)
    if m:
        rescued = m.group(1).strip().lower()
        if 1 <= len(rescued.split()) <= 5 and len(rescued) > 1:
            return rescued
    return None


# ============================================================
# 候选去重 / 多数投票
# ============================================================

def dedupe_candidates(pool: list) -> list:
    """词边界子串去重：``pandas`` ⊂ ``giant pandas`` → 合并保留短的；
    ``possum`` 不是 ``opossum`` 的 word-boundary 子串 → 不合并。

    - 完全相同 → 合并
    - existing 是 ans 的完整单词子串 → 保留 existing（短的）
    - ans 是 existing 的完整单词子串 → 用 ans 替换 existing（保留短的）
    """
    def _is_word_substr(short: str, long_: str) -> bool:
        return bool(re.search(r'\b' + re.escape(short) + r'\b', long_))

    deduped: list = []
    for ans in pool:
        merged = False
        for j, existing in enumerate(deduped):
            if ans == existing:
                merged = True
                break
            if _is_word_substr(existing, ans):
                merged = True
                break
            if _is_word_substr(ans, existing):
                deduped[j] = ans
                merged = True
                break
        if not merged:
            deduped.append(ans)
    return deduped


def majority_vote(predictions: list) -> Optional[str]:
    """多数票投票。"""
    valid = [p for p in predictions if p]
    if not valid:
        return None
    counter = Counter(valid)
    winner, count = counter.most_common(1)[0]
    logger.info(f"[Task7] 投票 {len(valid)} 候选 分布={dict(counter)} "
                f"胜出='{winner[:60]}' ({count} 票)")
    return winner


__all__ = [
    'TASK7_MAX_TOKENS', 'TASK7_TIMEOUT', 'TASK7_ENABLE_THINKING',
    'TASK7_REP_PENALTY', 'TASK7_TEMPERATURE',
    'TASK7_MAX_INPUT_LENGTH', 'TASK7_MIN_INPUT_LENGTH',
    'TASK7_VERIFIER_MAX_TOKENS',
    'TASK7_NOTHINK_MAX_TOKENS', 'TASK7_NOTHINK_TIMEOUT',
    'TASK7_STRATEGIES',
    'Task7ExampleSelector', '_format_example_line',
    'build_task7_prompt', 'build_verifier_prompt', 'build_task7_retry_prompt',
    'candidate_in_clue_trap',
    'parse_category_letter_constraint', 'candidate_satisfies_letter',
    'postprocess_task7', 'validate_prediction', 'rescue_from_thinking',
    'dedupe_candidates', 'majority_vote',
]
