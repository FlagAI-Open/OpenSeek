"""method.py — Task 6 自包含模块

Task 6: openseek-6_mnli_same_genre_classification
  输入：一对英文句子 + ICL 示例池
  输出：单字符 Y / N（两句是否属于同一 genre）

模块组成：
  * _BM25：基于词袋的相似度检索
  * Task6ExampleSelector：6 轴正交示例选择器
      A subset_mode    : bm25_full / genre_filter / bm25_plus_genre
      B label_balance  : natural / 5050 / boost_y / boost_n
      C order          : recency_up / recency_down / interleaved
      D prompt_style   : neutral / strict_y / strict_n / inprompt_cot / linguist_cot
      E cot_inject     : off / on（on 时启用 cot_only 池过滤）
      F think_inject   : off / on（自动要求 cot_inject=True）
  * build_task6_prompt：5 种 prompt_style 模板
  * validate_task6_prediction：白名单 genre 校正 + 多级 fallback 抽 Y/N
  * majority_vote：3 轮平权硬投票（平票取 N，全 null 兜底 N）
  * TASK6_MULTI_VIEW_CONFIGS：3 轮（R0+R2+R4）配置表
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Task 6 标签集 + Genre 白名单
# ============================================================

TASK6_LABEL_SET = ['N', 'Y']

# 官方任务定义中声明的 10 类 genre
TASK6_GENRE_SET = frozenset([
    'face-to-face', 'government', 'letters', '9/11', 'slate',
    'fiction', 'telephone', 'travel', 'oup', 'verbatim',
])

TASK6_GENRE_WHITELIST_ORDER = [
    'face-to-face', 'government', 'letters', '9/11', 'slate',
    'fiction', 'telephone', 'travel', 'oup', 'verbatim',
]


# ============================================================
# 推理参数
# ============================================================

TASK6_MAX_TOKENS = 2048
TASK6_TIMEOUT = 600                   # 单次请求超时
TASK6_TEMPERATURE = 0.0               # 确定性输出
TASK6_REPETITION_PENALTY = 1.1        # 抑制 thinking 模式下死循环复读
TASK6_MAX_INPUT_LENGTH = 31000        # 单条 ICL prompt 的 token 上限（赛题硬约束 ≥ 30K，留 1K 余量）
TASK6_MIN_INPUT_LENGTH = 30000        # 单条 ICL prompt 的 token 下限（赛题硬约束 ≥ 30K，必须达标）


# ============================================================
# 3 轮正交多视角配置（R0 + R2 + R4，全 api_thinking=True）
# ============================================================

TASK6_N_ROUNDS = 3

TASK6_MULTI_VIEW_CONFIGS = [
    # R0：中性锚轮（0 注入），保证下限稳定
    {
        'subset_mode':   'bm25_full',
        'label_balance': 'natural',
        'order':         'recency_up',
        'prompt_style':  'neutral',
        'cot_inject':    False,
        'think_inject':  False,
        'api_thinking':  True,
    },
    # R1：strict_n + cot/think 注入（保守压 Y）
    {
        'subset_mode':   'bm25_full',
        'label_balance': 'natural',
        'order':         'recency_up',
        'prompt_style':  'strict_n',
        'cot_inject':    True,
        'think_inject':  True,
        'api_thinking':  True,
    },
    # R2：极端保守（genre_filter + boost_n + interleaved + strict_n + cot/think 注入）
    {
        'subset_mode':   'genre_filter',
        'label_balance': 'boost_n',
        'order':         'interleaved',
        'prompt_style':  'strict_n',
        'cot_inject':    True,
        'think_inject':  True,
        'api_thinking':  True,
    },
]

assert len(TASK6_MULTI_VIEW_CONFIGS) == TASK6_N_ROUNDS, (
    f"TASK6_MULTI_VIEW_CONFIGS 长度必须等于 TASK6_N_ROUNDS={TASK6_N_ROUNDS}, "
    f"实际 {len(TASK6_MULTI_VIEW_CONFIGS)}"
)


# ============================================================
# BM25 检索模型
# ============================================================

class _BM25:
    """BM25 检索模型，用于 Task 6 输入文本相似度计算。"""

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
# 6 轴示例选择器
# ============================================================

class Task6ExampleSelector:
    """Task 6 6 轴正交示例选择器（A subset / B balance / C order / E cot / F think）。"""

    def __init__(self, tokenizer,
                 max_context_tokens: int = TASK6_MAX_INPUT_LENGTH,
                 min_context_tokens: int = TASK6_MIN_INPUT_LENGTH):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.min_context_tokens = min_context_tokens
        self._examples_cache = None
        self._bm25: Optional[_BM25] = None

    # ---------- 文本辅助 ----------

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _extract_sentences_text(input_text: str) -> str:
        """提取 s1 + s2 纯文本（去掉 'Sentence 1/2:' 与 'Genre:' 结构标签）。"""
        s1 = s2 = ''
        m1 = re.search(r'Sentence 1:\s*(.+?)(?=Sentence 2:|Genre:|$)', input_text,
                       re.DOTALL | re.IGNORECASE)
        if m1:
            s1 = m1.group(1).strip()
        m2 = re.search(r'Sentence 2:\s*(.+?)(?=Genre:|$)', input_text,
                       re.DOTALL | re.IGNORECASE)
        if m2:
            s2 = m2.group(1).strip()
        return (s1 + ' ' + s2).strip() or input_text

    @staticmethod
    def _extract_genre(input_text: str) -> str:
        """从 input 末尾提取 Genre 值，如 'Genre: government.' -> 'government'。"""
        m = re.search(r'Genre:\s*([\w/\-]+)\.?\s*$', input_text, re.IGNORECASE)
        return m.group(1).strip().lower() if m else ''

    def _init_pools(self, all_examples: list):
        """Pre-cache BM25（只初始化一次）。"""
        if self._examples_cache is all_examples:
            return
        corpus = [self._extract_sentences_text(ex['input']) for ex in all_examples]
        self._bm25 = _BM25(corpus)
        self._examples_cache = all_examples
        logger.info(f"[Task6] BM25 构建完成: N={len(corpus)}, avgdl={self._bm25.avgdl:.1f}")

    # ---------- token 截断 ----------

    def _truncate_by_tokens(self, example_lines: list,
                            reverse_select: bool = True) -> tuple[str, int]:
        """Token 截断。reverse_select=True 时从末尾开始选（优先保留高相似度示例）。"""
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
                          round_idx: int, mode: str, label_info: str):
        compliance_tag = "✅" if total_tokens >= self.min_context_tokens else "⚠️<30K"
        msg = (f"[Task6] R{round_idx} 截断后: {n_examples} 条 ({total_tokens} tokens "
               f"{compliance_tag}, 区间 [{self.min_context_tokens}, "
               f"{self.max_context_tokens}], {mode}) 标签分布: {label_info}")
        if total_tokens < self.min_context_tokens:
            logger.warning(f"[ICL 长度不足] {msg}")
        else:
            logger.info(msg)

    # ---------- A 轴：subset_mode ----------

    def _apply_subset(self, all_examples: list, ranked_desc: list,
                      subset_mode: str, stated_genre: str) -> list:
        """输入 BM25 降序全池索引 ranked_desc，返回过滤/扩展后的索引列表。"""
        if subset_mode == 'bm25_full':
            return list(ranked_desc)

        sg = (stated_genre or '').strip().lower()

        if subset_mode == 'genre_filter':
            if not sg:
                return list(ranked_desc)
            same = [i for i in ranked_desc
                    if self._extract_genre(all_examples[i]['input']) == sg]
            if len(same) < 300:
                logger.info(f"[Task6] genre_filter 回落: 同 genre({sg}) 仅 "
                            f"{len(same)} 条 (<300)，回落 bm25_full")
                return list(ranked_desc)
            return same

        if subset_mode == 'bm25_plus_genre':
            if not sg:
                return list(ranked_desc)
            same = [i for i in ranked_desc
                    if self._extract_genre(all_examples[i]['input']) == sg]
            same_set = set(same)
            diff = [i for i in ranked_desc if i not in same_set]
            return diff + same

        return list(ranked_desc)

    # ---------- E 轴辅助：cot_only 过滤 ----------

    @staticmethod
    def _filter_cot_nonempty(pool_indices: list, all_examples: list) -> list:
        """仅保留 cot 字段非空的示例索引，保持相对顺序。"""
        return [i for i in pool_indices
                if (all_examples[i].get('cot') or '').strip()]

    # ---------- B 轴：label_balance ----------

    @staticmethod
    def _ex_label(ex: dict) -> str:
        out = ex['output']
        return (out[0] if isinstance(out, list) else out).strip().upper()

    def _apply_label_balance(self, pool_indices: list, all_examples: list,
                             balance: str) -> list:
        if balance == 'natural':
            return list(pool_indices)

        n_list, y_list = [], []
        for i in pool_indices:
            if self._ex_label(all_examples[i]) == 'Y':
                y_list.append(i)
            else:
                n_list.append(i)

        if balance == '5050':
            merged: list = []
            for a, b in zip(n_list, y_list):
                merged.extend([a, b])
            merged.extend(n_list[len(y_list):] if len(n_list) > len(y_list)
                          else y_list[len(n_list):])
            return merged

        if balance == 'boost_y':
            return self._ratio_merge(n_list, y_list, 2, 3)

        if balance == 'boost_n':
            return self._ratio_merge(n_list, y_list, 3, 2)

        return list(pool_indices)

    @staticmethod
    def _ratio_merge(n_list: list, y_list: list, nk: int, yk: int) -> list:
        ni, yi = 0, 0
        merged: list = []
        while ni < len(n_list) or yi < len(y_list):
            for _ in range(nk):
                if ni < len(n_list):
                    merged.append(n_list[ni]); ni += 1
            for _ in range(yk):
                if yi < len(y_list):
                    merged.append(y_list[yi]); yi += 1
        return merged

    # ---------- C 轴：order ----------

    def _apply_order(self, pool_indices: list, all_examples: list,
                     order: str) -> list:
        if order == 'recency_down':
            return list(pool_indices)
        if order == 'interleaved':
            n_list, y_list = [], []
            for i in pool_indices:
                if self._ex_label(all_examples[i]) == 'Y':
                    y_list.append(i)
                else:
                    n_list.append(i)
            merged: list = []
            for a, b in zip(n_list, y_list):
                merged.extend([a, b])
            merged.extend(n_list[len(y_list):] if len(n_list) > len(y_list)
                          else y_list[len(n_list):])
            return merged
        # recency_up（默认）：BM25 降序 -> 反转，最相似放末尾
        return list(reversed(pool_indices))

    # ---------- E/F 轴：示例行格式 ----------

    @staticmethod
    def _format_example_line(ex: dict, cot_inject: bool, think_inject: bool) -> str:
        output_text = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
        if cot_inject:
            cot = (ex.get('cot') or '').strip()
            think = (ex.get('think') or '').strip()
            if think_inject and think and cot:
                return f"# {ex['input']}\n{think}\n{cot}\n<label>{output_text}</label>\n"
            if cot:
                return f"# {ex['input']}\n{cot}\n<label>{output_text}</label>\n"
            return f"# {ex['input']} <label>{output_text}</label>\n"
        return f"# {ex['input']} <label>{output_text}</label>\n"

    # ---------- 主入口 ----------

    def select(self, all_examples: list, test_input: str,
               round_idx: int = 0,
               subset_mode: str = 'bm25_full',
               label_balance: str = 'natural',
               order: str = 'recency_up',
               cot_inject: bool = False,
               think_inject: bool = False) -> str:
        """6 轴示例选择主入口，返回拼接好的 examples_str。"""
        self._init_pools(all_examples)

        # think_inject=True 强制 cot_inject=True（格式上依赖 cot 锚点）
        if think_inject and not cot_inject:
            logger.warning(f"[Task6] R{round_idx} think_inject=True 强制 cot_inject=True")
            cot_inject = True

        stated_genre = self._extract_genre(test_input)
        query_text = self._extract_sentences_text(test_input)
        sims = self._bm25.scores(query_text)

        # 全池 BM25 降序
        ranked_desc = sorted(range(len(all_examples)),
                             key=lambda i: sims[i], reverse=True)

        # A 轴
        pool_indices = self._apply_subset(all_examples, ranked_desc,
                                          subset_mode, stated_genre)

        # E 轴的 cot_only 过滤（注入 cot/think 时生效）
        if cot_inject:
            pre_n = len(pool_indices)
            pool_indices = self._filter_cot_nonempty(pool_indices, all_examples)
            logger.info(f"[Task6] R{round_idx} cot_only 过滤: "
                        f"{pre_n} -> {len(pool_indices)} 条（cot 非空）")

        # B 轴
        balanced = self._apply_label_balance(pool_indices, all_examples,
                                             label_balance)
        # C 轴
        ordered = self._apply_order(balanced, all_examples, order)

        logger.info(f"[Task6] R{round_idx}: subset={subset_mode}|{len(pool_indices)} "
                    f"balance={label_balance}|{len(balanced)} order={order} "
                    f"cot_inject={cot_inject} think_inject={think_inject}")

        # E/F 轴：示例行格式
        example_lines = [self._format_example_line(all_examples[idx],
                                                   cot_inject, think_inject)
                         for idx in ordered]

        # Token 截断
        reverse_select = (order == 'recency_up')
        truncated_str, total_tokens = self._truncate_by_tokens(
            example_lines, reverse_select=reverse_select)

        # 截断后统计
        trunc_labels: dict = defaultdict(int)
        for line in truncated_str.split('\n'):
            m = re.search(r'<label>(\w+)</label>', line)
            if m:
                trunc_labels[m.group(1)] += 1
        label_info = ', '.join(f'{k}:{v}' for k, v in sorted(trunc_labels.items()))
        total_trunc = sum(trunc_labels.values())
        mode = "reverse_select" if reverse_select else "forward_select"
        self._log_token_budget(total_trunc, total_tokens, round_idx, mode, label_info)

        return truncated_str


# ============================================================
# 答案校验：白名单 genre 校正 + 多级 fallback
# ============================================================

def _correct_by_genre_logic(raw: str, stated_genre: str) -> Optional[str]:
    """白名单版双向 genre 校正：
    - 仅在 s1/s2 genre **均落在 10 类白名单内**时才触发校正
        * 两者 == stated genre -> 强制 Y
        * 任一 != stated genre -> 强制 N
    - 任一 genre 落在白名单之外 -> 返回 None，交由 <label> 原判托底
    - 未解析到结构化 genre -> 返回 None
    """
    if not stated_genre:
        return None
    sg = stated_genre.strip().lower()

    # 主格式：sentence1 genre: xxx / sentence2 genre: xxx
    s1_match = re.search(r'sentence1 genre:\s*([\w/\-]+)', raw, re.IGNORECASE)
    s2_match = re.search(r'sentence2 genre:\s*([\w/\-]+)', raw, re.IGNORECASE)
    if s1_match and s2_match:
        s1_genre = s1_match.group(1).strip().lower()
        s2_genre = s2_match.group(1).strip().lower()
        if s1_genre not in TASK6_GENRE_SET or s2_genre not in TASK6_GENRE_SET:
            logger.info(f"[Task6] genre 校正跳过(白名单外): stated='{sg}' "
                        f"S1='{s1_genre}' S2='{s2_genre}' -> 交由 <label> 解析")
            return None
        if s1_genre == sg and s2_genre == sg:
            logger.info(f"[Task6] genre 双向校正: stated='{sg}' "
                        f"S1='{s1_genre}' S2='{s2_genre}' -> 强制 Y")
            return 'Y'
        logger.info(f"[Task6] genre 双向校正: stated='{sg}' "
                    f"S1='{s1_genre}' S2='{s2_genre}' -> 强制 N")
        return 'N'

    # 备选格式：S1: xxx [v/x/✓/✗]? | S2: xxx
    m = re.search(
        r'S1:\s*([\w/\-]+)\s*[\u2713\u2717vx]?\s*\|.*?S2:\s*([\w/\-]+)',
        raw, re.IGNORECASE,
    )
    if m:
        s1_genre = m.group(1).strip().lower()
        s2_genre = m.group(2).strip().lower()
        if s1_genre not in TASK6_GENRE_SET or s2_genre not in TASK6_GENRE_SET:
            return None
        if s1_genre == sg and s2_genre == sg:
            return 'Y'
        return 'N'

    return None


def validate_task6_prediction(raw: Optional[str],
                              stated_genre: str = '') -> Optional[str]:
    """从模型输出中提取 Y/N 标签（白名单 genre 校正 + 多级 fallback）。"""
    if not raw or raw.strip() in ('None', 'none', '...', ''):
        return None

    # genre 逻辑校正（优先级最高，但仅白名单内触发）
    if stated_genre:
        corrected = _correct_by_genre_logic(raw, stated_genre)
        if corrected is not None:
            return corrected

    # 解析 <label>
    m = re.search(r'<label>\s*([YN])\s*</label>', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
    if lines and lines[-1].upper() in ('Y', 'N'):
        return lines[-1].upper()

    for line in reversed(lines):
        m2 = re.search(r'\b([YN])\b', line, re.IGNORECASE)
        if m2:
            return m2.group(1).upper()

    logger.warning(f"[Task6] 答案解析失败: {raw[:100]}")
    return None


def majority_vote(predictions: list) -> Optional[str]:
    """多数投票，平局取 N（保守）。"""
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None
    counter = Counter(valid)
    most_common = counter.most_common()
    if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
        logger.warning(f"[Task6] 投票平局: {dict(counter)}，取 N（保守）")
        return 'N'
    return most_common[0][0]


# ============================================================
# Prompt 构建（5 种 prompt_style）
# ============================================================

_GENRE_REF_BLOCK = (
    "### Genre Reference\n"
    "- government: formal bureaucratic tone, agencies (GAO, EPA), regulations, federal programs\n"
    "- fiction: narrative storytelling, character names, dialogue tags (said, asked), emotional descriptions\n"
    "- telephone: spoken markers (yeah, uh, um, you know, gonna), conversational fragments, informal\n"
    "- travel: place descriptions, landmarks, tourism (island, temple, museum, beach), sightseeing\n"
    "- slate: opinion/analysis, journalistic commentary, cultural criticism, named public figures\n"
    "- 9/11: references to attacks, hijacking, FAA, NORAD, Pentagon, bin Laden, al Qaeda\n"
    "- oup: academic research, child development, textile/fabric industry, cognitive/social science\n"
    "- verbatim: linguistics discussion, word origins, etymology, language usage, dictionary\n"
    "- face-to-face: in-person conversation (yeah, okay, right, so, like), similar to telephone\n"
    "- letters: philanthropic appeals, donation requests, dear [recipient], fundraising, charity\n\n"
)

_WHITELIST_BLOCK = (
    "### Genre Whitelist (STRICT)\n"
    "sentence1 genre and sentence2 genre MUST be EXACTLY one of the following 10 values:\n"
    "  face-to-face, government, letters, 9/11, slate, fiction, telephone, travel, oup, verbatim\n"
    "DO NOT invent new genre names such as 'fashion', 'history', 'personal', 'opinion',\n"
    "'narrative', 'conversation', 'academic', 'regulation', 'tourism', 'linguistics'.\n"
    "If a sentence seems ambiguous, pick the CLOSEST matching value from the 10 genres above.\n\n"
)

_OUTPUT_BLOCK = (
    "### Output Format\n"
    "Output EXACTLY four lines, no explanation:\n"
    "stated genre: [one of the 10 genres]\n"
    "sentence1 genre: [one of the 10 genres]\n"
    "sentence2 genre: [one of the 10 genres]\n"
    "<label>N or Y</label>\n"
)


def _build_rules(prompt_style: str, stated_genre: str) -> str:
    genre_hint = f" ({stated_genre})" if stated_genre else ""
    base_y = f"- For label Y: sentence1 genre AND sentence2 genre must both equal the stated genre{genre_hint}\n"
    base_n = f"- For label N: at least one genre does NOT equal the stated genre{genre_hint}\n"

    if prompt_style == 'strict_y':
        return (
            "### Rules\n"
            "- Use the genre name that BEST describes each sentence\n"
            + base_y + base_n +
            f"- Label Y bias: if BOTH sentences clearly exhibit markers of '{stated_genre or 'the stated genre'}', "
            "do NOT over-reject on minor stylistic variation — choose Y\n"
            "- Only mark N when you have CONCRETE evidence that a sentence belongs to a DIFFERENT genre\n\n"
        )
    if prompt_style == 'strict_n':
        return (
            "### Rules\n"
            "- Use the genre name that BEST describes each sentence\n"
            + base_y + base_n +
            "- Label N bias: if EITHER sentence shows markers of ANOTHER genre (even mild), label N\n"
            f"- Only mark Y when BOTH sentences strongly and unambiguously match '{stated_genre or 'the stated genre'}'\n\n"
        )
    return (
        "### Rules\n"
        "- Use the genre name that BEST describes each sentence\n"
        + base_y + base_n + "\n"
    )


def _build_instruction(prompt_style: str, stated_genre: str) -> str:
    genre_hint = f" ({stated_genre})" if stated_genre else ""
    if prompt_style == 'inprompt_cot':
        return (
            "Analyze step by step (keep each step one short sentence):\n"
            "  Step 1: list 1-2 KEY linguistic markers in sentence1.\n"
            "  Step 2: decide sentence1 genre (must be from the 10-genre whitelist).\n"
            "  Step 3: list 1-2 KEY linguistic markers in sentence2.\n"
            "  Step 4: decide sentence2 genre (must be from the 10-genre whitelist).\n"
            f"  Step 5: compare both with the stated genre{genre_hint} and decide Y/N.\n"
            "Then produce EXACTLY the four-line output format below.\n\n"
        )
    if prompt_style == 'linguist_cot':
        return (
            "Adopt the perspective of a CORPUS LINGUIST. For each sentence consider:\n"
            "  - register (formal / informal / conversational / narrative / academic)\n"
            "  - cohesive markers (discourse particles, fillers, conjunctions)\n"
            "  - proper nouns and named entities (agencies, people, places, brands)\n"
            "  - pronoun usage (1st/2nd/3rd person) and tense\n"
            "Based on those linguistic cues, pick each sentence's BEST-MATCHING genre from "
            f"the 10-genre whitelist, then compare with the stated genre{genre_hint} and decide Y/N.\n\n"
        )
    return (
        "For each sentence, identify 1-2 KEY linguistic/stylistic markers.\n"
        f"Determine if each sentence matches the stated genre{genre_hint}.\n\n"
    )


def build_task6_prompt(task_description: str, text2annotate: str,
                       examples_str: str,
                       prompt_style: str = 'neutral') -> str:
    """Task 6 ICL prompt 构建（5 种 prompt_style）。"""
    stated_genre = Task6ExampleSelector._extract_genre(text2annotate)
    header = "You are analyzing whether sentences belong to a stated genre.\n\n"
    if prompt_style == 'linguist_cot':
        header = ("You are a corpus linguist analyzing whether sentences "
                  "belong to a stated genre.\n\n")

    return (
        header
        + "### Task\n"
        + f"{task_description}\n\n"
        + _GENRE_REF_BLOCK
        + "### Examples\n"
        + f"{examples_str}\n\n"
        + "### Text to Classify\n"
        + f"{text2annotate}\n\n"
        + _build_instruction(prompt_style, stated_genre)
        + _build_rules(prompt_style, stated_genre)
        + _WHITELIST_BLOCK
        + _OUTPUT_BLOCK
    )


__all__ = [
    'TASK6_LABEL_SET', 'TASK6_GENRE_SET', 'TASK6_GENRE_WHITELIST_ORDER',
    'TASK6_MAX_TOKENS', 'TASK6_TIMEOUT', 'TASK6_TEMPERATURE',
    'TASK6_REPETITION_PENALTY', 'TASK6_MAX_INPUT_LENGTH', 'TASK6_MIN_INPUT_LENGTH',
    'TASK6_N_ROUNDS', 'TASK6_MULTI_VIEW_CONFIGS',
    'Task6ExampleSelector', 'build_task6_prompt',
    'validate_task6_prediction', 'majority_vote',
]
