"""
method.py — Task 5 (tweet_sadness_detection) 方法模块

包含 Task 5 全部专用逻辑：
- ``_BM25``                   : 纯 Python BM25 检索（剥离 @mentions 噪声）
- ``Task5ExampleSelector``    : 池构成混合 + 标签比例 + 排序 三轴可正交配置的示例选择器
- ``build_task5_prompt``      : 5 套 prompt 模板（neutral / not_sad_default / sad_default
                                 / cot_dual / psychologist_cot）
- ``extract_label_set``       : 从示例池提取标签集合
- ``validate_prediction``     : 约束输出到合法标签集（精确 / 大小写 / 子串）
- ``majority_vote``           : 5 轮平权硬投票
- ``TASK5_MULTI_VIEW_CONFIGS``: 5 轮正交多视角配置 (A/B/C/D 四轴)

通用 LLM 调用、答案抽取等能力位于 ``COM/src/common/llm_client.py``。
"""

import math
import re
import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Task 5 常量配置
# ============================================================

TASK5_MAX_TOKENS = 512              # R0/R1/R2 用
TASK5_MAX_TOKENS_COT = 1024         # R3/R4 In-Prompt CoT 用
TASK5_TIMEOUT = 600                 # 单次请求超时
TASK5_TEMPERATURE = 0.0             # 确定性输出
TASK5_REPETITION_PENALTY = 1.0      # 不干预
TASK5_ENABLE_THINKING = False       # 关闭 thinking，避免不可复现随机

TASK5_MAX_INPUT_LENGTH = 31000      # 单条 ICL prompt 的 token 上限（赛题硬约束 ≥ 30K，留 1K 余量）
TASK5_MIN_INPUT_LENGTH = 30000      # 单条 ICL prompt 的 token 下限（赛题硬约束 ≥ 30K，必须达标）

# 5 轮正交多视角配置：
#   pool_composition: full_top | top_mid_mix | top_bottom_mix | mid_bottom_mix
#   label_balance:    natural | forced_50 | reverse_40_60
#   order:            recency_up | recency_down | interleaved
#   prompt_style:     neutral | not_sad_default | sad_default | cot_dual | psychologist_cot
TASK5_N_ROUNDS = 5
TASK5_MULTI_VIEW_CONFIGS = [
    # R0: 稳定锚点 = 纯高相关性 + 中性 prompt
    {
        'pool_composition': 'full_top',
        'label_balance':    'natural',
        'order':            'recency_up',
        'prompt_style':     'neutral',
        'max_tokens':       TASK5_MAX_TOKENS,
    },
    # R1: top + mid 50:50 混合，Not-sad default 压制 Sad 过度预测
    {
        'pool_composition': 'top_mid_mix',
        'label_balance':    'forced_50',
        'order':            'interleaved',
        'prompt_style':     'not_sad_default',
        'max_tokens':       TASK5_MAX_TOKENS,
    },
    # R2: top + bottom 50:50 混合（相似度极端对冲），Sad default 对抗抑制过度
    {
        'pool_composition': 'top_bottom_mix',
        'label_balance':    'reverse_40_60',
        'order':            'recency_up',
        'prompt_style':     'sad_default',
        'max_tokens':       TASK5_MAX_TOKENS,
    },
    # R3: full_top + In-Prompt CoT 双面分析（greedy 确定性）
    {
        'pool_composition': 'full_top',
        'label_balance':    'forced_50',
        'order':            'recency_down',
        'prompt_style':     'cot_dual',
        'max_tokens':       TASK5_MAX_TOKENS_COT,
    },
    # R4: mid + bottom 混合（远相关性扰动），心理学家 CoT
    {
        'pool_composition': 'mid_bottom_mix',
        'label_balance':    'natural',
        'order':            'interleaved',
        'prompt_style':     'psychologist_cot',
        'max_tokens':       TASK5_MAX_TOKENS_COT,
    },
]


# ============================================================
# 1. BM25 检索（剥离 @mentions 噪声）
# ============================================================

_MENTION_RE = re.compile(r'@\S+')


def _strip_mentions(text: str) -> str:
    if not text:
        return text
    return _MENTION_RE.sub('', text)


class _BM25:
    """pure-Python BM25, k1=1.5, b=0.75."""

    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(doc.split()) for doc in corpus) / self.N if self.N > 0 else 0
        self.df = {}
        self.tf = []
        self.doc_len = []
        for doc in corpus:
            words = doc.lower().split()
            self.doc_len.append(len(words))
            tf_dict = {}
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
# 2. Task 5 示例选择器（A/B/C 三轴可正交配置）
# ============================================================

class Task5ExampleSelector:
    """Task 5 示例选择器：A/B/C 三轴可正交配置。

    A 池构成 (pool_composition):
        full_top       : top_1/3 全量 + mid_1/3 前 _MIX_HALF (候选~1083)
        top_mid_mix    : top_1/3 前 _MIX_HALF + mid_1/3 前 _MIX_HALF (1000)
        top_bottom_mix : top_1/3 前 _MIX_HALF + bottom_1/3 前 _MIX_HALF (1000)
        mid_bottom_mix : mid_1/3 前 _MIX_HALF + bottom_1/3 前 _MIX_HALF (1000)

    B 标签比例 (label_balance):
        natural        : 保留池内原生分布
        forced_50      : 严格 50:50 轮询
        reverse_40_60  : 偏向 Not sad (40% Sad, 60% Not sad)

    C 示例排序 (order):
        recency_up     : BM25 分升序（最相似放末尾，recency bias）
        recency_down   : BM25 分降序（最相似放开头）
        interleaved    : 按 BM25 分排序后交替打散，高/低相似度穿插

    每个候选池经 B/C 轴重排后，靠 token 上限截断稳定填充到 30K+ tokens
    （赛题硬约束 ≥ 30K）。
    """

    # 每段混合时的取样条数；mid/bottom 段推文平均更短，多备素材保证 token 填满 30K
    _MIX_HALF = 500

    def __init__(self, tokenizer,
                 max_context_tokens: int = TASK5_MAX_INPUT_LENGTH,
                 min_context_tokens: int = TASK5_MIN_INPUT_LENGTH):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.min_context_tokens = min_context_tokens
        self._bm25 = None
        self._examples_cache = None

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _log_token_budget(self, n_examples: int, total_tokens: int,
                          round_idx: int, mode: str, label_info: str) -> None:
        """统一日志：输出 token 数 + 标签分布；低于下限时升级为 WARNING。"""
        compliance_tag = "✅" if total_tokens >= self.min_context_tokens else "⚠️<30K"
        msg = (f"[Task5] R{round_idx} 截断后: {n_examples} 条 ({total_tokens} tokens "
               f"{compliance_tag}, 区间 [{self.min_context_tokens}, "
               f"{self.max_context_tokens}], {mode}) 标签分布: {label_info}")
        if total_tokens < self.min_context_tokens:
            logger.warning(f"[ICL 长度不足] {msg}")
        else:
            logger.info(msg)

    def _init_bm25(self, examples: list) -> None:
        if self._examples_cache is examples and self._bm25 is not None:
            return
        corpus = [_strip_mentions(ex['input']) for ex in examples]
        logger.info(f"[Task5] 构建 BM25 索引（已剥离@mentions），共 {len(corpus)} 个示例...")
        self._bm25 = _BM25(corpus)
        self._examples_cache = examples
        label_dist = Counter(
            ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            for ex in examples
        )
        logger.info(f"[Task5] BM25 构建完成: N={self._bm25.N}, "
                    f"avgdl={self._bm25.avgdl:.1f}, vocab={len(self._bm25.df)}, "
                    f"标签分布: {dict(label_dist)}")

    def _compute_similarity(self, test_input: str) -> np.ndarray:
        return self._bm25.scores(_strip_mentions(test_input))

    # ---- A 轴: 池构成混合（按 BM25 排名分段后混合取样）----
    def _pool_composition(self, ranked: list, composition: str) -> list:
        pool_size = len(ranked)
        third = pool_size // 3
        top_seg = ranked[:third]
        mid_seg = ranked[third:2 * third]
        bot_seg = ranked[2 * third:]
        half = self._MIX_HALF

        if composition == 'full_top':
            # 纯相关性优先: top 全量 + mid 前 _MIX_HALF；后续 token 截断会优先保留最相似
            return top_seg + mid_seg[:half]
        if composition == 'top_mid_mix':
            return top_seg[:half] + mid_seg[:half]
        if composition == 'top_bottom_mix':
            return top_seg[:half] + bot_seg[:half]
        if composition == 'mid_bottom_mix':
            return mid_seg[:half] + bot_seg[:half]
        # 兜底：等同 full_top
        return top_seg + mid_seg[:half]

    # ---- B 轴: 标签比例采样 ----
    def _apply_label_balance(self, pool_indices: list, all_examples: list,
                             sims: np.ndarray, label_balance: str) -> list:
        """返回 [(idx, sim, ex), ...] 列表，按 label_balance 策略采样。"""
        groups = defaultdict(list)
        for idx in pool_indices:
            ex = all_examples[idx]
            output = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            groups[output].append((idx, float(sims[idx]), ex))

        if label_balance == 'natural':
            merged = []
            for items in groups.values():
                merged.extend(items)
            return merged

        if label_balance == 'forced_50':
            # 严格 50:50 轮询
            labels = sorted(groups.keys())
            iters = {l: iter(groups[l]) for l in labels}
            out = []
            exhausted = set()
            while len(exhausted) < len(labels):
                for l in labels:
                    if l in exhausted:
                        continue
                    try:
                        out.append(next(iters[l]))
                    except StopIteration:
                        exhausted.add(l)
            return out

        if label_balance == 'reverse_40_60':
            # 40% Sad, 60% Not sad；用 4:6 的比例轮询
            sad_items = groups.get('Sad', [])
            not_items = groups.get('Not sad', [])
            out = []
            si, ni = 0, 0
            while si < len(sad_items) or ni < len(not_items):
                for _ in range(4):
                    if si < len(sad_items):
                        out.append(sad_items[si])
                        si += 1
                for _ in range(6):
                    if ni < len(not_items):
                        out.append(not_items[ni])
                        ni += 1
                if (si >= len(sad_items)) and (ni >= len(not_items)):
                    break
            return out

        # 兜底
        merged = []
        for items in groups.values():
            merged.extend(items)
        return merged

    # ---- C 轴: 示例排序 ----
    def _apply_order(self, balanced: list, order: str) -> list:
        if order == 'recency_up':
            return sorted(balanced, key=lambda x: x[1])
        if order == 'recency_down':
            return sorted(balanced, key=lambda x: x[1], reverse=True)
        if order == 'interleaved':
            ranked_desc = sorted(balanced, key=lambda x: x[1], reverse=True)
            n = len(ranked_desc)
            half = (n + 1) // 2
            out = []
            for i in range(half):
                out.append(ranked_desc[i])
                if i + half < n:
                    out.append(ranked_desc[i + half])
            return out
        return balanced

    # ---- Token 截断 ----
    def _truncate_by_tokens(self, example_lines: list,
                            reverse_select: bool = True) -> tuple:
        if reverse_select:
            selected, total = [], 0
            for line in reversed(example_lines):
                n = self._count_tokens(line)
                if total + n > self.max_context_tokens:
                    break
                selected.append(line)
                total += n
            selected.reverse()
            return "".join(selected), total
        selected, total = [], 0
        for line in example_lines:
            n = self._count_tokens(line)
            if total + n > self.max_context_tokens:
                break
            selected.append(line)
            total += n
        return "".join(selected), total

    # ---- 主入口 ----
    def select(self, all_examples: list, test_input: str,
               round_idx: int = 0,
               pool_composition: str = 'full_top',
               label_balance: str = 'natural',
               order: str = 'recency_up') -> str:
        """三轴正交示例选择主入口。"""
        self._init_bm25(all_examples)
        sims = self._compute_similarity(test_input)

        # BM25 分降序全池排名
        ranked = sorted(range(len(all_examples)), key=lambda i: sims[i], reverse=True)

        # A 轴：池构成混合
        pool_indices = self._pool_composition(ranked, pool_composition)

        # B 轴：标签比例
        balanced = self._apply_label_balance(pool_indices, all_examples, sims, label_balance)

        # C 轴：排序策略
        ordered = self._apply_order(balanced, order)

        logger.info(f"[Task5] R{round_idx}: pool_composition={pool_composition}|{len(pool_indices)} "
                    f"label_balance={label_balance}|{len(balanced)} order={order}")

        # 格式化行
        lines = []
        for _, _, ex in ordered:
            output_text = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            lines.append(f"# {ex['input']} <label>{output_text}</label>\n")

        # Token 截断（order=recency_up 时优先保留末尾，其它优先保留开头）
        reverse_select = (order == 'recency_up')
        truncated_str, total_tokens = self._truncate_by_tokens(
            lines, reverse_select=reverse_select)

        # 统计截断后实际进入 prompt 的标签分布 + ICL token 占用（合规观测）
        trunc_labels = defaultdict(int)
        for line in truncated_str.split('\n'):
            m = re.search(r'<label>(\w[\w\s]*)</label>', line)
            if m:
                trunc_labels[m.group(1)] += 1
        label_info = ', '.join(f'{k}:{v}' for k, v in sorted(trunc_labels.items()))
        total_trunc = sum(trunc_labels.values())

        mode = 'reverse_select' if reverse_select else 'forward_select'
        self._log_token_budget(total_trunc, total_tokens, round_idx, mode, label_info)

        return truncated_str


# ============================================================
# 3. 5 套 Prompt 模板（D 轴）
# ============================================================

def _prompt_neutral(task_description, text2annotate, examples_str, labels_display) -> str:
    return (
        "### Role\n"
        "You are a professional text classification expert.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Valid Labels\n"
        f"You MUST choose exactly one from: {labels_display}\n\n"
        "### Rules\n"
        "1. Study the examples carefully to understand the classification criteria.\n"
        "2. Classify the test input into exactly one of the valid labels above.\n"
        "3. Output ONLY the label wrapped in <label> tags. No explanation, no extra text.\n\n"
        "### Examples\n"
        f"{examples_str}\n\n"
        "### Text to Classify\n"
        f"{text2annotate}\n\n"
        "Output ONLY the label wrapped in <label> tags."
    )


def _prompt_not_sad_default(task_description, text2annotate, examples_str, labels_display) -> str:
    return (
        "### Role\n"
        "You are a cautious classification expert who applies a strict threshold for \"Sad\".\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Valid Labels\n"
        f"You MUST choose exactly one from: {labels_display}\n\n"
        "### Rules\n"
        "1. Output \"Sad\" ONLY when the tweet CLEARLY expresses sadness, grief, depression, heartbreak, or despair.\n"
        "2. Tweets with neutral, informational, sarcastic, humorous, frustrated, or angry tones should be \"Not sad\".\n"
        "3. Mere complaints, annoyance, tiredness, or negativity without genuine sadness are \"Not sad\".\n"
        "4. When in doubt, default to \"Not sad\".\n"
        "5. Output ONLY the label wrapped in <label> tags. No explanation, no extra text.\n\n"
        "### Examples\n"
        f"{examples_str}\n\n"
        "### Text to Classify\n"
        f"{text2annotate}\n\n"
        "Output ONLY the label wrapped in <label> tags."
    )


def _prompt_sad_default(task_description, text2annotate, examples_str, labels_display) -> str:
    return (
        "### Role\n"
        "You are a sensitive emotional-analysis expert who catches subtle emotional cues.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Valid Labels\n"
        f"You MUST choose exactly one from: {labels_display}\n\n"
        "### Rules\n"
        "1. Output \"Sad\" if the tweet shows ANY tone of sadness, disappointment, hurt, loneliness, "
        "grief, downheartedness, or emotional pain.\n"
        "2. Subtle cues count: certain emojis (😢💔😭), hashtags (#sad #lonely), word choices "
        "(miss, alone, empty, broken), or context that implies emotional distress.\n"
        "3. Output \"Not sad\" ONLY when the tweet is clearly neutral, joyful, promotional, or purely informational.\n"
        "4. Output ONLY the label wrapped in <label> tags. No explanation, no extra text.\n\n"
        "### Examples\n"
        f"{examples_str}\n\n"
        "### Text to Classify\n"
        f"{text2annotate}\n\n"
        "Output ONLY the label wrapped in <label> tags."
    )


def _prompt_cot_dual(task_description, text2annotate, examples_str, labels_display) -> str:
    return (
        "### Role\n"
        "You are a rigorous classification expert who always weighs both sides before deciding.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Valid Labels\n"
        f"You MUST choose exactly one from: {labels_display}\n\n"
        "### Rules\n"
        "1. First, list evidence in the tweet that supports \"Sad\" (if any).\n"
        "2. Second, list evidence in the tweet that supports \"Not sad\" (if any).\n"
        "3. Weigh both sides and give a brief verdict.\n"
        "4. Finally, output the label wrapped in <label> tags.\n\n"
        "### Output Format (you MUST follow this structure)\n"
        "Reasoning:\n"
        "- Sad evidence: <bullet list, or 'none'>\n"
        "- Not sad evidence: <bullet list, or 'none'>\n"
        "- Verdict: <one short sentence explaining which side wins>\n"
        "<label>Sad</label>   or   <label>Not sad</label>\n\n"
        "### Examples\n"
        f"{examples_str}\n\n"
        "### Text to Classify\n"
        f"{text2annotate}\n\n"
        "Follow the Output Format. End with <label>...</label>."
    )


def _prompt_psychologist_cot(task_description, text2annotate, examples_str, labels_display) -> str:
    return (
        "### Role\n"
        "You are a clinical psychologist specializing in sentiment analysis of social media text. "
        "Apply clinical rigor: only label \"Sad\" when the emotion belongs to the sadness family.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        "### Valid Labels\n"
        f"You MUST choose exactly one from: {labels_display}\n\n"
        "### Three-Step Judgement (MUST follow)\n"
        "Step 1: Identify the author's primary emotion in 1-3 words.\n"
        "Step 2: Decide whether that emotion belongs to the sadness family "
        "(sorrow, grief, depression, loneliness, hurt, disappointment, heartbreak). "
        "Answer yes or no.\n"
        "Step 3: Output the final label.\n\n"
        "### Output Format\n"
        "Emotion: <1-3 words>\n"
        "In-sadness-family: <yes|no>\n"
        "<label>Sad</label>   or   <label>Not sad</label>\n\n"
        "### Examples\n"
        f"{examples_str}\n\n"
        "### Text to Classify\n"
        f"{text2annotate}\n\n"
        "Follow the Output Format. End with <label>...</label>."
    )


_PROMPT_BUILDERS = {
    'neutral':           _prompt_neutral,
    'not_sad_default':   _prompt_not_sad_default,
    'sad_default':       _prompt_sad_default,
    'cot_dual':          _prompt_cot_dual,
    'psychologist_cot':  _prompt_psychologist_cot,
}


def build_task5_prompt(task_description: str, text2annotate: str,
                       examples_str: str, label_set: list,
                       prompt_style: str = 'neutral') -> str:
    """Task 5 分类 Prompt 构造。按 prompt_style 分发到具体模板。"""
    labels_display = ", ".join(f'"{l}"' for l in label_set)
    builder = _PROMPT_BUILDERS.get(prompt_style, _prompt_neutral)
    return builder(task_description, text2annotate, examples_str, labels_display)


# ============================================================
# 4. 标签集与预测合法性约束
# ============================================================

def extract_label_set(examples: list) -> list:
    labels = set()
    for ex in examples:
        output = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
        labels.add(output)
    return sorted(labels)


def validate_prediction(prediction: Optional[str],
                        label_set: Optional[list] = None) -> Optional[str]:
    """约束输出到合法标签集；依次尝试精确 / 大小写不敏感 / 子串包含。"""
    if prediction is None:
        return None
    prediction = prediction.strip()
    if not prediction:
        return None
    if prediction in ('...', '…', '..', 'N/A', 'n/a', 'None', 'none', 'null'):
        return None
    if label_set:
        if prediction in label_set:
            return prediction
        for label in label_set:
            if label.lower() == prediction.lower():
                return label
        for label in label_set:
            if label in prediction:
                return label
        logger.warning(f"[Task5] 答案不在标签集: '{prediction[:60]}'")
    return prediction


# ============================================================
# 5. 多数票投票（5 轮平权硬投票）
# ============================================================

def majority_vote(predictions: list) -> Optional[str]:
    if not predictions:
        return None
    counter = Counter(predictions)
    winner, count = counter.most_common(1)[0]
    logger.info(f"[Task5] 投票: {len(predictions)} 候选, 分布={dict(counter)}, "
                f"胜出='{winner}' ({count}票)")
    return winner
