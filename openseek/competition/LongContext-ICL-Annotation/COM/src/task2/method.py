"""
method.py — Task 2 (count_nouns_verbs) 方法模块

包含 Task 2 全部专用逻辑：
- ``AUX_VERBS``               : 助动词/情态动词黑名单（不计入动词数）
- ``_BM25``                   : 纯句子相似度检索（任务专用排序信号）
- ``Task2ExampleSelector``    : 类型分桶 (noun/verb) → BM25 排序 → 含 think 的示例选择器
- ``build_task2_prompt``      : Task 2 专用 Prompt（与 CoT 生成侧规则完全一致）

通用 LLM 调用、答案抽取等能力位于 ``COM/src/common/llm_client.py``。
"""

import math
import logging

import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================
# 1. 助动词/情态动词黑名单
# ============================================================

AUX_VERBS = {
    'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
    'do', 'does', 'did',
    'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must',
}
# 注：保留 has/have/had（可作实义动词"持有"，靠 prompt 词性规则区分）


# ============================================================
# 2. BM25 检索模型（用于纯句子相似度）
# ============================================================

class _BM25:
    """BM25 检索模型，用于 Task 2 纯句子相似度计算。"""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
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
# 3. Task 2 示例选择器（含 think 字段）
# ============================================================

class Task2ExampleSelector:
    """
    Task 2 专用示例选择器。

    流程：
    1. 按目标 (noun/verb) 预过滤示例池（一次性缓存 + 构建 BM25）
    2. 默认过滤无 cot 的降级样本（降级样本作为 shot 有隐性负信号）；
       过滤后池容量不足 ``min_pool_size`` 时自动回落不过滤
    3. 用 BM25 对纯句子做相似度排序，相似度低→高，最相似排在末尾（recency bias）
    4. token 截断采用 ``reverse_select=True``：从末尾开始累加，
       优先保留高相关示例

    示例格式（有 think+cot）：
        # {input}
        {think}
        {cot}
        <label>{output}</label>

    示例格式（仅 cot）：
        # {input}
        {cot}
        <label>{output}</label>

    示例格式（无 cot 兜底）：
        # {input} <label>{output}</label>
    """

    def __init__(self, tokenizer: AutoTokenizer, max_context_tokens: int = 31000,
                 min_context_tokens: int = 30000, use_think: bool = True,
                 filter_uncot: bool = True, min_pool_size: int = 100):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.min_context_tokens = min_context_tokens
        self.use_think = use_think
        self.filter_uncot = filter_uncot
        self.min_pool_size = min_pool_size

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _extract_sentence(input_text: str) -> str:
        """用 split 代替 regex，避免 girl's 等撇号导致提前截断。"""
        for prefix, close in [("Sentence: '", "'. Count"), ("sentence: '", "'. Count"),
                              ('Sentence: "', '". Count'), ('sentence: "', '". Count')]:
            if prefix in input_text:
                return input_text.split(prefix, 1)[1].split(close, 1)[0]
        return input_text

    def _init_task2_pools(self, all_examples: list[dict]):
        """预缓存 noun/verb 示例池 + BM25（只初始化一次）"""
        if hasattr(self, '_task2_pools'):
            return
        self._task2_pools = {}
        for target in ('noun', 'verb'):
            pool = [ex for ex in all_examples if target in ex['input'].lower()]
            raw_size = len(pool)
            logger.info(f"Task 2 ({target}): 类型过滤 {len(all_examples)}→{raw_size}")
            if self.filter_uncot:
                filtered = [ex for ex in pool if ex.get('cot', '').strip()]
                if len(filtered) >= self.min_pool_size:
                    pool = filtered
                    logger.info(f"Task 2 ({target}): 过滤无cot降级样本 {raw_size}→{len(pool)} "
                                f"(移除 {raw_size - len(pool)} 条降级)")
                else:
                    logger.warning(f"Task 2 ({target}): 过滤后仅 {len(filtered)} 条 < "
                                   f"阈值 {self.min_pool_size}，回落不过滤")
            if pool:
                has_think = sum(1 for e in pool if e.get('think'))
                has_cot = sum(1 for e in pool if e.get('cot'))
                logger.info(f"Task 2 ({target}): think={has_think}/{len(pool)}, "
                            f"cot={has_cot}/{len(pool)}")
                corpus = [self._extract_sentence(ex['input']) for ex in pool]
                bm25 = _BM25(corpus)
                logger.info(f"Task 2 ({target}) BM25 构建完成: {len(corpus)} 条, "
                            f"avgdl={bm25.avgdl:.1f}, vocab={len(bm25.df)}")
                self._task2_pools[target] = (pool, bm25)
            else:
                self._task2_pools[target] = ([], None)

    def _truncate_by_tokens(self, example_lines: list[str],
                            reverse_select: bool = False) -> tuple[str, int]:
        """从末尾（最相似）开始选，优先保留高分示例。"""
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
        """Task 2: 类型预过滤 → BM25 评分排序 → 含 think 的示例字符串"""
        self._init_task2_pools(all_examples)

        text_lower = test_input.lower()
        target = 'noun' if 'noun' in text_lower else 'verb'
        pool, bm25 = self._task2_pools.get(target, ([], None))
        if not pool or bm25 is None:
            return ""

        # BM25 评分排序（低→高，最相似放最后 recency bias）
        test_sentence = self._extract_sentence(test_input)
        scores = bm25.scores(test_sentence)
        ranked = sorted(range(len(pool)), key=lambda i: scores[i])

        example_lines = []
        for idx in ranked:
            ex = pool[idx]
            output = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
            think = ex.get('think', '').strip() if self.use_think else ''
            cot = ex.get('cot', '').strip()
            if think and cot:
                line = f"# {ex['input']}\n{think}\n{cot}\n<label>{output}</label>\n"
            elif cot:
                line = f"# {ex['input']}\n{cot}\n<label>{output}</label>\n"
            else:
                line = f"# {ex['input']} <label>{output}</label>\n"
            example_lines.append(line)

        return self._truncate_by_tokens(example_lines, reverse_select=True)[0]


# ============================================================
# 4. Task 2 Prompt 构建（与 generate_cot.py 规则完全一致）
# ============================================================

_VERB_RULES = (
    "### Rules for counting verbs\n"
    "Count ONLY action/lexical verbs (e.g. run, eat, play, walk, catch).\n"
    "EXCLUDE these auxiliary/modal verbs:\n"
    "  is, are, was, were, am, be, been, being,\n"
    "  do, does, did,\n"
    "  will, would, shall, should,\n"
    "  can, could, may, might, must.\n"
    "\n"
    "Special case for has / have / had (decide by what follows them):\n"
    "  - COUNT as a VERB when followed by a NOUN or noun phrase\n"
    "    (main verb, meaning \"to possess/own\"):\n"
    "      \"I have a cat\"         -> have = verb (counted)\n"
    "      \"She has blue eyes\"    -> has  = verb (counted)\n"
    "      \"They had three dogs\"  -> had  = verb (counted)\n"
    "  - EXCLUDE when followed by a PAST PARTICIPLE\n"
    "    (auxiliary verb forming perfect tenses):\n"
    "      \"has eaten\"             -> has  = auxiliary (excluded)\n"
    "      \"have been working\"     -> have = auxiliary (excluded)\n"
    "      \"had finished the task\" -> had  = auxiliary (excluded)\n"
    "\n"
    "Additional inclusion rules (IMPORTANT, do NOT miss these):\n"
    "  - -ing forms (present participles / gerunds) ARE counted as verbs,\n"
    "    EVEN WHEN they modify a noun (adjectival use) or act as a noun:\n"
    "      \"boiling water\"             -> boiling  = verb (counted)\n"
    "      \"tasty looking food\"        -> looking  = verb (counted)\n"
    "      \"for loading passengers\"    -> loading  = verb (counted)\n"
    "      \"I enjoy swimming\"          -> swimming = verb (counted)\n"
    "      \"a running man\"             -> running  = verb (counted)\n"
    "  - Past participles that act as the MAIN verb in passive voice or\n"
    "    perfect tense ARE counted (the auxiliary is/are/was/were/being/been\n"
    "    is still EXCLUDED as listed above, but the participle itself IS):\n"
    "      \"is lowered\"        -> lowered = verb (counted), is    = excluded\n"
    "      \"are being canned\"  -> canned  = verb (counted), are/being = excluded\n"
    "      \"has been opened\"   -> opened  = verb (counted), has/been  = excluded\n"
    "      \"was broken\"        -> broken  = verb (counted), was   = excluded\n"
)

_NOUN_RULES = (
    "### Rules for counting nouns\n"
    "Count ALL nouns: people, places, things, animals, abstract concepts.\n"
    "Both singular and plural forms count (e.g. \"cat\" and \"cats\" each count as 1).\n"
    "Proper nouns count too (e.g. \"John\", \"Paris\").\n"
    "Gerunds used as nouns are still counted as verbs (see verb rules),\n"
    "NOT as nouns.\n"
)


def build_task2_prompt(task_description: str, text2annotate: str,
                       examples_str: str) -> str:
    """Task 2 (count_nouns_verbs) Prompt — 与 CoT 生成侧规则完全一致"""
    return (
        "### Task\n"
        f"{task_description}\n\n"

        f"{_VERB_RULES}\n"
        f"{_NOUN_RULES}\n"

        "### Examples\n"
        f"{examples_str}\n\n"

        "### Input to Annotate\n"
        f"{text2annotate}\n\n"

        "Count carefully following the rules above.\n"
        "First, list the found words briefly in one line, e.g. \"Nouns: cat, dog, house\" or \"Verbs: run, eat, play\".\n"
        "If NO matching words are found, write \"Verbs: none\" or \"Nouns: none\".\n"
        "Then output the count in <label> tags.\n"
        "Example formats:\n"
        "Nouns: cat, dog, house\n"
        "<label>3</label>\n\n"
        "Verbs: run, eat [skip: is, can]\n"
        "<label>2</label>\n\n"
        "Verbs: none [skip: is, are]\n"
        "<label>0</label>"
    )
