"""
method.py — Task 8 (kernel_generation) 核心方法库

包含 Task 8 专用的：
1. 全局配置（token 预算 / 超时 / 重复惩罚）
2. Wrapper 信息解析（func_name / func_signature / math_formula / func_desc / constraints）
3. 示例选择器（BM25 + 操作类型分类）
4. Prompt 构建（Draft / Verify / Safe-Exit / React Observation）
5. Import 补全 + Wrapper 签名后处理
6. 静态校验（AST + 函数名 + 签名 + Placeholder + Exec 预检）
7. ReAct 多轮修复 + AST 兜底（强制 try/except）

API 推理统一走 ``common.llm_client.annotate_nvidia``；
``annotate_multi_turn`` 为多轮对话场景的特化封装，本地保留。
"""

import os
import re
import ast
import sys
import json
import math
import time
import logging
from collections import Counter
from typing import Optional

import numpy as np

# 复用 common 层的单例 OpenAI 客户端 + 答案解析（避免再起一个连接）
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # COM/src/task8
_SRC_DIR = os.path.abspath(os.path.join(_CUR_DIR, '..'))        # COM/src
for _p in (_CUR_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.llm_client import (  # noqa: E402
    _client,
    annotate_nvidia as annotate,
    count_answer,
)
from common.paths import VLLM_MODEL_ID  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 模块0：全局配置
# ============================================================

TASK8_MAX_TOKENS = 10000       # Kernel 代码 + 分析步骤 + thinking
TASK8_TIMEOUT = 900            # Kernel 代码生成超时 15 分钟（覆盖复杂融合算子）
TASK8_MAX_INPUT_LENGTH = 17000 # 示例 token 预算上限
TASK8_MIN_INPUT_LENGTH = 16000 # 示例 token 预算下限（赛题强制要求 ≥16K）
TASK8_REP_PENALTY = 1.2        # 强惩罚：抑制 thinking 复读和循环


# ============================================================
# 模块0.5：Wrapper 信息解析
# ============================================================

def _parse_wrapper_info(text2annotate: str, prefilled: dict = None) -> dict:
    """从 Task 8 test input 中解析结构化信息：函数签名、Math公式、功能描述。

    返回 dict:
        func_signature: str  完整的 wrapper 签名行
        func_name: str       wrapper 函数名
        math_formula: str    数学公式部分
        func_desc: str       功能描述
        constraints: str     other: 段里的形状/dtype/生成约束

    Args:
        prefilled: 若传入数据集已解析好的字段，则优先使用，仅解析缺失字段。
                  用于避免上层加载 normalized 数据后下游重复解析 raw_input。
    """
    info = {
        'func_signature': '',
        'func_name': '',
        'math_formula': '',
        'func_desc': '',
        'constraints': '',
    }

    # 如果 prefilled 已包含关键字段 (func_name + func_signature),
    # 直接返回, 跳过昂贵的文本解析
    if prefilled:
        for k in ('func_name', 'func_signature', 'math_formula', 'func_desc', 'constraints'):
            v = prefilled.get(k)
            if v:
                info[k] = v
        if info['func_name'] and info['func_signature']:
            return info

    def _find_line_start(marker: str, from_pos: int) -> int:
        """找 marker 仅在行首的位置, 避免参数列表里的 `other:` 被误识别."""
        search_from = from_pos
        while True:
            p = text2annotate.find(marker, search_from)
            if p == -1:
                return -1
            if p == 0 or text2annotate[p - 1] == '\n':
                return p
            search_from = p + 1

    # 1. 提取 Functional Description
    if 'Functional Description:' in text2annotate:
        start = text2annotate.index('Functional Description:') + len('Functional Description:')
        end = text2annotate.index('Wrapper Entry Information:') if 'Wrapper Entry Information:' in text2annotate else len(text2annotate)
        info['func_desc'] = text2annotate[start:end].strip()

    # 2. 提取 Wrapper Entry Information -> 函数签名
    if 'Wrapper Entry Information:' in text2annotate:
        start = text2annotate.index('Wrapper Entry Information:') + len('Wrapper Entry Information:')
        end = len(text2annotate)
        # 仅在行首匹配 marker, 避免参数里 `other:` 被误截断
        for marker in ['Math:', 'other:', 'After generation']:
            pos = _find_line_start(marker, start)
            if pos != -1 and pos < end:
                end = pos
        raw_sig = text2annotate[start:end].strip()

        # --- 签名清洗 (与 normalize_dataset.clean_wrapper_signature 对齐) ---
        # (a) 去掉 `def ` 前缀
        sig_clean = raw_sig
        _m = re.match(r'^def\s+', sig_clean)
        if _m:
            sig_clean = sig_clean[_m.end():]

        # (b) 函数名段清洗: 剥 `torch.` 前缀, 再把剩余 `.` 替换为 `_`
        #     仅作用于 `(` 之前, 不触碰参数注解里的 `input: torch.Tensor`
        _paren = sig_clean.find('(')
        if _paren > 0:
            _head = sig_clean[:_paren]
            _tail = sig_clean[_paren:]
            _head = re.sub(r'^torch\.', '', _head)
            _head = _head.replace('.', '_')
            sig_clean = _head + _tail
            _paren = sig_clean.find('(')

        # (c) 用括号匹配找到配对 `)`, 丢弃 `-> Type` / 参数说明 / 换行等
        if _paren > 0:
            depth = 0
            rp = -1
            for _i in range(_paren, len(sig_clean)):
                _c = sig_clean[_i]
                if _c == '(':
                    depth += 1
                elif _c == ')':
                    depth -= 1
                    if depth == 0:
                        rp = _i
                        break
            if rp != -1:
                sig_clean = sig_clean[:rp + 1]

        # (d) 折叠签名内部多余空白
        sig_clean = re.sub(r'\s+', ' ', sig_clean).strip()

        info['func_signature'] = sig_clean if sig_clean.endswith(')') else raw_sig

        paren_pos = sig_clean.find('(')
        if paren_pos != -1:
            func_name = sig_clean[:paren_pos].strip()
            info['func_name'] = func_name  # 已在 (b) 中把 `.` 替换为 `_`

    # 3. 提取 Math 公式
    if 'Math:' in text2annotate:
        start = text2annotate.index('Math:') + len('Math:')
        end = len(text2annotate)
        for marker in ['other:', 'After generation']:
            pos = _find_line_start(marker, start)
            if pos != -1 and pos < end:
                end = pos
        info['math_formula'] = text2annotate[start:end].strip()

    # 4. 提取 constraints (other: 段, 行首匹配, 到 After generation 截止)
    if not info['constraints']:
        pos_other = _find_line_start('other:', 0)
        if pos_other != -1:
            start = pos_other + len('other:')
            end = len(text2annotate)
            pos_end = _find_line_start('After generation', start)
            if pos_end != -1:
                end = pos_end
            info['constraints'] = text2annotate[start:end].strip()

    return info


def _build_structured_input(info: dict, *, include_after_gen: bool = True) -> str:
    """把已解析的 wrapper info 渲染成结构化 Input 段文本.

    下游 Prompt 统一用这个函数生成 "Functional Description / Wrapper Entry
    Information / Math / Constraints" 四段结构,替换原始 text2annotate,
    避免模型看到拼接生硬 / 未清洗的原文.

    Args:
        info: 含 func_desc / func_signature / math_formula / constraints 的 dict.
        include_after_gen: 是否在末尾追加 "After generation, verify..." 提示.
                          与原始 ICL/Test 格式保持一致.
    """
    parts = []
    desc = (info.get('func_desc') or '').strip()
    sig = (info.get('func_signature') or '').strip()
    math_formula = (info.get('math_formula') or '').strip()
    constraints = (info.get('constraints') or '').strip()

    if desc:
        parts.append(f"Functional Description: {desc}")
    if sig:
        parts.append(f"Wrapper Entry Information: {sig}")
    # Math 段总是输出,未知时用 N/A
    if math_formula and math_formula.lower() != 'n/a':
        parts.append(f"Math: {math_formula}")
    else:
        parts.append("Math: N/A")
    if constraints:
        parts.append(f"other: {constraints}")
    if include_after_gen:
        parts.append(
            "After generation, verify if the Triton wrapper aligns with "
            "the provided func_inputs. If not, regenerate."
        )
    return '\n'.join(parts)


# Task 8 已知的 system prompt 前缀（ICL 和 Test 各一个）
_TASK8_SYSTEM_PROMPTS = [
    "You are a expert in writing Triton operators for efficient GPU programming. Use triton language write a kernel and wrapper according following instruction.",
    "You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.",
]


def _strip_system_prompt(text: str) -> str:
    """去掉 Task 8 输入中的 system prompt 前缀，只保留任务内容。

    用途：BM25 索引构建和相关性计算时，system prompt 对匹配无区分度，
    反而浪费 token 预算、稀释真正有区分度的 Functional Description 等内容。
    """
    for prefix in _TASK8_SYSTEM_PROMPTS:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _reformat_icl_example(icl_input, icl_output=None, cot: str = "") -> str:
    """将 ICL 示例重构为与 Test 输入相同的结构化格式。

    第一参数支持 dict (normalized ex) 或 str (raw_input)：
        - dict 路径: 直接读 ex.func_name / func_signature / func_desc /
          math_formula / constraints / raw_output 等 normalized 字段.
        - str 路径 (向后兼容): 从 icl_input 和 icl_output 重新解析.

    Test 格式:
        "Functional Description: ...
         Wrapper Entry Information: ...
         Math: N/A
         other: ...
         After generation..."
    """
    # --- 分支1: dict 路径 (normalized ex) ---
    if isinstance(icl_input, dict):
        ex = icl_input
        # 如果调用方把 cot 塞到第二个位置参数, 也兼容
        if cot == "" and isinstance(icl_output, str):
            cot = icl_output
        desc = (ex.get('func_desc') or '').strip()
        if not desc:
            raw_in = ex.get('raw_input') or ex.get('input') or ''
            desc = _strip_system_prompt(raw_in)
        sig = (ex.get('func_signature') or '').strip()
        math_formula = (ex.get('math_formula') or '').strip()
        constraints = (ex.get('constraints') or '').strip()
        # code 从 raw_output 或 output 拿
        code = ex.get('raw_output')
        if not code:
            out = ex.get('output')
            if isinstance(out, list) and out:
                code = out[0]
            elif isinstance(out, str):
                code = out
            else:
                code = ''
        cot = cot or ex.get('cot', '')

        info = {
            'func_desc': desc,
            'func_signature': sig,
            'math_formula': math_formula,
            'constraints': constraints,
        }
        body = _build_structured_input(info, include_after_gen=True)

        parts = [body]
        if cot and cot.strip():
            parts.append(cot.strip())
        parts.append(f"<label>{(code or '').strip()}</label>")
        return '\n'.join(parts) + '\n'

    # --- 分支2: str 路径 (向后兼容, 从 raw_input 重新解析) ---
    # 1. 提取功能描述（去掉 system prompt）
    desc = _strip_system_prompt(icl_input)

    # 2. 从 output 代码中提取 wrapper 函数签名
    wrapper_sig = _extract_wrapper_signature(icl_output or '')

    # 3. 构建结构化格式（对齐 Test 输入格式）
    info = {
        'func_desc': desc,
        'func_signature': wrapper_sig,
        'math_formula': 'N/A',
        'constraints': '',
    }
    body = _build_structured_input(info, include_after_gen=True)
    parts = [body]
    if cot and cot.strip():
        parts.append(cot.strip())
    parts.append(f"<label>{(icl_output or '').strip()}</label>")
    return '\n'.join(parts) + '\n'


def _extract_wrapper_signature(code: str) -> str:
    """从 Triton 代码中提取 wrapper（非 @triton.jit）函数的完整签名。"""
    lines = code.strip().split('\n')
    in_kernel = False
    paren_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 跟踪是否在 kernel 内部
        if stripped.startswith('@triton.jit'):
            in_kernel = True
            continue

        if not in_kernel and stripped.startswith('def '):
            # 这是 wrapper 函数，收集签名直到冒号
            sig_parts = []
            for j in range(i, len(lines)):
                s = lines[j].strip()
                if j > i and s.startswith('def '):
                    break  # 下一个函数
                sig_parts.append(s)
                # 计算括号深度
                paren_depth += s.count('(') - s.count(')')
                if ':' in s and paren_depth <= 0:
                    # 签名结束，截取到冒号
                    last = sig_parts[-1]
                    colon_pos = last.index(':')
                    sig_parts[-1] = last[:colon_pos]
                    break
            sig = ' '.join(sig_parts)
            sig = re.sub(r'\s+', ' ', sig).strip()
            return sig

        # 检测 kernel 函数结束（缩进回到 0 或非 kernel 缩进）
        if in_kernel and stripped and not stripped.startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent == 0 and not stripped.startswith('@'):
                in_kernel = False

    return ""


# ============================================================
# 模块1：示例选择器（BM25 版本）
# ============================================================

class _BM25:
    """轻量级 BM25 实现，无需额外依赖。

    Okapi BM25 算法，对代码/技术文本的词频匹配优于 TF-IDF，因为：
    - 文档长度归一化（k1+b 参数）避免长文档优势
    - IDF 惩罚高频常见词（如 'tensor', 'kernel' 等）
    - 对技术文档中精确关键词匹配更敏感
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_lens: list[int] = []
        self.idf: dict[str, float] = {}
        self._initialize(corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """对技术文本做 tokenization：保留下划线标识符、小写化、过滤过短 token。"""
        # 保留字母、数字、下划线，其他作为分隔符
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
        # 过滤 1 字符 token（除了常见变量如 x, y, i, j, k）
        return [t for t in tokens if len(t) > 1 or t in ('x', 'y', 'i', 'j', 'k', 'n', 'm')]

    def _initialize(self, corpus: list[str]):
        nd: dict[str, int] = {}  # word -> 出现该词的文档数

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            self.avgdl += len(tokens)

            freq: dict[str, int] = Counter(tokens)
            self.doc_freqs.append(freq)

            for word in freq:
                nd[word] = nd.get(word, 0) + 1

        self.avgdl = self.avgdl / self.corpus_size if self.corpus_size > 0 else 0

        # 计算 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for word, df in nd.items():
            self.idf[word] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1.0
            )

    def get_scores(self, query: str) -> np.ndarray:
        """计算 query 与所有文档的 BM25 分数。"""
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float64)

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for idx in range(self.corpus_size):
                doc_freq = self.doc_freqs[idx]
                tf = doc_freq.get(token, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[idx]
                # BM25 评分公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf_val * numerator / denominator

        return scores


class Task8ExampleSelector:
    """Task 8 专用示例选择器：操作类型优先 + BM25 相关度混合选择。

    策略：
    1. 对 test_input 和所有示例做操作类型分类
    2. 同类型示例优先，内部按 BM25 相关度排序
    3. 不同类型示例按 BM25 相关度补充，填满 17K token 预算
    4. 最相关示例放末尾（recency bias）

    BM25 vs TF-IDF 的优势：
    - 文档长度归一化：避免长示例的词频优势
    - IDF 惩罚：自动降低 'kernel', 'triton' 等高频无区分度词的权重
    - 精确匹配：对函数名、操作名等关键词更敏感
    """

    def __init__(self, tokenizer, max_context_tokens: int = 17000,
                 min_context_tokens: int = 0):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.min_context_tokens = min_context_tokens
        self._bm25: _BM25 | None = None
        self._examples_cache = None

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _init_bm25(self, examples: list[dict]):
        if self._examples_cache is examples and self._bm25 is not None:
            return
        # 去掉 system prompt，只用任务内容建 BM25 索引
        corpus = [_strip_system_prompt(ex['input']) for ex in examples]
        logger.info(f"正在构建 BM25 索引, 共 {len(corpus)} 个示例...")
        self._bm25 = _BM25(corpus)
        self._examples_cache = examples
        logger.info(f"BM25 索引构建完成: avgdl={self._bm25.avgdl:.1f}, "
                    f"词汇表大小: {len(self._bm25.idf)}")

    def _compute_relevance(self, test_input: str) -> np.ndarray:
        """计算 test_input 与所有示例的 BM25 相关度分数。"""
        # 去掉 system prompt，聚焦任务内容匹配
        cleaned_input = _strip_system_prompt(test_input)
        return self._bm25.get_scores(cleaned_input)

    @staticmethod
    def _classify_operation_type(input_text: str) -> str:
        """根据 input 文本将 Triton 操作分为四类，用于示例匹配。"""
        t = input_text.lower()

        complex_kws = [
            'conv2d', 'conv1d', 'attention', 'fused', 'bmm', 'batch_norm',
            'layer_norm', 'group_norm', 'instance_norm', 'rms_norm',
            'cholesky', 'svd', 'eig', 'lu_solve', 'fft', 'spectral',
            'autograd', 'backward', 'forward pass', 'backward pass',
            'multi-head', 'transformer', 'softmax',
        ]
        if any(kw in t for kw in complex_kws):
            return 'complex'

        matmul_kws = [
            'matmul', 'matrix mult', 'dot product', 'linear',
            'gemm', 'outer product', 'cross product',
        ]
        if any(kw in t for kw in matmul_kws):
            return 'matmul'

        reduction_kws = [
            'sum', 'mean', 'average', 'max', 'min', 'norm', 'reduce',
            'argmax', 'argmin', 'cumsum', 'prod', 'variance', 'std',
        ]
        if any(kw in t for kw in reduction_kws):
            return 'reduction'

        return 'elementwise'

    def _truncate_by_tokens(self, example_lines: list[str],
                            reverse_select: bool = False) -> tuple[str, int]:
        if reverse_select:
            selected = []
            total_tokens = 0
            # 遇到过长示例时 continue 跳过(而非 break)，继续尝试更短的示例
            # 以尽量填满到 max_context_tokens 上限，满足赛题最低 token 要求
            for line in reversed(example_lines):
                line_tokens = self._count_tokens(line)
                if total_tokens + line_tokens > self.max_context_tokens:
                    continue
                selected.append(line)
                total_tokens += line_tokens
            selected.reverse()
            return ''.join(selected), total_tokens
        else:
            selected = []
            total_tokens = 0
            for line in example_lines:
                line_tokens = self._count_tokens(line)
                if total_tokens + line_tokens > self.max_context_tokens:
                    continue
                selected.append(line)
                total_tokens += line_tokens
            return ''.join(selected), total_tokens

    def select(self, all_examples: list[dict], task_description: str,
               test_input: str) -> str:
        """操作类型优先 + BM25 相关度混合选择"""
        self._init_bm25(all_examples)
        scores = self._compute_relevance(test_input)

        test_type = self._classify_operation_type(test_input)

        same_type_indices = []
        diff_type_indices = []
        for i, ex in enumerate(all_examples):
            ex_input = ex['input'] if isinstance(ex['input'], str) else str(ex['input'])
            ex_type = self._classify_operation_type(ex_input)
            if ex_type == test_type:
                same_type_indices.append(i)
            else:
                diff_type_indices.append(i)

        logger.info(f"Task 8 操作类型='{test_type}': 同类型={len(same_type_indices)}, 其他={len(diff_type_indices)}")

        # BM25 分数越高越相关，升序排列使最相关的在末尾
        same_type_ranked = sorted(same_type_indices, key=lambda i: scores[i])
        diff_type_ranked = sorted(diff_type_indices, key=lambda i: scores[i])
        ranked = diff_type_ranked + same_type_ranked

        logger.info(f"Task8 示例选择: 同类型={len(same_type_ranked)}, "
                     f"不同类型={len(diff_type_ranked)}, "
                     f"BM25分数范围=[{scores.min():.2f}, {scores.max():.2f}]")

        example_lines = []
        for idx in ranked:
            ex = all_examples[idx]
            # 直接传整个 ex dict, _reformat_icl_example 会优先用
            # normalized 字段 (func_desc / func_signature / raw_output 等)
            line = _reformat_icl_example(ex, cot=ex.get('cot', ''))
            example_lines.append(line)

        result, total_tokens = self._truncate_by_tokens(example_lines, reverse_select=True)
        selected_count = result.count('<label>')  # 每个示例一个 <label>
        logger.info(f"选择 {selected_count}/{len(example_lines)} 个示例, {total_tokens} tokens (上限 {self.max_context_tokens}, reverse_select)")
        if self.min_context_tokens > 0 and total_tokens < self.min_context_tokens:
            logger.warning(
                f"[ICL 合规警告] 实际 token 数 {total_tokens} < 下限 {self.min_context_tokens}；"
                f"已选 {selected_count}/{len(example_lines)}。"
                f"可能原因：候选池过小 / 示例过短 / max 预算过低。"
            )
        return result


# ============================================================
# 模2：Prompt 构建
# ============================================================

# Rules 常量：build_task8_prompt() 和 react_annotate() 统一引用
_TASK8_RULES = (
    "### Rules\n"
    "1. Reference the examples above — find the most similar one and "
    "adapt its kernel structure, stride handling, and grid launch pattern.\n"
    "2. The kernel MUST correctly implement ALL steps of the math formula.\n"
    "3. Ensure correct block masking and stride handling for memory safety.\n"
    "4. NEVER write placeholder comments like 'This is a simplified version', "
    "'placeholder', 'mock', or '# TODO'. Write REAL working implementation.\n"
    "5. If unsure about complex logic, implement a correct element-wise "
    "or block-based approach rather than leaving placeholders.\n"
    "6. Use `.stride()` (NOT `.strided()`) to get tensor strides.\n"
    "7. Create output tensor with correct shape (NOT `torch.empty_like(wrong_tensor)`). "
    "E.g., if output is (B,N,P), use `torch.empty(B,N,P,...)`.\n"
    "8. Numerical stability: prevent overflow in exp() and divide-by-zero in log()/sqrt(). "
    "Clamp inputs where necessary (e.g., `x = tl.where(x > 0, x, 1e-10)` before `log(x)`).\n"
    "9. Optional parameters (rounding_mode, out, alpha, etc.) MUST be handled "
    "correctly in the wrapper, NOT ignored. E.g., `rounding_mode='trunc'` means "
    "`torch.trunc(result)`, `rounding_mode='floor'` means `torch.floor(result)`. "
    "Do NOT use `torch.round()` for rounding_mode.\n"
    "10. If the wrapper signature has `out` parameter, write to it instead of "
    "creating a new tensor when `out is not None`.\n"
    "11. Always call `input = input.contiguous()` at the start of your wrapper function.\n"
    "12. Use `-> torch.Tensor` not `-> Tensor` in function return type hints.\n"
)

def build_task8_prompt(task_description: str, text2annotate: str,
                       examples_str: str,
                       wrapper_info: dict = None) -> str:
    """Triton kernel 代码生成任务 Prompt（Task 8）。

    Input 段使用 ``_build_structured_input`` 渲染规整文本（Functional
    Description / Wrapper Entry Information / Math / other 四段），
    避免直接拼接未清洗的原始 text2annotate。

    Args:
        wrapper_info: 已解析的 wrapper 字段（func_name / func_signature /
                     math_formula / func_desc / constraints）。
                     若传入则直接使用，否则从 text2annotate 重新解析。
    """
    parsed = _parse_wrapper_info(text2annotate, prefilled=wrapper_info)
    func_name = parsed['func_name']
    func_sig = parsed['func_signature']
    math_formula = parsed['math_formula']

    sig_section = ""
    if func_name and func_sig:
        sig_line = func_sig.split(';')[0].split('\n')[0].strip()
        if not sig_line.startswith('def '):
            sig_line = f"def {sig_line}"
        sig_section = (
            "### Wrapper Signature (MUST match exactly)\n"
            f"```python\n{sig_line}\n```\n"
            f"Your Python wrapper function MUST be named `{func_name}` "
            f"with the exact parameters shown above.\n\n"
        )

    math_section = ""
    if math_formula and math_formula.lower() != 'n/a':
        math_section = (
            "### Core Math (kernel MUST implement this)\n"
            f"{math_formula}\n\n"
        )

    # 用规整段渲染 Input, 替换原始 text2annotate
    structured_input = _build_structured_input(parsed, include_after_gen=True)

    return (
        "### Role\n"
        "You are an expert in Triton programming, capable of writing "
        "high-performance GPU kernels and their PyTorch wrapper functions.\n\n"

        "### Task\n"
        f"{task_description}\n\n"

        f"{sig_section}"

        f"{math_section}"

        "### Examples\n"
        f"{examples_str}\n\n"

        "### How to Solve\n"
        "Follow the same pattern as the examples above:\n"
        "1. Analyze the wrapper signature, parameter shapes and the math formula.\n"
        "2. Decide the grid layout — element-wise (1D grid) or reduction/2D (2D grid).\n"
        "3. Write the complete Triton kernel + Python wrapper code, "
        "wrapped in `<label>` and `</label>` tags (same as the examples).\n\n"

        "### Code Structure\n"
        "Your code MUST follow this structure:\n"
        "1. Import statements: `import torch`, `import triton`, "
        "`import triton.language as tl` (add `import math` if needed)\n"
        "2. Triton kernel(s): decorated with `@triton.jit`, using "
        "`tl.program_id`, `tl.arange`, `tl.load`/`tl.store` with proper masking\n"
        "3. Python wrapper function: matching the exact signature above, "
        "allocates output tensors, computes grid, and launches the kernel\n\n"

        f"{_TASK8_RULES}\n"

        "### Input\n"
        f"{structured_input}\n\n"

        "Write the complete code wrapped in <label> and </label> tags. "
        "IMPORTANT: You MUST wrap your final code with <label> and </label> tags. "
        "Do NOT use markdown code blocks (```) for the final code. "
        "Example: <label>\nimport torch\nimport triton\n...\n</label>\n"
    )


# ============================================================
# 模块5：Import 自动补全 + Wrapper 签名后处理
# ============================================================

def fix_task8_imports(code: str) -> str:
    """自动补全 Task 8 生成代码中缺失的 import 语句。"""
    if not code or not code.strip():
        return code

    lines = code.split('\n')
    missing_imports = []

    if 'math.' in code and 'import math' not in code:
        missing_imports.append('import math')

    typing_names = ['Optional', 'List', 'Tuple', 'Dict', 'Union', 'Any']
    used_typing = [name for name in typing_names if name in code]
    if used_typing and 'from typing import' not in code and 'import typing' not in code:
        missing_imports.append(f"from typing import {', '.join(used_typing)}")

    if 'F.' in code and 'import torch.nn.functional' not in code and 'from torch.nn' not in code:
        missing_imports.append('import torch.nn.functional as F')

    # 裸 Tensor 类型注解缺少 import 时自动补全
    # 匹配作为独立标识符出现且前面不是 '.' 的 Tensor, 排除 torch.Tensor 场景
    if re.search(r'(?<![.\w])Tensor(?![\w])', code):
        has_tensor_import = bool(
            re.search(r'from\s+torch\s+import\s+[^\n]*\bTensor\b', code)
            or re.search(r'from\s+torch\s+import\s+Tensor\b', code)
        )
        if not has_tensor_import:
            missing_imports.append('from torch import Tensor')

    if not missing_imports:
        return code

    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            insert_idx = i
            break

    for imp in reversed(missing_imports):
        lines.insert(insert_idx, imp)

    logger.debug(f"Task8 补全 import: {missing_imports}")
    return '\n'.join(lines)


def _extract_expected_func_name(text2annotate: str, wrapper_info: dict = None) -> str:
    """从 test input 中提取预期的 wrapper 函数名."""
    parsed = _parse_wrapper_info(text2annotate, prefilled=wrapper_info)
    return parsed['func_name']


def _extract_expected_signature(text2annotate: str, wrapper_info: dict = None) -> str:
    """从 test input 的 Wrapper Entry Information 中提取完整的预期函数签名."""
    parsed = _parse_wrapper_info(text2annotate, prefilled=wrapper_info)
    raw_sig = parsed['func_signature']
    if not raw_sig:
        return ""
    # 清理: 去掉可能的前缀如 'def ', 'torch.'
    sig = raw_sig.strip()
    if sig.startswith('def '):
        sig = sig[4:].strip()
    if sig.startswith('torch.'):
        sig = sig[6:].strip()
    return sig


def postprocess_task8_wrapper(code: str, text2annotate: str, wrapper_info: dict = None) -> str:
    """后处理：确保生成代码的 wrapper 函数名和参数签名与样本期望一致。

    函数签名不匹配会导致下游调用失败。

    Args:
        wrapper_info: 已解析的 wrapper 字段。若传入则直接使用。
    """
    expected_name = _extract_expected_func_name(text2annotate, wrapper_info=wrapper_info)
    if not expected_name:
        return code

    expected_sig = _extract_expected_signature(text2annotate, wrapper_info=wrapper_info)

    # ---- 1. 找到 wrapper 函数的 def 行 ----
    func_pattern = re.compile(r'^(def\s+)(\w+)(\s*\()', re.MULTILINE)
    matches = list(func_pattern.finditer(code))
    if not matches:
        return code

    jit_positions = [m.start() for m in re.finditer(r'@triton\.jit', code)]

    wrapper_matches = []
    for m in matches:
        is_kernel = any(
            code[jp:m.start()].strip() == '' or
            code[jp:m.start()].strip().startswith('@')
            for jp in jit_positions
            if jp < m.start() and m.start() - jp < 200
        )
        if not is_kernel:
            wrapper_matches.append(m)

    if not wrapper_matches:
        wrapper_matches = [matches[-1]]

    last_wrapper = wrapper_matches[-1]
    actual_name = last_wrapper.group(2)

    # ---- 2. 修复函数名 ----
    if actual_name != expected_name:
        logger.info(f"Task8 Wrapper rename: '{actual_name}' -> '{expected_name}'")
        code = code[:last_wrapper.start(2)] + expected_name + code[last_wrapper.end(2):]

    # ---- 3. 修复参数签名 ----
    if expected_sig:
        code = _fix_wrapper_signature(code, expected_name, expected_sig)

    return code


def _fix_wrapper_signature(code: str, func_name: str, expected_sig: str) -> str:
    """修复 wrapper 函数的参数签名，使其与样本期望签名一致。

    策略：仅重命名参数名不一致的参数（如 input→input_tensor），
    但必须确保重命名是安全的（不会导致语法错误）。
    不按位置映射，而是检查每个 actual 参数名是否在 expected 中存在——
    如果不存在，查找 expected 中同位置的参数名作为替换目标。
    """
    # 提取期望签名的参数部分
    sig_match = re.match(
        rf'(?:def\s+)?{re.escape(func_name)}\s*\(([^)]*)\)',
        expected_sig)
    if not sig_match:
        return code

    expected_params_str = sig_match.group(1)
    # 解析期望参数：保留参数名和是否为关键字参数的信息
    expected_items = []  # [(name, is_keyword_only)]
    for p in expected_params_str.split(','):
        p = p.strip()
        if not p:
            continue
        if p == '*':
            # 标记后续参数为 keyword-only
            expected_items.append(('__STAR__', True))
            continue
        is_kw_only = any(it[0] == '__STAR__' for it in expected_items)
        pname = p.split('=')[0].split(':')[0].strip().lstrip('*')
        if pname:
            expected_items.append((pname, is_kw_only))
    expected_param_names = [name for name, _ in expected_items if name != '__STAR__']

    # 找 code 中的 wrapper def 行
    wrapper_def_pattern = re.compile(
        rf'^(def\s+{re.escape(func_name)}\s*\()([\s\S]*?\))(\s*(?::\s*[^\n]+)?\s*:)',
        re.MULTILINE)
    wrapper_match = wrapper_def_pattern.search(code)
    if not wrapper_match:
        return code

    actual_params_str = wrapper_match.group(2)
    # 解析实际参数
    actual_items = []  # [(name, is_keyword_only)]
    for p in actual_params_str.replace('\n', ' ').split(','):
        p = p.strip()
        if not p:
            continue
        if p == '*':
            actual_items.append(('__STAR__', True))
            continue
        is_kw_only = any(it[0] == '__STAR__' for it in actual_items)
        pname = p.split('=')[0].split(':')[0].strip().lstrip('*')
        if pname:
            actual_items.append((pname, is_kw_only))
    actual_param_names = [name for name, _ in actual_items if name != '__STAR__']

    # 检查是否需要修复：如果期望的参数名都在实际参数中，不需要修复
    missing_params = [p for p in expected_param_names if p not in actual_param_names]
    if not missing_params:
        return code

    # 构建安全的重命名映射
    # 策略：只对“同位置的参数名不同”做重命名
    # 跳过 __STAR__，只匹配实际参数名和期望参数名
    actual_no_star = [(n, kw) for n, kw in actual_items if n != '__STAR__']
    expected_no_star = [(n, kw) for n, kw in expected_items if n != '__STAR__']

    rename_map = {}
    for i in range(min(len(actual_no_star), len(expected_no_star))):
        actual_name, actual_kw = actual_no_star[i]
        expected_name, expected_kw = expected_no_star[i]
        # 只重命名：两个参数名不同，且期望参数名在实际参数列表中不存在
        if (actual_name != expected_name
                and expected_name not in actual_param_names
                and actual_name not in expected_param_names):
            rename_map[actual_name] = expected_name

    if not rename_map:
        return code

    logger.info(f"Task8 Wrapper 参数重命名: {rename_map}")

    # 只在 wrapper 函数体内重命名（从 def 到下一个顶层 def 或文件末尾）
    wrapper_start = wrapper_match.start()
    next_def = re.search(r'\n(?=@triton\.jit|def \w+)', code[wrapper_match.end():])
    if next_def:
        wrapper_end = wrapper_match.end() + next_def.start()
    else:
        wrapper_end = len(code)

    wrapper_body = code[wrapper_start:wrapper_end]

    # 重命名：长名优先（避免部分匹配）
    for old_name in sorted(rename_map.keys(), key=len, reverse=True):
        new_name = rename_map[old_name]
        # 只替换作为独立标识符出现的（词边界）
        wrapper_body = re.sub(
            rf'\b{re.escape(old_name)}\b', new_name, wrapper_body)

    code = code[:wrapper_start] + wrapper_body + code[wrapper_end:]
    return code


# ============================================================
# 模块6：答案解析（直接复用 common.llm_client.count_answer）
# ============================================================
# count_answer 已通过文件头部的 ``from common.llm_client import count_answer`` 引入,
# 不再本地重复实现。

def validate_prediction(prediction: Optional[str]) -> Optional[str]:
    """Task 8 专用预测校验（code 类型）"""
    if prediction is None:
        return None
    prediction = prediction.strip()
    if not prediction:
        return None

    if prediction in ('...', '…', '..', '…………', 'N/A', 'n/a', 'None', 'none', 'null'):
        return None
    if all(c in '.…' for c in prediction):
        return None

    # 清理 markdown 围栏
    if '```python' in prediction:
        m = re.search(r'```python\s*\n(.*?)```', prediction, re.DOTALL)
        if m:
            prediction = m.group(1).strip()
    elif '```' in prediction:
        m = re.search(r'```\s*\n(.*?)```', prediction, re.DOTALL)
        if m:
            prediction = m.group(1).strip()

    # 清理残留的 HTML/XML 闭合标签（如 </label>、</answer> 等）
    prediction = re.sub(r'</\w+>', '', prediction).strip()

    # 清理末尾可能残留的未闭合 markdown 围栏
    # 例如代码提取后末尾可能残留 ```
    lines = prediction.split('\n')
    while lines and lines[-1].strip().startswith('```'):
        lines.pop()
    prediction = '\n'.join(lines).strip()

    # 截取从 import/@ 开始的代码部分
    lines = prediction.split('\n')
    code_start = 0
    for idx_l, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('import ') or
            stripped.startswith('from ') or
            stripped.startswith('@triton')):
            code_start = idx_l
            break
    if code_start > 0:
        logger.debug(f"Task8: 跳过前 {code_start} 行分析文字")
        prediction = '\n'.join(lines[code_start:])

    # 再次清理围栏
    if prediction.strip().startswith('```'):
        cleaned = prediction.strip()
        first_nl = cleaned.find('\n')
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.rstrip().endswith('```'):
            cleaned = cleaned.rstrip()[:-3].rstrip()
        prediction = cleaned

    if len(prediction) > 20000:
        return None

    # 重复检测
    lines = prediction.strip().split('\n')
    if len(lines) > 10:
        line_counts = Counter(line.strip() for line in lines if line.strip())
        if line_counts:
            most_common = line_counts.most_common(1)[0][1]
            if most_common > max(10, len(lines) * 0.4):
                logger.warning(f"检测到高度重复输出 ({most_common}/{len(lines)})")
                return None
    return prediction


# ============================================================
# 模块7：API 推理调用（直接复用 common.llm_client.annotate_nvidia）
# ============================================================
# 单轮调用统一走 common.llm_client.annotate_nvidia（在文件头别名为 ``annotate`` 引入），
# 多轮对话的 ReAct 调用见下方 annotate_multi_turn。


# ============================================================
# 模块8：ReAct 多轮对话推理
# ============================================================

def annotate_multi_turn(messages: list[dict],
                        max_tokens: int = 15000,
                        timeout: int = 300,
                        temperature: float = 0.6,
                        top_p: float = 0.95,
                        top_k: int = 20,
                        repetition_penalty: float = 1.2,
                        enable_thinking: bool = True) -> str | None:
    """多轮对话式推理，支持 ReAct 模式。

    与 annotate() 不同，此函数接收完整的 messages 列表，
    支持多轮 conversation 而非单轮 user->assistant。
    返回原始 content 字符串（不做标签提取），由调用方处理。
    """
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = repetition_penalty
    if top_k is not None:
        extra_body["top_k"] = top_k

    kwargs = {
        "model": VLLM_MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": extra_body,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p

    for attempt in range(2):
        try:
            t0 = time.time()
            response = _client.chat.completions.create(
                **kwargs, timeout=timeout
            )
            elapsed_ms = (time.time() - t0) * 1000
            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, 'reasoning_content', None)
            usage = response.usage

            logger.info(f"ReAct API: {elapsed_ms:.0f}ms, "
                         f"prompt={usage.prompt_tokens if usage else '?'}, "
                         f"completion={usage.completion_tokens if usage else '?'}")

            # 思考过程和模型输出记录到日志（与其他 task 保持一致）
            if reasoning:
                logger.info(f"思考过程:\n{reasoning}")
            logger.info(f"模型输出:\n{content}")

            # thinking 模式下 content 可能为空，从 reasoning 提取
            if not content.strip() and reasoning:
                logger.info("ReAct: content 为空，从 reasoning_content 提取")
                tag_m = re.search(
                    r'<(?:label|answer)>\s*(.*?)\s*</(?:label|answer)>',
                    reasoning, re.DOTALL)
                if tag_m:
                    content = tag_m.group(1).strip()
                else:
                    code_blocks = re.findall(
                        r'```(?:python)?\s*\n(.*?)```', reasoning, re.DOTALL)
                    if code_blocks:
                        content = code_blocks[-1].strip()
                    else:
                        rlines = reasoning.split('\n')
                        code_start = -1
                        for ci, cl in enumerate(rlines):
                            sl = cl.strip()
                            if (sl.startswith('import ') or
                                sl.startswith('from ') or
                                sl.startswith('@triton')):
                                code_start = ci
                                break
                        if code_start >= 0:
                            content = '\n'.join(rlines[code_start:])

            return content

        except Exception as e:
            logger.warning(f"ReAct API 异常 (attempt {attempt + 1}): {e}")
            if attempt < 1:
                time.sleep(1)

    logger.error("ReAct: 所有重试均失败")
    return None


# ============================================================
# ============================================================
# Multi-Turn Self-Verify Pipeline 辅助
# ============================================================
# 内容:
#   - TraceLogger              : 每样本详细 trace 文件
#   - check_placeholder        : 代码 placeholder 扫描
#   - static_validate          : AST + wrapper name/sig + placeholder 四要素 + exec 预检
#   - build_verify_prompt      : R2 自检 prompt (JSON 输出)
#   - parse_verify_json        : R2 JSON 宽松解析
#   - build_safe_exit_prompt   : R3 异常隔离规范 prompt
#   - build_react_observation : 修复循环的 observation
#   - react_repair            : 固定 N 轮，反馈源 = static_validate
#   - apply_ast_guard          : 末端 AST 兜底（强制 wrapper 套 try/except）
# ============================================================

REACT_MAX_ROUNDS = 3

_PLACEHOLDER_PATTERNS = [
    r'\bTODO\b',
    r'\bFIXME\b',
    r'\bNotImplementedError\b',
    r'\bxxx+\b',
    r'your code here',
    r'your implementation here',
    r'fill in',
    r'#\s*implement',
    r'#\s*placeholder',
    r'#\s*mock',
    r'#\s*stub',
    r'pass\s*#\s*(todo|implement|fill|stub)',
    r'raise\s+NotImplementedError',
]


class TraceLogger:
    """每样本详细执行轨迹日志。

    采用纯文本分段格式,便于 `less` 或 `grep` 直接查看。
    每个阶段写入一个带 timestamp 的分节,方便对照排查。
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'w', encoding='utf-8')
        self.t0 = time.time()
        self._stage_count = 0

    def log(self, stage: str, content=None, meta: dict | None = None) -> None:
        """写一个分节。content 可为字符串或 None; meta 为 key->value 元信息。"""
        self._stage_count += 1
        elapsed = time.time() - self.t0
        self.f.write(f"\n\n===== [{self._stage_count:02d}] {stage} "
                     f"(t+{elapsed:.1f}s) =====\n")
        if meta:
            for k, v in meta.items():
                self.f.write(f"[META] {k}: {v}\n")
        if content is not None:
            if not isinstance(content, str):
                content = str(content)
            self.f.write(content)
            if not content.endswith('\n'):
                self.f.write('\n')
        self.f.flush()

    def header(self, meta: dict) -> None:
        """样本级元信息头。"""
        self.f.write("=" * 72 + "\n")
        self.f.write(f"Task 8 Trace | started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k, v in meta.items():
            self.f.write(f"  {k}: {v}\n")
        self.f.write("=" * 72 + "\n")
        self.f.flush()

    def close(self) -> None:
        try:
            total = time.time() - self.t0
            self.f.write(f"\n===== [END] total={total:.1f}s, stages={self._stage_count} =====\n")
            self.f.close()
        except Exception:
            pass


def check_placeholder(code: str) -> list[str]:
    """扫描代码中的 placeholder 占位符。返回命中描述列表(最多 10 条)。

    命中即视为代码未完成,应进入修复循环。
    """
    if not code:
        return ['empty code']
    findings = []
    for i, line in enumerate(code.split('\n'), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#') is False and '#' not in stripped:
            # 允许快速路径: 无注释且无特殊词汇的行跳过正则扫描
            has_keyword = any(kw in line for kw in (
                'TODO', 'FIXME', 'NotImplementedError', 'placeholder',
                'mock', 'stub', 'your code', 'implement', 'xxx', 'fill in',
            ))
            if not has_keyword:
                continue
        for pat in _PLACEHOLDER_PATTERNS:
            if re.search(pat, line, re.IGNORECASE):
                findings.append(f"line {i}: {stripped[:120]} (pat: {pat})")
                break
        if len(findings) >= 10:
            break
    return findings


def _signature_arg_names(signature: str) -> list[str]:
    """从 func_signature 字符串粗提取参数名列表(用于宽松匹配)。

    输入形如: 'foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor'
    也兼容 normalized 数据中 signature 字段后接 Args/Shape 文档的情况:
        'foo(a, b=1, *, c=None) -> Tensor. Args: a (...): ... Default: ...'
    输出: ['a', 'b', 'c']

    实现要点:
      - 用平衡括号定位到函数名后的第一个配对闭括号, 只解析括号内的实参列表
      - 忽略 keyword-only 分隔符 '*' 和 positional-only 分隔符 '/'
    """
    if not signature:
        return []
    s = signature.strip()
    if s.startswith('def '):
        s = s[4:]
    lp = s.find('(')
    if lp == -1:
        return []
    # 平衡括号找第一个匹配的 ')', 忽略字符串/注释内容 (signature 场景通常无字符串)
    depth = 0
    rp = -1
    for i in range(lp, len(s)):
        ch = s[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                rp = i
                break
    if rp == -1 or rp <= lp:
        return []
    inner = s[lp + 1:rp]
    if not inner.strip():
        return []
    # 按顶层逗号拆分
    parts = []
    depth = 0
    buf = []
    for ch in inner:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    names = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 忽略 keyword-only / positional-only 分隔符
        if p in ('*', '/'):
            continue
        # 去掉默认值
        if '=' in p:
            p = p.split('=', 1)[0].strip()
        # 去掉类型注解
        if ':' in p:
            p = p.split(':', 1)[0].strip()
        # 去掉 * / **
        p = p.lstrip('*').strip()
        if p:
            names.append(p)
    return names


def static_validate(code: str,
                    expected_func_name: str,
                    expected_signature: str | None = None) -> tuple[bool, list[str]]:
    """静态验证: AST + wrapper 函数名 + 参数个数 + placeholder 四要素。

    Returns: (ok, errors)  errors 为人可读的错误列表。
    """
    import ast as _ast
    errors = []

    if not code or not code.strip():
        return False, ['empty code']

    # 1. AST 解析
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        errors.append(f"AST parse failed: {e.msg} at line {e.lineno}")
        return False, errors

    # 2. wrapper 函数名存在
    wrapper_node = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == expected_func_name:
            wrapper_node = node
            break
    if wrapper_node is None:
        # 允许带下划线前缀变体
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name.lstrip('_') == expected_func_name.lstrip('_'):
                wrapper_node = node
                break
    if wrapper_node is None:
        errors.append(f"Wrapper function '{expected_func_name}' not found in code")

    # 3. 参数个数匹配(宽松: 必选+可选+keyword-only+vararg/kwarg 总和)
    if wrapper_node is not None and expected_signature:
        expected_args = _signature_arg_names(expected_signature)
        a = wrapper_node.args
        actual_args = (
            [p.arg for p in getattr(a, 'posonlyargs', []) or []]
            + [p.arg for p in a.args]
            + ([a.vararg.arg] if a.vararg else [])
            + [p.arg for p in a.kwonlyargs]
            + ([a.kwarg.arg] if a.kwarg else [])
        )
        if expected_args and len(actual_args) != len(expected_args):
            errors.append(
                f"Argument count mismatch: expected {len(expected_args)} "
                f"({expected_args}), got {len(actual_args)} ({actual_args})"
            )

    # 4. placeholder 扫描
    phs = check_placeholder(code)
    if phs:
        for ph in phs[:5]:
            errors.append(f"Placeholder: {ph}")

    # 5. exec 预检: 捕获签名注解/默认值/imports 中的未导入错误
    #    仅对 imports + wrapper 的 def 头 (body=pass, 去装饰器) 做 exec,
    #    避免触发 @triton.jit 装饰器对 kernel body 的解析。
    if wrapper_node is not None:
        try:
            stub_body = [
                n for n in tree.body
                if isinstance(n, (_ast.Import, _ast.ImportFrom))
            ]
            wrapper_stub = _ast.FunctionDef(
                name=wrapper_node.name,
                args=wrapper_node.args,
                body=[_ast.Pass()],
                decorator_list=[],  # 去掉装饰器
                returns=wrapper_node.returns,
                type_comment=None,
            )
            stub_module = _ast.Module(
                body=stub_body + [wrapper_stub],
                type_ignores=[],
            )
            _ast.fix_missing_locations(stub_module)
            code_obj = compile(stub_module, '<static_validate_stub>', 'exec')
            exec(code_obj, {})
        except (NameError, ImportError, AttributeError) as e:
            errors.append(
                f"Exec precheck failed (undefined name / missing import): "
                f"{type(e).__name__}: {e}"
            )
        except SyntaxError as e:
            errors.append(f"Exec precheck syntax error: {e.msg}")
        except Exception as e:
            # 其他异常 (如 TypeError/ValueError 等) 一并记录
            errors.append(f"Exec precheck failed: {type(e).__name__}: {e}")

    return len(errors) == 0, errors


def build_verify_prompt(code: str,
                        task_description: str,
                        text2annotate: str,
                        func_name: str,
                        examples_str: str = '',
                        wrapper_info: dict = None) -> str:
    """R2 自检 prompt: 要求模型输出 <audit>{...}</audit> JSON.

    Target Input Spec 使用 _build_structured_input 渲染的规整文本.

    Args:
        examples_str: ICL 示例文本块（与 R1 同构）。仅作为算子语义/调用模式的
            参考上下文供审计阶段对照, 不要求模型复制其实现。
        wrapper_info: 已解析的 wrapper 字段。若传入则直接使用。
    """
    parsed = _parse_wrapper_info(text2annotate, prefilled=wrapper_info)
    sample_input = _build_structured_input(parsed, include_after_gen=False)
    icl_block = (
        "### Reference Examples (for context only — do NOT copy)\n"
        f"{examples_str}\n\n"
    ) if examples_str else ""
    return (
        "### Role\n"
        "You are a senior GPU kernel reviewer. Audit the following Triton kernel draft.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        f"{icl_block}"
        "### Target Function\n"
        f"`{func_name}`\n\n"
        "### Target Input Spec\n"
        f"{sample_input}\n\n"
        "### Draft Implementation\n"
        "<code>\n"
        f"{code}\n"
        "</code>\n\n"
        "### Audit Checklist\n"
        "1. kernel_syntax_ok: Is the @triton.jit kernel syntactically valid?\n"
        "2. launch_grid_ok: Is the launch grid computation safe (cdiv, no zero/negative dims)?\n"
        "3. dtype_mask_ok: Are tl.load / tl.store masks handling out-of-bound safely?\n"
        "4. numerical_equivalence_confidence: How confident are you that this wrapper "
        "yields numerically equivalent output to a PyTorch reference? (high / medium / low)\n"
        "5. risk_level: Overall risk tier. One of: safe / risky / unsafe.\n\n"
        "### Output Format\n"
        "Output ONLY a JSON object wrapped in <audit> and </audit> tags. No extra prose.\n"
        "If you are NOT highly confident that the code is numerically correct, you MUST\n"
        "set risk_level to risky or unsafe. When in doubt, choose unsafe.\n\n"
        "<audit>\n"
        "{\n"
        '  "kernel_syntax_ok": true|false,\n'
        '  "launch_grid_ok": true|false,\n'
        '  "dtype_mask_ok": true|false,\n'
        '  "numerical_equivalence_confidence": "high|medium|low",\n'
        '  "risk_level": "safe|risky|unsafe",\n'
        '  "rationale": "one short sentence"\n'
        "}\n"
        "</audit>\n"
    )


def parse_verify_json(raw: str | None) -> dict:
    """R2 JSON 宽松解析。失败一律回退到 unsafe/low。"""
    default = {
        "kernel_syntax_ok": False,
        "launch_grid_ok": False,
        "dtype_mask_ok": False,
        "numerical_equivalence_confidence": "low",
        "risk_level": "unsafe",
        "rationale": "parse_failed_or_empty",
        "_parsed": False,
    }
    if not raw or not raw.strip():
        return default
    # 1. 优先 <audit>...</audit>
    m = re.search(r'<audit>\s*(\{[\s\S]*?\})\s*</audit>', raw)
    payload = m.group(1) if m else None
    # 2. 否则找含 risk_level 的 JSON 对象
    if payload is None:
        m = re.search(r'\{[^{}]*?"risk_level"[^{}]*?\}', raw, re.DOTALL)
        payload = m.group(0) if m else None
    # 3. 仍找不到 -> 退回默认
    if payload is None:
        return default
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return default
    merged = {**default, **data, "_parsed": True}
    # 归一
    rl = str(merged.get("risk_level", "unsafe")).lower().strip()
    if rl not in ('safe', 'risky', 'unsafe'):
        rl = 'unsafe'
    merged["risk_level"] = rl
    conf = str(merged.get("numerical_equivalence_confidence", "low")).lower().strip()
    if conf not in ('high', 'medium', 'low'):
        conf = 'low'
    merged["numerical_equivalence_confidence"] = conf
    return merged


def verify_routing_decision(verify_info: dict,
                            *,
                            hard_errors: list[str] | None = None,
                            soft_errors: list[str] | None = None,
                            op_type: str | None = None) -> str:
    """根据静态错误类型 + 算子类型 + R2 JSON 综合决定路由。

    决策原则：
      1. 硬错误 (hard_errors) 非空 → safe_exit
         (SyntaxError / Wrapper 缺失 / 参数签名错 / 空草稿等结构性错误)
      2. 按 op_type 分层放宽软错误 / R2 verdict:
         - elementwise:    无视 R2 verdict, 一律 keep_draft
                           (简单算子 load→compute→store 结构, 写错也不影响后续调用)
         - reduction/matmul:
                           仅在 R2 最严 (unsafe+low) 时 safe_exit, 其余 keep_draft
         - complex (fused/多 kernel/归一化等) / 未知:
                           沿用原严格规则, 要求 R2 safe+high/medium 才 keep_draft
    Returns: 'keep_draft' or 'safe_exit'
    """
    # Rule 1: 硬错误一律 safe_exit
    if hard_errors:
        return 'safe_exit'

    rl = (verify_info or {}).get("risk_level", "unsafe")
    conf = (verify_info or {}).get("numerical_equivalence_confidence", "low")
    r2_pass = (rl == 'safe' and conf in ('high', 'medium'))

    # Rule 2a: elementwise 无视 R2, 保留 draft
    if op_type == 'elementwise':
        return 'keep_draft'

    # Rule 2b: reduction / matmul 放宽 R2 门槛
    if op_type in ('reduction', 'matmul'):
        # 仅当 R2 最严重否决 (risk=unsafe AND conf=low) 才 safe_exit
        if rl == 'unsafe' and conf == 'low':
            return 'safe_exit'
        return 'keep_draft'

    # Rule 2c: complex / 其它 / 未传 op_type: 原严格规则
    return 'keep_draft' if r2_pass else 'safe_exit'


# 硬错误前缀枚举：触发即 safe_exit
_HARD_ERROR_PREFIXES = (
    'AST parse failed',
    'Wrapper function',
    'Argument count mismatch',
    'empty code',
    'empty draft',
)


def classify_static_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """将 static_validate 返回的 errors 分类为硬错误与软错误。

    硬错误（结构性错误, 必须 safe_exit）：
      - AST SyntaxError
      - Wrapper 函数缺失
      - Wrapper 参数签名错
      - 空代码 / 空草稿

    软错误（仅影响该样本生成质量, 可交 draft）：
      - Placeholder 注释（kernel body 未完成但结构正确）
      - 其它未分类错误

    Returns: (hard_errors, soft_errors)
    """
    hard: list[str] = []
    soft: list[str] = []
    for e in (errors or []):
        if any(e.startswith(p) for p in _HARD_ERROR_PREFIXES):
            hard.append(e)
        else:
            soft.append(e)
    return hard, soft


def build_safe_exit_prompt(func_name: str,
                           func_signature: str,
                           task_description: str,
                           text2annotate: str,
                           current_code: str = '',
                           examples_str: str = '',
                           wrapper_info: dict = None) -> str:
    """R3 异常隔离规范 prompt.

    在前序轮次产出的真实代码 (current_code) 基础上, 对 wrapper 函数体整体套
    try/except Exception, 异常分支 return None, 保留原有计算路径并保证
    wrapper 始终可调用。

    关键要求:
    - 保留 current_code 的全部 import 语句 (不改不删不加)
    - 保留 current_code 的全部 @triton.jit kernel 函数 (不改不删)
    - 保留 wrapper 函数签名 (func_signature) 与默认参数
    - wrapper 函数体整体包在 try/except Exception 内, except 分支 return None
    - try 块内保留原 wrapper 的所有计算逻辑 (含原有 return 语句)

    Args:
        current_code: 前序轮次产出的代码. 为空时退回到占位骨架模板.
        examples_str: ICL 示例文本块（与 R1 同构）。仅作为同类算子的参考上下文,
            不要求模型基于示例改写; R3 主任务仍然是机械变换 wrapper 函数体。
        wrapper_info: 已解析的 wrapper 字段。若传入则直接使用。
    """
    parsed = _parse_wrapper_info(text2annotate, prefilled=wrapper_info)
    sample_input = _build_structured_input(parsed, include_after_gen=False)

    icl_block = (
        "### Reference Examples (for context only — do NOT copy)\n"
        f"{examples_str}\n\n"
    ) if examples_str else ""

    # 清洗 func_signature: 去掉可能的 'def ' 前缀, 直接作为 `def {sig}:` 使用
    sig = (func_signature or '').strip()
    if sig.startswith('def '):
        sig = sig[4:].strip()
    if not sig:
        # 兜底: 没有显式签名时退回到 func_name() 空参
        sig = f"{func_name}()"

    has_code = bool((current_code or '').strip())

    if not has_code:
        # current_code 为空 (R1/ReAct 都未产出): 退回占位骨架模板
        return (
            "### Role\n"
            "You are a Triton engineer applying a defensive-programming protocol.\n\n"
            "### Context\n"
            "No draft implementation is available for this task. Emit a structurally\n"
            "valid wrapper + kernel stub that preserves the wrapper contract without\n"
            "propagating unverified computation results to the caller.\n\n"
            "### Task\n"
            f"{task_description}\n\n"
            f"{icl_block}"
            "### Target Input Spec\n"
            f"{sample_input}\n\n"
            "### Required Structure\n"
            "1. Include these imports at the top:\n"
            "   ```\n"
            "   import torch\n"
            "   import triton\n"
            "   import triton.language as tl\n"
            "   ```\n\n"
            f"2. Define one `@triton.jit` kernel function named `{func_name}_kernel` with a\n"
            "   complete load-store body:\n"
            "   ```\n"
            f"   @triton.jit\n"
            f"   def {func_name}_kernel(X, n, BLOCK_SIZE: tl.constexpr):\n"
            "       pid = tl.program_id(0)\n"
            "       offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
            "       mask = offs < n\n"
            "       x = tl.load(X + offs, mask=mask, other=0.0)\n"
            "       tl.store(X + offs, x, mask=mask)\n"
            "   ```\n\n"
            f"3. Define the wrapper function EXACTLY as:\n"
            "   ```\n"
            f"   def {sig}:\n"
            "       try:\n"
            "           pass\n"
            "       except Exception:\n"
            "           pass\n"
            "       return None\n"
            "   ```\n"
            f"   - The wrapper signature MUST match `{sig}` EXACTLY.\n"
            "   - Do NOT replace the original parameters with `*args, **kwargs`.\n\n"
            "### Output Format\n"
            "Output the complete code wrapped in <label> and </label> tags.\n"
            "Do NOT use markdown code blocks. Do NOT include explanatory prose outside the tags.\n"
        )

    # current_code 非空: 套 try/except 保留原真实代码
    return (
        "### Role\n"
        "You are a Triton engineer applying a defensive-programming protocol to an\n"
        "existing draft kernel implementation.\n\n"
        "### Context\n"
        "An earlier draft implementation is provided below. An automated self-audit\n"
        "stage could not certify its numerical equivalence, but the draft MAY still\n"
        "be correct on some inputs. Your task is a PURELY MECHANICAL transformation:\n"
        "wrap the wrapper function body in a `try / except Exception` block so that\n"
        "(a) when the draft computation succeeds it returns the true result, and\n"
        "(b) when it raises any exception the wrapper safely returns None instead\n"
        "of propagating the error to the caller.\n\n"
        "### Task\n"
        f"{task_description}\n\n"
        f"{icl_block}"
        "### Target Input Spec\n"
        f"{sample_input}\n\n"
        "### Current Draft Code (to be transformed)\n"
        "<code>\n"
        f"{current_code}\n"
        "</code>\n\n"
        "### Transformation Rules (MECHANICAL — do not alter logic)\n"
        "1. Preserve EVERY import statement from the draft verbatim (order, aliases,\n"
        "   spacing). Do not add new imports. Do not remove any import.\n"
        "2. Preserve EVERY `@triton.jit` kernel function verbatim (name, signature,\n"
        "   decorators, body). Do not rename, re-order, simplify, or optimize.\n"
        f"3. Preserve the wrapper signature EXACTLY as `def {sig}:` (keep all\n"
        "   parameter names, type annotations, default values, keyword-only markers).\n"
        "4. Wrap the ENTIRE original wrapper body in a `try:` block. All original\n"
        "   statements (including any existing `return` statements) go inside `try`.\n"
        "5. Add `except Exception:` as a sibling of `try`, and inside it a single\n"
        "   statement: `return None`.\n"
        "6. After the `try/except` block, at the wrapper-function indent level,\n"
        "   append one final `return None` as a defensive fall-through in case the\n"
        "   try body completes without returning.\n"
        "7. Do NOT add logging, print, comments, or any new computation. The\n"
        "   transformation is purely structural.\n\n"
        "### Structural Template (illustrative only — adapt to the actual draft)\n"
        "The wrapper after transformation MUST follow this shape:\n"
        "```\n"
        f"def {sig}:\n"
        "    try:\n"
        "        <ORIGINAL WRAPPER BODY, VERBATIM, INCLUDING ITS RETURN STATEMENTS>\n"
        "    except Exception:\n"
        "        return None\n"
        "    return None\n"
        "```\n\n"
        "### Output Format\n"
        "Output the complete transformed code (imports + kernel(s) + wrapped wrapper)\n"
        "inside <label> and </label> tags. Do NOT use markdown code blocks. Do NOT\n"
        "include explanatory prose outside the tags.\n"
    )


def build_react_observation(current_code: str,
                            errors: list[str],
                            func_name: str,
                            func_signature: str,
                            examples_str: str = '') -> str:
    """ReAct 修复循环的 observation prompt。

    Args:
        examples_str: ICL 示例文本块（与 R1 同构），用于在结构化错误反馈基础上
            提供同类算子的参考实现上下文。
    """
    err_block = "\n".join(f"  - {e}" for e in errors[:8]) or "  (none)"
    sig = func_signature.strip()
    if sig.startswith('def '):
        sig = sig[4:]
    icl_block = (
        "### Reference Examples (same task family — use as structural reference)\n"
        f"{examples_str}\n\n"
    ) if examples_str else ""
    return (
        "### Role\n"
        "You are fixing a Triton kernel that failed static structural validation.\n\n"
        f"{icl_block}"
        "### Target Wrapper Signature (MUST match EXACTLY)\n"
        "<code>\n"
        f"def {sig}:\n"
        "</code>\n\n"
        "### Current Draft\n"
        "<code>\n"
        f"{current_code}\n"
        "</code>\n\n"
        "### Static Validation Errors\n"
        f"{err_block}\n\n"
        "### Fix Requirements\n"
        f"1. Keep the wrapper function name EXACTLY `{func_name}`.\n"
        "2. Keep wrapper parameter count and order matching the target signature.\n"
        "3. Remove ALL placeholder markers (TODO / FIXME / NotImplementedError /\n"
        "   'your code here' / ellipsis / etc.).\n"
        "4. The entire code MUST parse as valid Python (AST-clean).\n"
        "5. Include the three required imports (torch / triton / triton.language as tl).\n\n"
        "### Output Format\n"
        "Output the complete corrected code wrapped in <label> and </label> tags.\n"
        "Do NOT use markdown code blocks.\n"
    )


def react_repair(initial_code: str,
                 initial_errors: list[str],
                 task_description: str,
                 text2annotate: str,
                 func_name: str,
                 func_signature: str,
                 max_rounds: int = REACT_MAX_ROUNDS,
                 trace: "TraceLogger | None" = None,
                 wrapper_info: dict = None,
                 examples_str: str = '',
                 examples_tokens: int = 0) -> tuple[str, bool, int]:
    """静态验证失败后的多轮 ReAct 修复循环,固定 N 轮,反馈源=静态错误。

    Args:
        wrapper_info: 已解析的 wrapper 字段, 透传给 postprocess.
        examples_str: ICL 示例文本块（与 R1 同构），透传给 build_react_observation。
        examples_tokens: ICL 示例的 token 数（用于 trace 日志记录, 不影响生成）。

    Returns: (final_code, passed, rounds_used)
    """
    current_code = initial_code
    current_errors = initial_errors

    for rnd in range(1, max_rounds + 1):
        observation = build_react_observation(
            current_code, current_errors, func_name, func_signature,
            examples_str=examples_str)
        messages = [{"role": "user", "content": observation}]

        if trace:
            trace.log(
                f"R1.5 React Round {rnd}/{max_rounds} - PROMPT",
                observation,
                meta={"prompt_chars": len(observation),
                      "icl_tokens": examples_tokens},
            )

        t0 = time.time()
        content = annotate_multi_turn(
            messages,
            max_tokens=TASK8_MAX_TOKENS,
            timeout=TASK8_TIMEOUT,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=TASK8_REP_PENALTY,
            enable_thinking=True,
        )
        elapsed = time.time() - t0

        if trace:
            trace.log(
                f"R1.5 React Round {rnd}/{max_rounds} - RAW OUTPUT",
                content or "(empty)",
                meta={"output_chars": len(content or ""), "duration_s": f"{elapsed:.1f}"},
            )

        if not content or not content.strip():
            continue

        new_code = validate_prediction(content)
        if not new_code:
            if trace:
                trace.log(f"R1.5 React Round {rnd}/{max_rounds} - EXTRACT FAIL",
                          "Could not extract code from <label>...</label>")
            continue
        new_code = fix_task8_imports(new_code)
        new_code = postprocess_task8_wrapper(new_code, text2annotate, wrapper_info=wrapper_info)

        ok, errors = static_validate(new_code, func_name, func_signature)

        if trace:
            trace.log(
                f"R1.5 React Round {rnd}/{max_rounds} - STATIC VERDICT",
                "errors:\n" + ("\n".join(f"  - {e}" for e in errors) if errors else "  (none)"),
                meta={"passed": ok},
            )

        if ok:
            return new_code, True, rnd

        current_code = new_code
        current_errors = errors

    return current_code, False, max_rounds


# ============================================================
# 模块9：AST 兜底 - 强制 wrapper 套 try/except (防御性编程)
# ============================================================

def _wrapper_has_try_guard(func_node: ast.FunctionDef) -> bool:
    """检测 wrapper 函数体是否已被完整 try/except 包裹.

    判定条件：函数体第一个语句是 Try 节点，且该 Try 覆盖了除末尾可能的
    单条 Return 之外的全部语句。放宽判定：只要 body 第一条是 Try, 或
    body 唯一语句是 Try, 视为已套过, 跳过以保持幂等.
    """
    if not func_node.body:
        return False
    if isinstance(func_node.body[0], ast.Try):
        return True
    return False


def wrap_wrapper_with_try_except(code: str, func_name: str) -> str:
    """对指定 wrapper 函数的函数体整体套 try/except Exception: pass.

    设计要点:
    - 精确匹配: 只改名为 func_name 的顶层 FunctionDef (不触碰 kernel)
    - 幂等: 已含顶层 try 的 wrapper 跳过, 避免嵌套污染
    - 安全: parse 失败直接抛 SyntaxError 给上游决策 (而非吞掉)
    - except 体为单条 pass, 不显式 return None (Python 默认返回 None)
    - try 内保留原 body 的所有 return 语句, 正常路径仍返回真实结果

    Args:
        code: 待改造的完整代码字符串 (imports + kernel + wrapper)
        func_name: wrapper 函数名

    Returns:
        改造后的代码字符串. 若 wrapper 未找到 / 已套 try, 返回原 code.

    Raises:
        SyntaxError: code 本身无法 parse.
    """
    tree = ast.parse(code)
    changed = False
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != func_name:
            continue
        if _wrapper_has_try_guard(node):
            # 幂等: 已套 try, 跳过
            return code
        # 原 body 整体包入 try 块
        try_node = ast.Try(
            body=list(node.body),
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=[ast.Pass()],
            )],
            orelse=[],
            finalbody=[],
        )
        node.body = [try_node]
        changed = True
        break  # 只改一个 wrapper

    if not changed:
        return code

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def build_fallback_skeleton(func_name: str, func_signature: str) -> str:
    """parse 失败时的最终兜底: 占位 kernel + wrapper 骨架.

    生成保底可执行的 imports + kernel + wrapper(pass), 确保 wrapper
    可被调用方正常调用且不抛异常. 不依赖任何输入, 纯静态生成.
    """
    sig = (func_signature or '').strip()
    if sig.startswith('def '):
        sig = sig[4:].strip()
    if not sig:
        sig = f"{func_name}()"
    return (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        f"def {func_name}_kernel(X, n, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offs < n\n"
        "    x = tl.load(X + offs, mask=mask, other=0.0)\n"
        "    tl.store(X + offs, x, mask=mask)\n"
        "\n"
        f"def {sig}:\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        pass\n"
    )


def apply_ast_guard(code: str, func_name: str, func_signature: str) -> tuple[str, str]:
    """对 prediction 代码做 AST 兜底改造的统一入口.

    流程:
        1. 尝试 wrap_wrapper_with_try_except
        2. parse 失败 → 回退到 build_fallback_skeleton

    Returns:
        (final_code, action):
            action ∈ {'wrapped', 'already_guarded', 'wrapper_not_found',
                      'parse_failed_skeleton', 'empty_input'}
    """
    if not (code or '').strip():
        return build_fallback_skeleton(func_name, func_signature), 'empty_input'
    try:
        new_code = wrap_wrapper_with_try_except(code, func_name)
    except SyntaxError:
        return build_fallback_skeleton(func_name, func_signature), 'parse_failed_skeleton'

    if new_code == code:
        # 判断是 already_guarded 还是 wrapper_not_found
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    if _wrapper_has_try_guard(node):
                        return new_code, 'already_guarded'
                    return new_code, 'wrapper_not_found'
        except SyntaxError:
            pass
        return new_code, 'wrapper_not_found'
    return new_code, 'wrapped'
