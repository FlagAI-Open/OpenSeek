import math
import os
import re
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from method_hyb_prompts import (
    TASK_PROMPT_BUILDERS,
    build_prompt,
    build_prompt____,
    build_prompt_backup,
    build_prompt_for_task,
    register_task_prompt,
)

""" Here is an example of implementation of Long-Context Data Annotation. """

# 混合检索默认：Qwen3 4B 向量模型 + 4B 重排模型（Hugging Face 官方仓库名）
DEFAULT_ICL_EMBEDDING_MODEL = "/data/llm_models/Qwen3/Qwen3-Embedding-8B"
DEFAULT_ICL_RERANK_MODEL = "/data/llm_models/Qwen3/Qwen3-Reranker-8B"

_ANNOTATE_DEBUG_PRINTED = 0
_ANNOTATE_DEBUG_LIMIT = int(os.environ.get("ANNOTATE_DEBUG_LIMIT", "5"))


def _debug_log(msg: str) -> None:
    """Rate-limited debug log for API failures and parse failures."""
    global _ANNOTATE_DEBUG_PRINTED
    if _ANNOTATE_DEBUG_PRINTED < _ANNOTATE_DEBUG_LIMIT:
        print(msg)
        _ANNOTATE_DEBUG_PRINTED += 1


def _normalize_output(output: Any) -> str:
    if isinstance(output, list) and output:
        return str(output[0])
    return str(output)


def _example_explanation(example: dict) -> str | None:
    """取单条示例的 explanation 文本；无则返回 None。"""
    exp = example.get("explanation")
    if exp is None:
        return None
    s = str(exp).strip()
    return s if s else None


def _word_jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa and not wb:
        return 0.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[str]:
        s = re.sub(r"\s+", " ", s.lower()).strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = ngrams(a), ngrams(b)
    if not ga and not gb:
        return 0.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _hybrid_similarity(
    text2annotate: str, task_description: str, example_input: str
) -> float:
    """
    混合检索分数：词级 Jaccard + 字符 n-gram Jaccard（对代码/长句更稳），
    再与任务描述和示例 input 的弱相关项加权融合。
    """
    q, t, x = text2annotate or "", task_description or "", example_input or ""
    lex = 0.5 * (_word_jaccard(q, x) + _char_ngram_jaccard(q, x))
    task_align = 0.25 * _word_jaccard(t, x) + 0.25 * _char_ngram_jaccard(t, x)
    return 0.75 * lex + 0.25 * task_align


def _icl_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


class _BM25Okapi:
    """Okapi BM25，词项为 ``_icl_tokenize`` 结果。"""

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = tokenized_corpus
        self.N = len(tokenized_corpus)
        self.doc_lens = [len(d) for d in tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0
        self.doc_freqs: list[Counter[str]] = [Counter(d) for d in tokenized_corpus]
        df: defaultdict[str, int] = defaultdict(int)
        for d in tokenized_corpus:
            for t in set(d):
                df[t] += 1
        self.df = dict(df)
        self.idf: dict[str, float] = {}
        for t, n_t in self.df.items():
            self.idf[t] = math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def scores(self, query_tokens: list[str]) -> list[float]:
        if not self.N:
            return []
        scores = [0.0] * self.N
        q_terms = set(query_tokens)
        for i in range(self.N):
            dl = self.doc_lens[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            tf_d = self.doc_freqs[i]
            s = 0.0
            for t in q_terms:
                if t not in self.idf:
                    continue
                tf = tf_d.get(t, 0)
                if tf == 0:
                    continue
                idf = self.idf[t]
                num = tf * (self.k1 + 1)
                den = tf + denom_norm
                s += idf * (num / den)
            scores[i] = s
        return scores


def _minmax_norm(xs: list[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


_EMBEDDER_STATE: tuple[Any, ...] | None = None
_CROSS_ENCODER_STATE: tuple[Any, ...] | None = None
_QWEN3_EMBED_STATE: tuple[Any, Any, str, str] | None = None
_ST_CROSS_ENCODER: tuple[Any, str] | None = None
_QWEN3_EMBED_LOAD_FAILED: set[str] = set()
_ST_CROSS_ENCODER_FAILED: set[str] = set()


def _icl_retrieval_device() -> str:
    d = os.environ.get("ICL_RETRIEVAL_DEVICE", "").strip()
    if d:
        return d
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _is_qwen3_embedding_model(model_name: str) -> bool:
    return "qwen3-embedding" in (model_name or "").lower()


def _is_qwen3_reranker_model(model_name: str) -> bool:
    n = (model_name or "").lower()
    return "qwen3-reranker" in n or "qwen3-rerank" in n


def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Qwen3-Embedding 官方推荐：取序列最后一个有效 token 的隐状态。"""
    import torch

    left_padding = bool(attention_mask[:, -1].sum().item() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def _get_qwen3_embedding_model(model_name: str) -> tuple[Any, Any, str] | None:
    """(tokenizer, model, device)；与通用 MiniLM 路径分缓存。"""
    global _QWEN3_EMBED_STATE
    if _QWEN3_EMBED_STATE is not None and _QWEN3_EMBED_STATE[2] == model_name:
        return _QWEN3_EMBED_STATE[0], _QWEN3_EMBED_STATE[1], _QWEN3_EMBED_STATE[3]
    if model_name in _QWEN3_EMBED_LOAD_FAILED:
        return None
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None
    device = _icl_retrieval_device()
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        extra: dict[str, Any] = {}
        dt = os.environ.get("ICL_QWEN_EMBED_DTYPE", "").strip().lower()
        if dt in {"float16", "fp16", "half"}:
            extra["torch_dtype"] = torch.float16
        elif dt in {"bfloat16", "bf16"}:
            extra["torch_dtype"] = torch.bfloat16
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **extra)
        model.eval()
        model.to(device)
    except Exception as e:
        _debug_log(f"[_get_qwen3_embedding_model] 加载失败 {model_name}: {e}")
        _QWEN3_EMBED_LOAD_FAILED.add(model_name)
        return None
    _QWEN3_EMBED_STATE = (tok, model, model_name, device)
    return tok, model, device


def _semantic_cosine_scores_qwen3_embedding(
    doc_texts: list[str], query_text: str, model_name: str
) -> list[float] | None:
    import torch
    import torch.nn.functional as F

    loaded = _get_qwen3_embedding_model(model_name)
    if loaded is None:
        return None
    tok, model, device = loaded
    max_len = int(os.environ.get("ICL_QWEN_EMBED_MAX_LENGTH", "8192"))
    instruct = os.environ.get(
        "ICL_QWEN_EMBED_INSTRUCT",
        "Given the annotation task and the text to annotate, retrieve similar labeled example inputs for in-context learning.",
    ).strip()
    q_formatted = f"Instruct: {instruct}\nQuery: {query_text}"
    batch = max(1, int(os.environ.get("ICL_QWEN_EMBED_BATCH", "4")))
    try:
        sims: list[float] = []
        with torch.no_grad():
            enc_q = tok(
                [q_formatted],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc_q = {k: v.to(device) for k, v in enc_q.items()}
            out_q = model(**enc_q)
            q_emb = _last_token_pool(out_q.last_hidden_state, enc_q["attention_mask"])
            q_emb = F.normalize(q_emb, p=2, dim=1).cpu()
            for i in range(0, len(doc_texts), batch):
                chunk = doc_texts[i : i + batch]
                enc_d = tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                enc_d = {k: v.to(device) for k, v in enc_d.items()}
                out_d = model(**enc_d)
                d_emb = _last_token_pool(out_d.last_hidden_state, enc_d["attention_mask"])
                d_emb = F.normalize(d_emb, p=2, dim=1).cpu()
                sims.extend(float(x) for x in (d_emb @ q_emb.T).squeeze(-1).tolist())
        return sims
    except Exception as e:
        _debug_log(f"[_semantic_cosine_scores_qwen3_embedding] 编码失败: {e}")
        return None


def _get_st_cross_encoder(model_name: str) -> Any | None:
    """Qwen3-Reranker 推荐用 sentence_transformers.CrossEncoder。"""
    global _ST_CROSS_ENCODER
    if _ST_CROSS_ENCODER is not None and _ST_CROSS_ENCODER[1] == model_name:
        return _ST_CROSS_ENCODER[0]
    if model_name in _ST_CROSS_ENCODER_FAILED:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        _debug_log(
            "[_get_st_cross_encoder] 未安装 sentence-transformers，无法加载 Qwen3-Reranker；"
            "可执行: pip install \"sentence-transformers>=2.7.0\""
        )
        return None
    instr = os.environ.get(
        "ICL_QWEN_RERANK_INSTRUCT",
        "Given an annotation task and a text to annotate, judge whether the document is a useful similar demonstration for the same labeling task.",
    ).strip()
    try:
        ce = CrossEncoder(
            model_name,
            trust_remote_code=True,
            prompts={"icl": instr},
            default_prompt_name="icl",
        )
    except TypeError:
        try:
            ce = CrossEncoder(model_name, trust_remote_code=True)
        except Exception as e:
            _debug_log(f"[_get_st_cross_encoder] CrossEncoder 加载失败: {e}")
            _ST_CROSS_ENCODER_FAILED.add(model_name)
            return None
    except Exception as e:
        _debug_log(f"[_get_st_cross_encoder] CrossEncoder 加载失败: {e}")
        _ST_CROSS_ENCODER_FAILED.add(model_name)
        return None
    # Qwen3-Reranker 在部分环境下 tokenizer 未定义 pad_token；
    # CrossEncoder.predict(batch_size>1) 会直接报错，因此这里做兼容修复。
    try:
        tok = ce.tokenizer
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif tok.unk_token is not None:
                tok.pad_token = tok.unk_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
                if hasattr(ce, "model") and ce.model is not None:
                    ce.model.resize_token_embeddings(len(tok))
        if tok.pad_token_id is not None and hasattr(ce, "model") and ce.model is not None:
            if getattr(ce.model.config, "pad_token_id", None) is None:
                ce.model.config.pad_token_id = tok.pad_token_id
    except Exception as e:
        _debug_log(f"[_get_st_cross_encoder] pad_token 兼容设置失败: {e}")
    _ST_CROSS_ENCODER = (ce, model_name)
    return ce


def _cross_encoder_scores_qwen_reranker(
    query: str,
    doc_texts: list[str],
    model_name: str,
    batch_size: int | None = None,
) -> list[float] | None:
    ce = _get_st_cross_encoder(model_name)
    if ce is None:
        return None
    bs = batch_size if batch_size is not None else int(os.environ.get("ICL_QWEN_RERANK_BATCH", "4"))
    pairs = [(query, d) for d in doc_texts]
    try:
        try:
            raw = ce.predict(pairs, batch_size=bs, show_progress_bar=False, prompt_name="icl")
        except TypeError:
            raw = ce.predict(pairs, batch_size=bs, show_progress_bar=False)
    except Exception as e:
        msg = str(e)
        if "no padding token is defined" in msg.lower() and bs > 1:
            _debug_log(
                "[_cross_encoder_scores_qwen_reranker] 检测到无 pad_token，"
                "自动降级 batch_size=1 重试。"
            )
            try:
                try:
                    raw = ce.predict(
                        pairs, batch_size=1, show_progress_bar=False, prompt_name="icl"
                    )
                except TypeError:
                    raw = ce.predict(pairs, batch_size=1, show_progress_bar=False)
            except Exception as e2:
                _debug_log(f"[_cross_encoder_scores_qwen_reranker] 重试仍失败: {e2}")
                return None
        else:
            _debug_log(f"[_cross_encoder_scores_qwen_reranker] predict 失败: {e}")
            return None
    try:
        import numpy as np

        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        return [float(x) for x in arr.tolist()]
    except Exception:
        flat = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return [float(x) for x in flat]


def _get_sentence_embedding_model(
    model_name: str,
) -> tuple[Any, Any, str] | None:
    """(tokenizer, model, device) 或加载失败返回 None。"""
    global _EMBEDDER_STATE
    if _EMBEDDER_STATE is not None and _EMBEDDER_STATE[2] == model_name:
        return _EMBEDDER_STATE[0], _EMBEDDER_STATE[1], _EMBEDDER_STATE[3]
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None
    device = _icl_retrieval_device()
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        model.to(device)
    except Exception as e:
        _debug_log(f"[_get_sentence_embedding_model] 加载失败 {model_name}: {e}")
        return None
    _EMBEDDER_STATE = (tok, model, model_name, device)
    return tok, model, device


def _mean_pool(last_hidden: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def _embed_texts_minilm(
    tok: Any, model: Any, device: str, texts: list[str], batch_size: int = 16
) -> Any:
    import torch

    all_embs: list[Any] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def _semantic_cosine_scores_minilm(
    doc_texts: list[str], query_text: str, model_name: str
) -> list[float] | None:
    """通用句向量：mean pool + 余弦（如 all-MiniLM-L6-v2）。"""
    loaded = _get_sentence_embedding_model(model_name)
    if loaded is None:
        return None
    tok, model, device = loaded
    try:
        q_emb = _embed_texts_minilm(tok, model, device, [query_text], batch_size=1)
        d_emb = _embed_texts_minilm(tok, model, device, doc_texts, batch_size=16)
        sims = (d_emb @ q_emb.T).squeeze(-1).tolist()
        return [float(x) for x in sims]
    except Exception as e:
        _debug_log(f"[_semantic_cosine_scores_minilm] 编码失败: {e}")
        return None


def _semantic_cosine_scores(
    doc_texts: list[str], query_text: str, model_name: str
) -> list[float] | None:
    """query 与每条 doc 的余弦相似度；Qwen3-Embedding 走 last_token_pool 与 Instruct/Query 格式。"""
    if _is_qwen3_embedding_model(model_name):
        return _semantic_cosine_scores_qwen3_embedding(doc_texts, query_text, model_name)
    return _semantic_cosine_scores_minilm(doc_texts, query_text, model_name)


def _get_cross_encoder(model_name: str) -> tuple[Any, Any, str] | None:
    global _CROSS_ENCODER_STATE
    if _CROSS_ENCODER_STATE is not None and _CROSS_ENCODER_STATE[2] == model_name:
        return _CROSS_ENCODER_STATE[0], _CROSS_ENCODER_STATE[1], _CROSS_ENCODER_STATE[3]
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    device = _icl_retrieval_device()
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        model.eval()
        model.to(device)
    except Exception as e:
        _debug_log(f"[_get_cross_encoder] 加载失败 {model_name}: {e}")
        return None
    _CROSS_ENCODER_STATE = (tok, model, model_name, device)
    return tok, model, device


def _cross_encoder_scores_sequence_classification(
    query: str,
    doc_texts: list[str],
    model_name: str,
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float] | None:
    """传统 CrossEncoder（如 ms-marco-MiniLM）：双句对分类 logits。"""
    loaded = _get_cross_encoder(model_name)
    if loaded is None:
        return None
    tok, model, device = loaded
    try:
        import torch

        scores: list[float] = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(doc_texts), batch_size):
                batch_docs = doc_texts[i : i + batch_size]
                enc = tok(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.shape[-1] == 2:
                    batch_scores = torch.nn.functional.log_softmax(logits, dim=-1)[:, 1]
                else:
                    batch_scores = logits[:, -1]
                scores.extend(batch_scores.float().cpu().tolist())
        return [float(x) for x in scores]
    except Exception as e:
        _debug_log(f"[_cross_encoder_scores_sequence_classification] 打分失败: {e}")
        return None


def _cross_encoder_scores(
    query: str,
    doc_texts: list[str],
    model_name: str,
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float] | None:
    """(query, doc) 重排分；Qwen3-Reranker 用 sentence_transformers.CrossEncoder。"""
    if _is_qwen3_reranker_model(model_name):
        return _cross_encoder_scores_qwen_reranker(query, doc_texts, model_name)
    return _cross_encoder_scores_sequence_classification(
        query, doc_texts, model_name, batch_size=batch_size, max_length=max_length
    )


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _format_icl_block(
    input_text: str,
    output_text: str,
    explanation: str | None,
) -> str:
    """单条 ICL：可选 explanation + 答案（<label>）。"""
    if explanation:
        return (
            f"# {input_text}\n"
            f"Explanation: {explanation}\n"
            f"<label> {output_text} </label>\n"
        )
    return f"# {input_text} <label> {output_text} </label>\n"


def _shrink_explanations_to_budget(
    tokenizer,
    parts: list[tuple[str, str, str | None]],
    budget: int,
) -> str:
    """
    parts: (input_text, output_text, explanation_or_none)
    在总 token 不超过 budget 的前提下拼接；必要时从最长 explanation 开始截断字符。
    """
    explanations = [list(p) for p in parts]  # mutable copies

    def build_string(rows: list[list[Any]]) -> str:
        return "".join(
            _format_icl_block(str(r[0]), str(r[1]), r[2] if r[2] else None)
            for r in rows
        )

    while True:
        s = build_string(explanations)
        n = _token_len(tokenizer, s)
        if n <= budget:
            return s
        # 找仍带 explanation 的最长一条，截断其 explanation
        best_i, best_len = -1, -1
        for i, r in enumerate(explanations):
            if r[2] and len(str(r[2])) > best_len:
                best_len = len(str(r[2]))
                best_i = i
        if best_i < 0:
            # 无法再截断：硬截断整块字符串（极少触发）
            enc = tokenizer.encode(s, add_special_tokens=False)[:budget]
            return tokenizer.decode(enc, skip_special_tokens=True)
        exp = str(explanations[best_i][2])
        if len(exp) <= 80:
            explanations[best_i][2] = None  # 去掉 explanation 再试
        else:
            explanations[best_i][2] = exp[: max(1, len(exp) * 3 // 4)]


def _icl_fusion_weights() -> tuple[float, float]:
    w_b = float(os.environ.get("ICL_FUSE_W_BM25", "0.45"))
    w_s = float(os.environ.get("ICL_FUSE_W_SEM", "0.55"))
    s = w_b + w_s
    if s < 1e-9:
        return 0.5, 0.5
    return w_b / s, w_s / s


def _force_disable_emb_rerank() -> bool:
    """
    全局强制开关：禁用 embedding / rerank 模型依赖。
    开启后，ICL 检索仅走 BM25/关键词或词法回退路径。
    """
    return os.environ.get("ICL_FORCE_LEXICAL_ONLY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _keyword_overlap_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """
    关键词重合分：query 关键词在 doc 中的覆盖率（兼顾短 query 的稳定性）。
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    q = set(query_tokens)
    d = set(doc_tokens)
    if not q:
        return 0.0
    return len(q & d) / len(q)


def _retrieve_doc_indices_bm25_keyword(
    doc_texts: list[str],
    query_text: str,
    top_k: int,
) -> list[int]:
    """
    BM25 + 关键词覆盖率融合检索（不依赖向量模型和 reranker）。
    """
    n_docs = len(doc_texts)
    if n_docs == 0:
        return []
    tk = max(1, top_k)

    tokenized = [_icl_tokenize(t) for t in doc_texts]
    q_tokens = _icl_tokenize(query_text)
    bm25 = _BM25Okapi(tokenized)
    raw_bm = bm25.scores(q_tokens)
    norm_bm = _minmax_norm(raw_bm)
    kw_scores = [_keyword_overlap_score(q_tokens, d) for d in tokenized]
    norm_kw = _minmax_norm(kw_scores)

    w_b = float(os.environ.get("ICL_FUSE_W_BM25", "0.7"))
    w_k = float(os.environ.get("ICL_FUSE_W_KEYWORD", "0.3"))
    s = w_b + w_k
    if s < 1e-9:
        w_b, w_k = 0.7, 0.3
    else:
        w_b, w_k = w_b / s, w_k / s

    fused = [w_b * a + w_k * b for a, b in zip(norm_bm, norm_kw, strict=True)]
    return sorted(range(n_docs), key=lambda i: (-fused[i], i))[:tk]


def _print_rerank_results(
    query_text: str,
    doc_texts: list[str],
    first_stage: list[int],
    final_indices: list[int],
    fused_scores: list[float],
    ce_scores: list[float] | None = None,
) -> None:
    """按需打印 rerank 过程与最终结果（默认关闭）。"""
    enabled = os.environ.get("ICL_PRINT_RERANK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return

    preview_len = int(os.environ.get("ICL_PRINT_RERANK_PREVIEW", "140"))
    query_preview = query_text[:preview_len].replace("\n", " ")
    print(
        f"[RERANK] query_preview={query_preview!r} "
        f"candidates={len(first_stage)} final_topk={len(final_indices)}"
    )
    for rank, doc_idx in enumerate(final_indices, start=1):
        fused = fused_scores[doc_idx] if 0 <= doc_idx < len(fused_scores) else None
        rerank_score = None
        if ce_scores is not None and doc_idx in first_stage:
            rerank_score = ce_scores[first_stage.index(doc_idx)]
        doc_preview = doc_texts[doc_idx][:preview_len].replace("\n", " ")
        print(
            f"[RERANK][TOP{rank}] idx={doc_idx} fused={fused:.6f} "
            f"rerank={rerank_score if rerank_score is not None else 'NA'} "
            f"doc_preview={doc_preview!r}"
        )


def _retrieve_doc_indices_bm25_semantic_rerank(
    doc_texts: list[str],
    query_text: str,
    top_k: int,
    rerank_pool_size: int,
    embedding_model: str,
    rerank_model: str,
) -> list[int]:
    """
    第一阶段：BM25 + 句向量余弦（min-max 后加权融合）→ 取前 ``rerank_pool_size``；
    第二阶段：Cross-Encoder 对 (query, doc) 重打分 → 取 ``top_k`` 文档下标。
    任一步失败时安全降级（仅 BM25 融合 / 跳过 rerank）。
    """
    n_docs = len(doc_texts)
    if n_docs == 0:
        return []
    tk = max(1, top_k)
    rpool = max(tk, min(rerank_pool_size, n_docs))

    tokenized = [_icl_tokenize(t) for t in doc_texts]
    q_tokens = _icl_tokenize(query_text)
    bm25 = _BM25Okapi(tokenized)
    raw_bm = bm25.scores(q_tokens)
    norm_bm = _minmax_norm(raw_bm)

    disable_sem = os.environ.get("ICL_DISABLE_SEMANTIC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    fused = list(norm_bm)
    if not disable_sem:
        sem = _semantic_cosine_scores(doc_texts, query_text, embedding_model)
        if sem is not None:
            norm_sem = _minmax_norm(sem)
            w_b, w_s = _icl_fusion_weights()
            fused = [w_b * a + w_s * b for a, b in zip(norm_bm, norm_sem, strict=True)]

    first_stage = sorted(range(n_docs), key=lambda i: (-fused[i], i))[:rpool]

    disable_rr = os.environ.get("ICL_DISABLE_RERANK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if disable_rr:
        final = first_stage[:tk]
        _print_rerank_results(query_text, doc_texts, first_stage, final, fused_scores=fused)
        return final

    max_doc = int(os.environ.get("ICL_RERANK_DOC_CHARS", "4000"))
    pool_docs = [doc_texts[i][:max_doc] for i in first_stage]
    ce_raw = _cross_encoder_scores(query_text, pool_docs, rerank_model)
    if ce_raw is None or len(ce_raw) != len(first_stage):
        final = first_stage[:tk]
        _print_rerank_results(query_text, doc_texts, first_stage, final, fused_scores=fused)
        return final

    order_in_pool = sorted(range(len(first_stage)), key=lambda j: (-ce_raw[j], j))
    final = [first_stage[j] for j in order_in_pool[:tk]]
    _print_rerank_results(
        query_text,
        doc_texts,
        first_stage,
        final,
        fused_scores=fused,
        ce_scores=ce_raw,
    )
    return final


def _retrieve_doc_indices_lexical_fallback(
    doc_texts: list[str],
    text2annotate: str,
    task_description: str,
    top_k: int,
) -> list[int]:
    """无 torch / 模型不可用时回退：原 Jaccard + ngram 混合分。"""
    scored = [
        (_hybrid_similarity(text2annotate, task_description, d), i)
        for i, d in enumerate(doc_texts)
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    k = max(1, top_k)
    return [i for _, i in scored[:k]]


def select_examples_hybrid(
    all_examples: list[dict],
    task_description: str,
    text2annotate: str,
    tokenizer_path: str | None = None,
    top_k: int = 3,
    target_length: int | None = None,
    use_explanation: bool = True,
    *,
    rerank_pool_size: int | None = None,
    embedding_model: str | None = None,
    rerank_model: str | None = None,
    use_bm25_semantic_rerank: bool = True,
    use_bm25_keyword_only: bool = False,
    exclude_example_id: str | None = None,
    retrieval_query_override: str | None = None,
) -> str:
    """
    混合检索 ICL（默认）：BM25 + **Qwen3-Embedding-4B** 语义向量 + **Qwen3-Reranker-4B** 重排后取 top_k；
    每条可带 ``explanation`` + ``<label>`` 答案；总长度用 Qwen tokenizer 压到预算内。

    环境变量（可选）：``ICL_EMBEDDING_MODEL``（默认 ``Qwen/Qwen3-Embedding-4B``）、
    ``ICL_RERANK_MODEL``（默认 ``Qwen/Qwen3-Reranker-4B``）、``ICL_RERANK_POOL``、
    ``ICL_FUSE_W_BM25`` / ``ICL_FUSE_W_SEM``、``ICL_DISABLE_SEMANTIC``、``ICL_DISABLE_RERANK``、
    ``ICL_RETRIEVAL_DEVICE``、``ICL_RERANK_DOC_CHARS``、
    ``ICL_QWEN_EMBED_INSTRUCT`` / ``ICL_QWEN_EMBED_MAX_LENGTH`` / ``ICL_QWEN_EMBED_BATCH``、
    ``ICL_QWEN_RERANK_INSTRUCT`` / ``ICL_QWEN_RERANK_BATCH``。
    重排依赖 ``sentence-transformers>=2.7.0`` 与 ``transformers>=4.51.0``（见 Qwen3 Embedding README）。

    ``use_bm25_keyword_only=True`` 时仅走 BM25+关键词融合（不加载向量模型/重排）。
    ``use_bm25_semantic_rerank=False`` 且 ``use_bm25_keyword_only=False`` 时使用原 Jaccard 混合分。

    ``retrieval_query_override``:
        若提供非空字符串，则 BM25/向量/重排的 query 使用该文本，而 **不** 使用
        ``{task_description}\\n{text2annotate}``；Jaccard 回退时 query 侧与 ``text2annotate`` 对齐项
        改为该字符串（仍保留 ``task_description`` 与文档的弱相关项）。用于 Jeopardy 等需削弱「整段题干
        复制」噪声的场景。
    """
    if tokenizer_path:
        tok = tokenizer_path
    else:
        _local = Path(__file__).resolve().parent.parent / "Qwen3-4B"
        tok = str(_local) if _local.is_dir() else "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(tok, trust_remote_code=True)
    budget = target_length if target_length is not None else 8192

    records: list[tuple[int, dict, str]] = []
    for idx, ex in enumerate(all_examples):
        if exclude_example_id is not None and str(ex.get("id", "")).strip() == str(exclude_example_id).strip():
            continue
        try:
            inp = ex["input"]
        except KeyError:
            continue
        records.append((idx, ex, inp))

    if not records:
        return ""

    doc_texts = [r[2] for r in records]
    query_text = (
        retrieval_query_override.strip()
        if (retrieval_query_override is not None and str(retrieval_query_override).strip())
        else f"{task_description}\n{text2annotate}".strip()
    )
    lexical_query = (
        retrieval_query_override.strip()
        if (retrieval_query_override is not None and str(retrieval_query_override).strip())
        else (text2annotate or "")
    )
    k = max(1, top_k)
    rpool = rerank_pool_size
    if rpool is None:
        rpool = int(os.environ.get("ICL_RERANK_POOL", "24"))
    rpool = max(k, rpool)

    emb_name = embedding_model or os.environ.get(
        "ICL_EMBEDDING_MODEL", DEFAULT_ICL_EMBEDDING_MODEL
    )
    rr_name = rerank_model or os.environ.get("ICL_RERANK_MODEL", DEFAULT_ICL_RERANK_MODEL)

    force_lexical_only = _force_disable_emb_rerank()
    if force_lexical_only:
        # 环境限制下不允许加载 emb/rerank 模型：统一退化为 BM25+关键词融合。
        use_bm25_keyword_only = True
        use_bm25_semantic_rerank = False

    if use_bm25_keyword_only:
        doc_indices = _retrieve_doc_indices_bm25_keyword(doc_texts, query_text, k)
    elif use_bm25_semantic_rerank:
        try:
            import torch  # noqa: F401
        except ImportError:
            doc_indices = _retrieve_doc_indices_lexical_fallback(
                doc_texts, lexical_query, task_description, k
            )
        else:
            try:
                doc_indices = _retrieve_doc_indices_bm25_semantic_rerank(
                    doc_texts,
                    query_text,
                    k,
                    rpool,
                    emb_name,
                    rr_name,
                )
            except Exception as e:
                _debug_log(f"[select_examples_hybrid] BM25/语义/rerank 失败，回退 Jaccard: {e}")
                doc_indices = _retrieve_doc_indices_lexical_fallback(
                    doc_texts, lexical_query, task_description, k
                )
            if not doc_indices:
                doc_indices = _retrieve_doc_indices_lexical_fallback(
                    doc_texts, lexical_query, task_description, k
                )
    else:
        doc_indices = _retrieve_doc_indices_lexical_fallback(
            doc_texts, lexical_query, task_description, k
        )

    parts: list[tuple[str, str, str | None]] = []
    for j in doc_indices:
        if j < 0 or j >= len(records):
            continue
        ex = records[j][1]
        try:
            input_text = ex["input"]
            output_text = _normalize_output(ex["output"])
        except KeyError:
            continue
        exp = _example_explanation(ex) if use_explanation else None
        parts.append((input_text, output_text, exp))

    if not parts:
        return ""

    return _shrink_explanations_to_budget(tokenizer, parts, budget)


def select_examples_backup(all_examples:list[dict], task_description:str, text2annotate:str)->str:
    """
        Select examples from all_examples to fit into the target context length.
        all_examples:
            A list of examples, where each example is a dict with keys 'input', 'output', and 'length'.
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review", "length": 79``},
        task_description:
            The description of the annotation task which may be used for example evaluation. 
            For example, ``Given an English language product review, 
            determine if it is a Good Review or a Bad Review.`` 
        text2annotate:
            The text that needs to be annotated  which may be used for example retrieval.
            For example, ``My son received this book as a gift. I was extremely disappointed.``
        
    """
    # Notice that the maximum context length is restricted.
    target_length = 10_000
    
    input_list = [example['input'] for example in all_examples]
    output_list = [example['output'][0] for example in all_examples]
    length_list = [example['length'] for example in all_examples]
    
    # <label> have 2 tokens; </label> have 3 tokens; \n have 1 token; # have 1 token.
    examples_str, token_num = "", 0
    for i, (input_text, output_text, length) in enumerate(zip(input_list, output_list, length_list)):
        if length + token_num <= target_length:
            token_num += (length + 2 + 3 + 1 + 1)
            example_str = f"# {input_text} <label> {output_text} </label>\n"
            examples_str += example_str
        else:
            return examples_str, i
    return examples_str

def select_examples(
    all_examples: list[dict],
    task_description: str,
    text2annotate: str,
    tokenizer_path: str | None = None,
    *,
    hybrid: bool = True,
    top_k: int = 3,
    target_length: int | None = None,
    use_explanation: bool = True,
    rerank_pool_size: int | None = None,
    embedding_model: str | None = None,
    rerank_model: str | None = None,
    use_bm25_semantic_rerank: bool = True,
    retrieval_query_override: str | None = None,
) -> str:
    """
    从 ``all_examples`` 中选取 ICL 片段（默认：BM25+语义+Cross-Encoder 重排 top_k + explanation + tokenizer 控长）。

    all_examples:
        每条为 dict，至少含 ``input``、``output``；若有 ``explanation`` 则与答案一并写入提示。
    hybrid:
        ``True``（默认）：``select_examples_hybrid``。
        ``False``：沿用原按列表顺序累加直至 token 上限的策略。
    rerank_pool_size / embedding_model / rerank_model / use_bm25_semantic_rerank:
        仅在 ``hybrid=True`` 时传给 ``select_examples_hybrid``；也可用环境变量覆盖默认模型与池大小。
    """
    if hybrid:
        return select_examples_hybrid(
            all_examples,
            task_description,
            text2annotate,
            tokenizer_path=tokenizer_path,
            top_k=top_k,
            target_length=target_length,
            use_explanation=use_explanation,
            rerank_pool_size=rerank_pool_size,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            use_bm25_semantic_rerank=use_bm25_semantic_rerank,
            retrieval_query_override=retrieval_query_override,
        )

    if tokenizer_path:
        tok = tokenizer_path
    else:
        _local = Path(__file__).resolve().parent.parent / "Qwen3-4B"
        tok = str(_local) if _local.is_dir() else "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(tok, trust_remote_code=True)
    tl = target_length if target_length is not None else 8192

    examples_str, token_num = "", 0
    for i, example in enumerate(all_examples):
        try:
            input_text = example["input"]
            output_text = _normalize_output(example["output"])
            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens
            if length + token_num <= tl:
                token_num += length + 2 + 3 + 1 + 1
                example_str = f"# {input_text} <label> {output_text} </label>\n"
                examples_str += example_str
            else:
                return examples_str
        except KeyError as e:
            print(f"警告：示例{i}缺少键{e}，跳过该示例")
            continue
    return examples_str




def count_answer(text: str) -> tuple[list, dict]:
    """
    提取字符串中<label>标签内的所有内容（字符串形式），统计出现次数最多的内容
    :param text: 包含<label>标签的原始字符串
    :return: 出现次数最多的内容列表、所有内容的频次统计字典
    """
    text = str(text or "")
    # 轻量归一化：兼容全角中括号与大小写 label 写法
    normalized = (
        text.replace("［", "[").replace("］", "]").replace("【", "[").replace("】", "]")
    )
    normalized = re.sub(r"\[\s*label\s*\]", "[label]", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\[\s*/\s*label\s*\]", "[/label]", normalized, flags=re.IGNORECASE)

    content_matches: list[str] = []
    # 主模式：标准 XML 标签
    content_matches.extend(re.findall(r"<label>\s*(.+?)\s*</label>", normalized, re.DOTALL))
    # 兼容模式：部分模型会输出 [label] ... [/label]
    if not content_matches:
        content_matches.extend(
            re.findall(r"\[label\]\s*(.+?)\s*\[/label\]", normalized, re.DOTALL)
        )
    # 回退模式：只出现起始标签，无闭合标签时，取起始标签到行尾
    if not content_matches:
        m = re.search(r"<label>\s*([^\n\r]+)", normalized, flags=re.IGNORECASE)
        if m:
            content_matches.append(m.group(1))
    if not content_matches:
        m = re.search(r"\[label\]\s*([^\n\r]+)", normalized)
        if m:
            content_matches.append(m.group(1))
    # 针对 [label][1,2,3] 这类常见格式，优先提取紧随其后的首个列表字面量
    if not content_matches:
        m = re.search(r"\[label\]\s*(\[[^\[\]\n\r]*\])", normalized)
        if m:
            content_matches.append(m.group(1))
    # 最后回退：无标签但答案本身是完整列表（task3 常见），仅取首个列表字面量
    if not content_matches:
        m = re.search(r"(\[[^\[\]\n\r]*\])", normalized)
        if m:
            content_matches.append(m.group(1))
    
    content_counter = Counter(content_matches)
    if not content_counter:
        # 可选兜底：当模型未输出 <label> 时，从纯文本中提取一行短答案。
        # 默认关闭，避免影响其它任务；由调用方通过环境变量显式开启。
        plain_fallback = os.environ.get("ANNOTATE_FALLBACK_PLAIN", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if plain_fallback:
            candidate = normalized.strip()
            candidate = re.sub(r"```.*?```", " ", candidate, flags=re.DOTALL)
            lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
            if lines:
                candidate = lines[-1]
            candidate = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"^[-*>\s]+", "", candidate).strip()
            if candidate:
                max_plain_chars = int(os.environ.get("ANNOTATE_MAX_PLAIN_CHARS", "120"))
                if 0 < len(candidate) <= max_plain_chars:
                    return candidate
        # task2 等数字任务兜底：无标签时从文本提取首个非负整数。
        number_fallback = os.environ.get("ANNOTATE_FALLBACK_NUMBER", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if number_fallback:
            m = re.search(r"\b(\d+)\b", normalized)
            if m:
                return m.group(1)
    if not content_counter:
        return None
    
    max_count = max(content_counter.values())
    answer = [content for content, count in content_counter.items() if count == max_count]
    
    final_answer = answer[0].strip()

    # 兼容 task8 等代码生成任务：默认不限制标签内容长度。
    # 若需要防异常超长输出，可设置环境变量 ANNOTATE_MAX_LABEL_CHARS（>0 生效）。
    max_label_chars = int(os.environ.get("ANNOTATE_MAX_LABEL_CHARS", "0"))
    if max_label_chars > 0 and len(final_answer) > max_label_chars:
        return None
    return final_answer


@contextmanager
def _global_api_call_lock():
    """
    全局请求锁（跨进程）：默认开启，避免并发请求导致结果相互影响。
    可通过 ANNOTATE_GLOBAL_API_LOCK=0 关闭。
    """
    lock_enabled = os.environ.get("ANNOTATE_GLOBAL_API_LOCK", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not lock_enabled:
        yield
        return

    default_lock_file = Path(__file__).resolve().parent.parent / ".annotate_api.lock"
    lock_file = Path(os.environ.get("ANNOTATE_LOCK_FILE", str(default_lock_file)))
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_file, "a+b") as fp:
        if fp.tell() == 0:
            fp.write(b"0")
            fp.flush()
        fp.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                fp.seek(0)
                msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def _nvidia_dashscope_chat_text(input_prompt: str) -> str | None:
    """
    调用本地 FlagOS（OpenAI 兼容）服务，返回助手完整文本；失败返回 None。
    默认连接 ``http://localhost:9010/v1``；与 ``annotate_nvidia`` 共享 thinking / 非流式降级逻辑。

    环境变量：
        ``FLAGSCALE_BASE_URL``（默认 ``http://localhost:9010/v1``）
        ``FLAGSCALE_API_KEY``（默认 ``EMPTY``）
        ``FLAGSCALE_MODEL``（默认 ``Qwen3-4B-ascend-flagos``）
    """
    from openai import OpenAI

    api_key = os.environ.get("FLAGSCALE_API_KEY", "EMPTY")
    base_url = os.environ.get("FLAGSCALE_BASE_URL", "http://localhost:9010/v1/").rstrip("/")
    model_id = os.environ.get("FLAGSCALE_MODEL", "Qwen3-4B-ascend-flagos")
    log_every_response = os.environ.get("ANNOTATE_LOG_EVERY_RESPONSE", "0") == "1"
    enable_thinking_raw = os.environ.get("DASHSCOPE_ENABLE_THINKING", "1").strip().lower()
    enable_thinking = enable_thinking_raw in {"1", "true", "yes", "on"}
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    try:
        with _global_api_call_lock():
            try:
                if enable_thinking:
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": input_prompt}],
                        max_tokens=10_000,
                        extra_body={"enable_thinking": True},
                        stream=True,
                    )
                    content_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    for chunk in completion:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            content_parts.append(content)
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_parts.append(reasoning)
                    whole_result = "".join(content_parts)
                    whole_reasoning = "".join(reasoning_parts)
                else:
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": input_prompt}],
                        max_tokens=10_000,
                        extra_body={"enable_thinking": False},
                        stream=False,
                    )
                    whole_result = completion.choices[0].message.content or ""
                    whole_reasoning = ""
            except Exception as e:
                err_text = str(e)
                if (
                    "parameter.enable_thinking must be set to false for non-streaming calls"
                    not in err_text
                ):
                    raise
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": input_prompt}],
                    max_tokens=10_000,
                    extra_body={"enable_thinking": False},
                    stream=False,
                )
                whole_result = completion.choices[0].message.content or ""
                whole_reasoning = ""
            if log_every_response:
                answer_preview = whole_result[:300].replace("\n", "\\n")
                print(f"[接口返回] model={model_id} thinking={enable_thinking}")
                if enable_thinking:
                    thinking_preview = whole_reasoning[:300].replace("\n", "\\n")
                    print(f"[thinking预览] {thinking_preview}")
                print(f"[answer预览] {answer_preview}")
    except Exception as e:
        _debug_log(f"[_nvidia_dashscope_chat_text] 请求失败: model={model_id}, error={e}")
        return None
    return whole_result


def annotate_nvidia_raw_text(input_prompt: str) -> str | None:
    """
    同 ``annotate_nvidia`` 的模型调用，返回**完整助手文本**（不经 ``count_answer`` 解析）。
    供 verifier / 结构化 JSON 等对输出格式有特殊需求的调用方。
    """
    text = _nvidia_dashscope_chat_text(input_prompt)
    return None if text is None else text


def annotate_nvidia(input_prompt:str)->list[str]:
    """
        使用本地 FlagOS OpenAI 兼容接口标注。
        需先启动本地服务（默认 ``http://localhost:9010/v1``，模型 ``Qwen3-4B-ascend-flagos``）。

        默认 ``DASHSCOPE_ENABLE_THINKING=1``：使用 ``stream=true`` 且 ``enable_thinking=true``
        （部分服务端要求二者同时出现；非流式请求不得带 ``enable_thinking``，否则会 400）。
        设 ``DASHSCOPE_ENABLE_THINKING=0``：非流式请求，``extra_body`` 中 ``enable_thinking=false``。
    """
    whole_result = _nvidia_dashscope_chat_text(input_prompt)
    if whole_result is None:
        return None

    prediction = count_answer(whole_result)
    if prediction is None:
        preview = whole_result[:200].replace("\n", "\\n")
        model_id = os.environ.get("FLAGSCALE_MODEL", "Qwen3-4B-ascend-flagos")
        _debug_log(f"[annotate_nvidia] prediction=None: model={model_id}, preview={preview}")
    return prediction

def annotate_ascend(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (Huawei Ascend).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=10_000,
        stream=False,
    )
    whole_result = response.choices[0].message.content
    prediction = count_answer(whole_result)
    return prediction
