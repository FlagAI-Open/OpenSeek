import math
import re
from collections import Counter
from typing import Dict, List


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def build_query(instruction: str, label_space: List[str], sample_text: str) -> str:
    parts = [instruction, " ".join(label_space[:8]), sample_text[:400]]
    return "\n".join(part for part in parts if part)


def _score(query_tokens: List[str], chunk_text: str) -> float:
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    counts = Counter(chunk_tokens)
    overlap = sum(counts[token] for token in query_tokens if token in counts)
    norm = math.sqrt(len(chunk_tokens))
    return overlap / norm if norm else 0.0


def retrieve_top_chunks(
    chunks: List[Dict[str, int | str]], query: str, top_k: int = 8
) -> List[Dict[str, int | str]]:
    query_tokens = _tokenize(query)
    scored = []
    for chunk in chunks:
        score = _score(query_tokens, str(chunk["text"]))
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
