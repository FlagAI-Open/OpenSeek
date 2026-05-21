from __future__ import annotations

import math
import re
from collections import Counter

from .data import Record

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def vectorize(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class LexicalRetriever:
    def __init__(self, examples: list[Record]) -> None:
        self.examples = examples
        self._vectors = [vectorize(example.text) for example in examples]

    def select(self, query: str, k: int) -> list[Record]:
        query_vector = vectorize(query)
        ranked = sorted(
            zip(self.examples, self._vectors, strict=True),
            key=lambda pair: cosine(query_vector, pair[1]),
            reverse=True,
        )
        return [example for example, _ in ranked[:k]]
