from __future__ import annotations


def split_chunks(text: str, chunk_chars: int) -> list[str]:
    if chunk_chars <= 0:
        return [text]
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if len(current) + len(paragraph) + 1 <= chunk_chars:
            current = f"{current}\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            while len(paragraph) > chunk_chars:
                chunks.append(paragraph[:chunk_chars])
                paragraph = paragraph[chunk_chars:]
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def compress_text(text: str, max_chars: int, chunk_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    chunks = split_chunks(text, chunk_chars)
    if len(chunks) <= 2:
        return text[:max_chars]

    head_budget = max_chars // 3
    tail_budget = max_chars // 3
    middle_budget = max_chars - head_budget - tail_budget - 80
    head = text[:head_budget]
    tail = text[-tail_budget:]
    middle_index = len(chunks) // 2
    middle = chunks[middle_index][: max(0, middle_budget)]
    return f"{head}\n\n[... middle omitted for length ...]\n\n{middle}\n\n[... tail ...]\n\n{tail}"
