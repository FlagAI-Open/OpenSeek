from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 896, overlap: int = 96) -> List[Dict[str, int | str]]:
    if not text:
        return [{"cid": "chunk-0", "text": "", "start": 0, "end": 0}]

    step = max(1, chunk_size - overlap)
    chunks: List[Dict[str, int | str]] = []
    index = 0
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(
            {
                "cid": f"chunk-{index}",
                "text": chunk,
                "start": start,
                "end": end,
            }
        )
        index += 1
        if end >= len(text):
            break
    return chunks
