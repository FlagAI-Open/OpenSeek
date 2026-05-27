from typing import Dict, List


def reorder_front_back(chunks: List[Dict[str, int | str]]) -> List[Dict[str, int | str]]:
    if len(chunks) <= 2:
        return chunks

    ordered: List[Dict[str, int | str]] = []
    left = 0
    right = len(chunks) - 1
    toggle = True
    while left <= right:
        if toggle:
            ordered.append(chunks[left])
            left += 1
        else:
            ordered.append(chunks[right])
            right -= 1
        toggle = not toggle
    return ordered
