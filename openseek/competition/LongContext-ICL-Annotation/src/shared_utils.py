"""
Shared utility functions extracted from main.py.
Used by all task modules.
"""
import re
import ast
from collections import Counter, defaultdict


# ==================== Text processing ====================

EMOJI_PATTERN = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U0001F900-\U0001F9FF"
    u"\U0001FA00-\U0001FA6F"
    u"\U0001FA70-\U0001FAFF"
    u"\U0000200D"
    "]+", flags=re.UNICODE)


def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub('', text).strip()


BROKEN_ENCODING_PATTERN = re.compile(r'\?\xad|\xad')


def fix_broken_encoding(text: str) -> str:
    return BROKEN_ENCODING_PATTERN.sub('', text)


# ==================== Token padding ====================

def pad_to_min_tokens_dynamic(dynamic_str, selected_set_ids, padding_pool, min_icl_tokens, max_icl_tokens, qwen_tokenizer):
    """Dynamic padding: add examples from padding_pool until token count reaches min_icl_tokens."""
    token_count = len(qwen_tokenizer.encode(dynamic_str, add_special_tokens=False))

    if token_count >= min_icl_tokens:
        return dynamic_str, token_count

    extras = [ex for ex in padding_pool if id(ex) not in selected_set_ids]

    prepend_parts = []
    for ex in extras:
        if token_count >= max_icl_tokens:
            break
        label_str = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
        part = f"Input:\n{ex['input']}\nOutput:\n<label>{label_str}</label>\n\n"

        part_tokens = len(qwen_tokenizer.encode(part, add_special_tokens=False))
        if token_count + part_tokens > max_icl_tokens:
            break

        prepend_parts.append(part)
        token_count += part_tokens

    if prepend_parts:
        dynamic_str = "".join(prepend_parts) + dynamic_str
        token_count = len(qwen_tokenizer.encode(dynamic_str, add_special_tokens=False))

    return dynamic_str, token_count


# ==================== Label balancing ====================

def balance_by_label_target_k(examples, target_k=20):
    """Select target_k examples with balanced labels, preserving BM25 order within each label."""
    groups = defaultdict(list)
    for idx, ex in enumerate(examples):
        label = str(ex['output'][0] if isinstance(ex['output'], list) else ex['output']).strip()
        groups[label].append((idx, ex))

    labels = list(groups.keys())
    if not labels:
        return []

    per_label = target_k // len(labels)
    balanced = []
    for label in labels:
        balanced.extend(groups[label][:per_label])

    remaining = target_k - len(balanced)
    if remaining > 0:
        used_indices = {idx for idx, _ in balanced}
        extras = [(idx, ex) for idx, ex in enumerate(examples) if idx not in used_indices]
        balanced.extend(extras[:remaining])

    balanced.sort(key=lambda x: x[0])
    return [ex for _, ex in balanced]


def balance_by_label(examples, use_k=40):
    """Balance labels for binary classification, preserving BM25 order."""
    groups = defaultdict(list)
    for idx, ex in enumerate(examples):
        label = str(ex['output'][0] if isinstance(ex['output'], list) else ex['output']).strip()
        groups[label].append((idx, ex))

    labels = list(groups.keys())
    per_label = use_k // len(labels) if labels else use_k

    balanced = []
    for label in labels:
        balanced.extend(groups[label][:per_label])

    remaining = use_k - len(balanced)
    if remaining > 0:
        used_indices = {idx for idx, _ in balanced}
        extras = [(idx, ex) for idx, ex in enumerate(examples) if idx not in used_indices]
        balanced.extend(extras[:remaining])

    balanced.sort(key=lambda x: x[0])
    return [ex for _, ex in balanced]


# ==================== Retry logic ====================

def needs_retry(pred, task_id):
    """Check if prediction needs retry."""
    if not pred or not pred.strip():
        return True, "empty"
    if task_id in (1, 2) and not re.fullmatch(r'-?\d+', pred):
        return True, f"invalid='{pred}'"
    if task_id == 5 and pred not in ("Sad", "Not sad"):
        return True, f"invalid='{pred}'"
    if task_id == 6 and pred not in ("Y", "N"):
        return True, f"invalid='{pred}'"
    if task_id == 7 and not pred:
        return True, "empty"
    if task_id == 8 and not pred:
        return True, "empty"
    return False, ""


# ==================== T4 validation ====================

def sort_by_score(examples, scores, reverse=False):
    """Sort examples by corresponding scores."""
    paired = list(zip(scores, examples))
    paired.sort(key=lambda x: x[0], reverse=reverse)
    return [ex for _, ex in paired]


# ==================== T4 validation ====================

def t4_validate_output(input_str: str, output: str) -> tuple:
    try:
        lst = ast.literal_eval(input_str)
    except Exception:
        return True, []

    expected = ''.join(lst)
    reasons = []

    if len(output) != len(expected):
        reasons.append(f"length_mismatch: got {len(output)}, expected {len(expected)}")

    if Counter(output) != Counter(expected):
        reasons.append("char_mismatch: output char frequencies differ from input")

    idx = -1
    order_ok = True
    for s in lst:
        for ch in s:
            idx = output.find(ch, idx + 1)
            if idx == -1:
                order_ok = False
                break
        if not order_ok:
            break
    if not order_ok:
        reasons.append("order_violation: some substring characters appear out of order")

    return (len(reasons) == 0, reasons)
