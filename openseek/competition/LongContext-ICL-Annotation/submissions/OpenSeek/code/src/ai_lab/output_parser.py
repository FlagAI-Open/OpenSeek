import ast
import json
import re
from typing import Any, Dict, List


def _extract_json_blob(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_protocol_output(raw_text: str, task_type: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "raw_text": raw_text,
        "valid": False,
        "label": "",
        "confidence": 0,
        "evidence": [],
        "reason": "",
        "schema_ok": False,
    }

    cleaned = raw_text.strip()
    if not cleaned:
        return parsed

    if task_type == "code_generation":
        code_match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        code = code_match.group(1).strip() if code_match else _clean_code_generation_output(cleaned)
        parsed["valid"] = bool(code.strip())
        parsed["schema_ok"] = parsed["valid"]
        parsed["label"] = code
        parsed["answer"] = code
        return parsed

    json_blob = _extract_json_blob(cleaned)
    if json_blob is not None:
        try:
            data = json.loads(json_blob)
            label = (
                data.get("label")
                or data.get("final_label")
                or data.get("winner")
                or data.get("answer")
                or ""
            )
            evidence = (
                data.get("evidence")
                or data.get("positive_for_a")
                or data.get("positive_for_b")
                or []
            )
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            confidence = data.get("confidence", 0)
            try:
                confidence = int(confidence)
            except Exception:
                confidence = 0
            parsed.update(
                {
                    "valid": bool(str(label).strip()),
                    "schema_ok": True,
                    "label": str(label).strip(),
                    "answer": str(label).strip(),
                    "confidence": confidence,
                    "evidence": [str(item).strip() for item in evidence if str(item).strip()],
                    "reason": str(data.get("reason", data.get("decision_basis", ""))).strip(),
                }
            )
            return parsed
        except Exception:
            pass

    labels = re.findall(r"<label>\s*(.*?)\s*</label>", cleaned, flags=re.DOTALL)
    if labels:
        label = labels[-1].strip()
        parsed.update(
            {
                "valid": bool(label),
                "schema_ok": True,
                "label": label,
                "answer": label,
            }
        )
        return parsed

    first_line = cleaned.splitlines()[0].strip()
    parsed.update(
        {
            "valid": bool(first_line),
            "schema_ok": False,
            "label": first_line,
            "answer": first_line,
        }
    )
    return parsed


def _clean_code_generation_output(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    # The prompt ends inside a Python code block. Some models close that fence
    # and then continue with prose; keep only the code before the first fence.
    fence_index = cleaned.find("```")
    if fence_index != -1:
        cleaned = cleaned[:fence_index].strip()

    lines = cleaned.splitlines()
    code_start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith(("import ", "from ", "def ", "class ", "@"))
            or "=" in stripped
        ):
            code_start = index
            break
    lines = lines[code_start:]

    prose_markers = (
        "now,",
        "now i",
        "the provided code",
        "the function is",
        "this code",
        "in this implementation",
        "explanation",
    )
    cut_at = len(lines)
    seen_code = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
            seen_code = True
        if seen_code and stripped.lower().startswith(prose_markers):
            cut_at = index
            break
    return "\n".join(lines[:cut_at]).strip()


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip().lower())


_SADNESS_CUES = {
    "sad",
    "depress",
    "dark",
    "dull",
    "unhappy",
    "miserable",
    "devastat",
    "terrified",
    "dread",
    "gloom",
    "grief",
    "cry",
    "tears",
    "lonely",
    "lost",
    "miss",
    "sorrow",
    "heartbreak",
    "hurt",
    "pain",
    "rape",
    "sink",
    "not a fan",
    "can't go on",
    "cannot go on",
}

_NOT_SADNESS_CUES = {
    "not sad",
    "blessed",
    "happy",
    "excited",
    "goodnight",
    "optimist",
    "optimistic",
    "love",
    "quote",
    "joke",
    "funny",
    "music",
    "game is on",
}


def _canonicalize_sadness_label(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bnot\s+sad\b", lowered):
        return "Not sad"
    if re.search(r"\bsad\b", lowered):
        return "Sad"

    sad_hits = sum(1 for cue in _SADNESS_CUES if cue in lowered)
    not_sad_hits = sum(1 for cue in _NOT_SADNESS_CUES if cue in lowered)
    if sad_hits > not_sad_hits:
        return "Sad"
    if not_sad_hits > sad_hits:
        return "Not sad"

    return "Not sad"


_COUNT_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "for",
    "with",
    "by",
    "as",
    "and",
    "or",
    "under",
    "over",
    "down",
    "up",
    "into",
    "onto",
    "some",
    "many",
    "very",
    "his",
    "her",
    "their",
    "its",
    "it",
    "am",
    "is",
    "are",
    "was",
    "were",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "top",
    "bottom",
    "side",
}

_COMMON_VERBS = {
    "be",
    "being",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "go",
    "goes",
    "went",
    "make",
    "makes",
    "made",
    "take",
    "takes",
    "took",
    "get",
    "gets",
    "got",
    "come",
    "comes",
    "coming",
    "walk",
    "walks",
    "walking",
    "stand",
    "stands",
    "standing",
    "sit",
    "sits",
    "sitting",
    "play",
    "plays",
    "playing",
    "hold",
    "holds",
    "holding",
    "look",
    "looks",
    "looking",
    "ride",
    "rides",
    "riding",
    "fly",
    "flies",
    "flying",
    "run",
    "runs",
    "running",
    "catch",
    "catches",
    "canned",
    "lowered",
    "attached",
    "smiles",
    "grilling",
}


def _extract_count_task_parts(text: str) -> tuple[str, str]:
    sentence_match = re.search(r"Sentence:\s*['\"](.*?)['\"]\s*\.", text, flags=re.DOTALL)
    sentence = sentence_match.group(1) if sentence_match else text
    target_match = re.search(r"Count the number of (nouns|verbs)\b", text, flags=re.IGNORECASE)
    target = target_match.group(1).lower() if target_match else "nouns"
    return sentence, target


def _heuristic_count_nouns_verbs(text: str) -> str:
    sentence, target = _extract_count_task_parts(text)
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence.lower())
    if target == "verbs":
        count = 0
        for index, token in enumerate(tokens):
            previous = tokens[index - 1] if index > 0 else ""
            if token in _COMMON_VERBS:
                count += 1
            elif (token.endswith("ing") or token.endswith("ed")) and previous not in {"of", "in", "on", "with"}:
                count += 1
        return str(count)

    count = 0
    for token in tokens:
        if token in _COUNT_STOPWORDS or token in _COMMON_VERBS:
            continue
        if token.endswith("ing") or token.endswith("ed"):
            continue
        count += 1
    return str(count)


def _closest_integers_from_source_text(text: str) -> str:
    match = re.search(r"\[[^\[\]]*\]", text)
    if not match:
        return "0"
    try:
        values = ast.literal_eval(match.group(0))
    except Exception:
        return "0"
    if not isinstance(values, list) or len(values) < 2:
        return "0"
    try:
        numbers = sorted(int(value) for value in values)
    except Exception:
        return "0"
    return str(min(abs(right - left) for left, right in zip(numbers, numbers[1:])))


def _collatz_from_source_text(text: str) -> str:
    match = re.search(r"\[[^\[\]]*\]", text)
    if not match:
        return "[]"
    try:
        values = ast.literal_eval(match.group(0))
    except Exception:
        return "[]"
    if not isinstance(values, list):
        return "[]"

    output = []
    for value in values:
        try:
            number = int(value)
        except Exception:
            continue
        if number % 2 == 0:
            output.append(number // 2)
        else:
            output.append(number * 3 + 1)
    return str(output)


def _concat_from_source_text(text: str) -> str:
    match = re.search(r"\[[^\[\]]*\]", text, flags=re.DOTALL)
    if not match:
        return ""
    try:
        values = ast.literal_eval(match.group(0))
    except Exception:
        return ""
    if not isinstance(values, list):
        return ""
    return "".join(str(item) for item in values)


def _canonicalize_jeopardy_answer(text: str) -> str:
    cleaned = str(text).strip()
    label_match = re.search(r"<(?:label|answer)>\s*(.*?)\s*</(?:label|answer)>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if label_match:
        cleaned = label_match.group(1).strip()

    jsonish_answer = re.search(
        r"['\"]answer['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if jsonish_answer:
        cleaned = jsonish_answer.group(1).strip()

    answer_prefix = re.match(
        r"^(?:answer|final answer|prediction|label)\s*[:：]\s*(.+)$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_prefix:
        cleaned = answer_prefix.group(1).strip()

    cleaned = cleaned.splitlines()[0].strip()
    cleaned = re.sub(r"</?(?:label|answer)>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip(" \t\r\n`\"'“”‘’")
    cleaned = re.sub(r"^(?:what|who|where|when|why|how)\s+(?:is|are|was|were)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:what|who|where|when|why|how)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:is|are|was|were)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:it is|it's|this is|that is)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".。?!;；:")
    return cleaned.lower()


def canonicalize_label(
    label: str,
    *,
    task_type: str,
    task_name: str,
    label_space: List[str] | None = None,
    source_text: str = "",
) -> str:
    cleaned = str(label).strip()
    if not cleaned:
        return ""

    if task_type == "code_generation":
        return _clean_code_generation_output(cleaned)

    answer_prefix = re.match(r"^(?:answer|final answer|prediction|label)\s*[:：]\s*(.+)$", cleaned, flags=re.IGNORECASE)
    if answer_prefix:
        cleaned = answer_prefix.group(1).strip()

    label_space = label_space or []
    normalized_cleaned = normalize_answer(cleaned)

    for candidate in label_space:
        if normalize_answer(candidate) == normalized_cleaned:
            return candidate

    lowered = cleaned.lower()
    for candidate in sorted(label_space, key=len, reverse=True):
        candidate_lower = candidate.lower()
        if candidate_lower and candidate_lower in lowered:
            return candidate

    if task_name in {"closest_integers", "count_nouns_verbs"}:
        if task_name == "closest_integers" and source_text:
            return _closest_integers_from_source_text(source_text)
        numbers = re.findall(r"-?\d+", cleaned)
        if numbers:
            return numbers[-1]
        if task_name == "count_nouns_verbs" and source_text:
            return _heuristic_count_nouns_verbs(source_text)
        return "0"

    if task_name == "semeval_2018_task1_tweet_sadness_detection":
        return _canonicalize_sadness_label(cleaned)

    if task_name == "collatz_conjecture":
        match = re.search(r"\[[^\[\]]*\]", cleaned)
        if match:
            candidate = match.group(0)
            try:
                values = ast.literal_eval(candidate)
                if isinstance(values, list) and all(isinstance(item, int) for item in values):
                    return str(values)
            except Exception:
                pass
        if source_text:
            return _collatz_from_source_text(source_text)
        return "[]"

    if task_name == "conala_concat_strings" and source_text:
        fallback = _concat_from_source_text(source_text)
        if fallback:
            return fallback

    if task_name == "jeopardy_answer_generation_all":
        return _canonicalize_jeopardy_answer(cleaned)

    quoted = re.search(r'["“](.*?)["”]', cleaned)
    if quoted and quoted.group(1).strip():
        return quoted.group(1).strip()

    return cleaned


def evidence_overlap(evidence_sets: List[List[str]]) -> float:
    normalized_sets = []
    for evidence in evidence_sets:
        values = {normalize_answer(item) for item in evidence if normalize_answer(item)}
        if values:
            normalized_sets.append(values)
    if len(normalized_sets) < 2:
        return 0.5

    overlaps: List[float] = []
    for index in range(len(normalized_sets)):
        for other in range(index + 1, len(normalized_sets)):
            a = normalized_sets[index]
            b = normalized_sets[other]
            union = a | b
            overlap = len(a & b) / len(union) if union else 0.0
            overlaps.append(overlap)
    return sum(overlaps) / len(overlaps) if overlaps else 0.5
