import argparse
import ast
import json
import os
import re
import requests
from collections import Counter
from functools import lru_cache
from tqdm import tqdm
from transformers import AutoTokenizer

_TOKENIZER = None
TASK1_LONG_CONTEXT_TARGET_TOKENS = 30000
TASK1_LONG_CONTEXT_MAX_TOKENS = 30500
DEFAULT_TRACE_LINE_NUMBERS = ""
DEFAULT_ONLY_LINE_NUMBERS = ""
TASK1_FILE = "./data/openseek-1_closest_integers.json"
TASK1_LONG_CONTEXT_INSTRUCTION = (
    "Long-context prepass reference: solve the minimum absolute difference task by using only the "
    "provided integers; duplicates imply answer 0; otherwise sort ascending and compare adjacent gaps; "
    "round one must return only the requested XML schema. "
)

def _get_qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        # 路径会由 evaluate.py 传入，这里作为默认 fallback
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen3-4B", trust_remote_code=True)
    return _TOKENIZER


def _build_task1_short_prompt(text2annotate: str) -> str:
    return (
        "You are solving minimum absolute difference.\n\n"
        "Given an integer list:\n"
        "1. Sort it in ascending order.\n"
        "2. Compute adjacent differences only:\n"
        "   diffs[i] = sorted[i+1] - sorted[i]\n"
        "3. The diffs list must contain exactly len(input)-1 integers.\n"
        "4. Do not copy the sorted list into <diffs>.\n"
        "5. Do not write equations or explanations inside <diffs>.\n"
        "6. Output only these three XML fields and nothing else:\n\n"
        "<sorted>[ascending sorted list]</sorted>\n"
        "<diffs>[comma-separated adjacent differences]</diffs>\n"
        "<label>minimum value in diffs</label>\n\n"
        "Example:\n"
        "Input: [5, -1, 9, 2]\n"
        "<sorted>[-1, 2, 5, 9]</sorted>\n"
        "<diffs>[3, 3, 4]</diffs>\n"
        "<label>3</label>\n\n"
        f"Now solve:\nInput: {text2annotate}\n"
    )


def _build_task1_xml_retry_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    feedback = ""
    if previous_output:
        feedback = (
            "The previous response was invalid.\n"
            "Common mistakes:\n"
            "1. <diffs> repeated the sorted list.\n"
            "2. <diffs> used negative values because subtraction direction was wrong.\n"
            "3. The response included extra words or explanations.\n"
            "4. The response omitted one of the required XML fields.\n\n"
        )
    return (
        "You are solving minimum absolute difference.\n"
        f"{feedback}"
        "Return only these three XML fields and nothing else:\n"
        "<sorted>[ascending sorted list]</sorted>\n"
        "<diffs>[comma-separated adjacent differences]</diffs>\n"
        "<label>minimum value in diffs</label>\n\n"
        "Rules:\n"
        "1. Use exactly the input integers.\n"
        "2. Sort ascending.\n"
        "3. In <diffs>, compute NEXT minus CURRENT.\n"
        "4. Every diff must be non-negative.\n"
        "5. The diffs list must contain exactly len(input)-1 integers.\n"
        "6. Do not output equations, words, or explanations.\n\n"
        "Example 1:\n"
        "Input: [71, -93, 63, -41, -18, 18]\n"
        "<sorted>[-93, -41, -18, 18, 63, 71]</sorted>\n"
        "<diffs>[52, 23, 36, 45, 8]</diffs>\n"
        "<label>8</label>\n\n"
        "Example 2:\n"
        "Input: [-84, 79, -59, -31, -62, -52, 78]\n"
        "<sorted>[-84, -62, -59, -52, -31, 78, 79]</sorted>\n"
        "<diffs>[22, 3, 7, 21, 109, 1]</diffs>\n"
        "<label>1</label>\n\n"
        f"Now solve:\nInput: {text2annotate}\n"
    )


@lru_cache(maxsize=1)
def _task1_unit_tokens() -> int:
    tokenizer = _get_qwen_tokenizer()
    return len(tokenizer.encode(TASK1_LONG_CONTEXT_INSTRUCTION, add_special_tokens=False))


def _task1_exact_token_len(text: str) -> int:
    tokenizer = _get_qwen_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _task1_build_30k_shell(task_description: str, text2annotate: str) -> str:
    short_prompt = _build_task1_short_prompt(text2annotate)
    unit_tokens = max(1, _task1_unit_tokens())
    short_tokens = _task1_exact_token_len(short_prompt)
    repeat_count = max(1, (TASK1_LONG_CONTEXT_TARGET_TOKENS - short_tokens) // unit_tokens)
    appendix = (TASK1_LONG_CONTEXT_INSTRUCTION * repeat_count).strip()
    shell = (
        "MinGapAudit-S12 two-round long-context wrapper.\n"
        "Round 1 must read the appendix and the active task, then output XML only.\n\n"
        "<reference_appendix>\n"
        f"{appendix}\n"
        "</reference_appendix>\n\n"
        "<active_task>\n"
        "Task Description:\n"
        f"{task_description}\n\n"
        "Text To Annotate:\n"
        f"{text2annotate}\n"
        "</active_task>\n\n"
        "Round 1 output schema only:\n"
        "<analysis><status>usable|fallback</status><duplicate_hint>yes|no|unknown</duplicate_hint>"
        "<focus>short hint or fallback</focus></analysis>\n"
    )
    while _task1_exact_token_len(shell) < TASK1_LONG_CONTEXT_TARGET_TOKENS:
        shell = shell.replace("</reference_appendix>", TASK1_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", 1)
    while _task1_exact_token_len(shell) > TASK1_LONG_CONTEXT_MAX_TOKENS:
        shell = shell.replace(TASK1_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", "</reference_appendix>", 1)
    return shell


def _task1_parse_shell(input_prompt: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"Task Description:\s*(.*?)\n\s*Text To Annotate:\s*(.*?)\s*</active_task>",
        input_prompt,
        flags=re.DOTALL,
    )
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _task1_parse_analysis(text: str | None) -> tuple[str | None, str | None, str | None]:
    if not text:
        return None, None, None
    status_match = re.search(r"<status>\s*(usable|fallback)\s*</status>", text, flags=re.IGNORECASE)
    duplicate_match = re.search(
        r"<duplicate_hint>\s*(yes|no|unknown)\s*</duplicate_hint>",
        text,
        flags=re.IGNORECASE,
    )
    focus_match = re.search(r"<focus>\s*(.*?)\s*</focus>", text, flags=re.IGNORECASE | re.DOTALL)
    status = status_match.group(1).lower() if status_match else None
    duplicate_hint = duplicate_match.group(1).lower() if duplicate_match else None
    focus = None
    if focus_match:
        focus = re.sub(r"\s+", " ", focus_match.group(1)).strip()
        if len(focus) > 80:
            focus = focus[:80].rstrip()
    return status, duplicate_hint, focus


@lru_cache(maxsize=1)
def _task1_model_id() -> str:
    try:
        resp = requests.get("http://0.0.0.0:2026/v1/models", timeout=30)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models and "id" in models[0]:
            return models[0]["id"]
    except Exception:
        pass
    return "../Qwen3-4B"


def _task1_chat_request(prompt: str, *, system: str, max_tokens: int, stop: list[str] | None) -> str | None:
    data = {
        "model": _task1_model_id(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stop is not None:
        data["stop"] = stop
    try:
        resp = requests.post("http://0.0.0.0:2026/v1/chat/completions", json=data, timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def build_prompt(task_id: int, task_description: str, text2annotate: str) -> str:
    if task_id == 1:
        return _task1_build_30k_shell(task_description, text2annotate)

    return (
        "### Task\n"
        f"{task_description}\n\n"
        "### Examples\n"
        "[[EXAMPLES]]\n\n"
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Output Format\n"
        "<label>YOUR_ANSWER</label>\n"
    )

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    del task_description
    tokenizer = _get_qwen_tokenizer()
    target_length = 4096
    query_nums = _parse_task1_input_list(text2annotate) or []

    def _profile(nums: list[int]) -> tuple[int, int, int]:
        has_duplicate = int(len(nums) != len(set(nums))) if nums else 0
        sign_mix = int(any(x < 0 for x in nums) and any(x >= 0 for x in nums)) if nums else 0
        span_bucket = 0
        if nums:
            span = max(nums) - min(nums)
            if span >= 120:
                span_bucket = 2
            elif span >= 40:
                span_bucket = 1
        return has_duplicate, sign_mix, span_bucket

    query_profile = _profile(query_nums)
    scored_examples: list[tuple[tuple[int, int, int, int], str]] = []
    for example in all_examples:
        try:
            example_input = example["input"]
            example_nums = _parse_task1_input_list(example_input)
            if example_nums is None:
                continue
            example_profile = _profile(example_nums)
            duplicate_match = int(example_profile[0] == query_profile[0])
            sign_match = int(example_profile[1] == query_profile[1])
            span_match = int(example_profile[2] == query_profile[2])
            len_gap = abs(len(example_nums) - len(query_nums))
            score = (-duplicate_match, -sign_match, -span_match, len_gap)
            example_str = f"Input: {example_input}\nOutput: <label>{example['output'][0]}</label>\n\n"
            scored_examples.append((score, example_str))
        except Exception:
            continue

    scored_examples.sort(key=lambda item: item[0])
    examples_str, token_num = "", 0
    for _, example_str in scored_examples:
        length = len(tokenizer.encode(example_str, add_special_tokens=False))
        if token_num + length <= target_length:
            examples_str += example_str
            token_num += length
        else:
            break
    return examples_str


def _extract_task1_text_from_prompt(input_prompt: str) -> str | None:
    for pattern in (
        r"### Text to Annotate\s*(.*?)\s*### Output Format",
        r"Text To Annotate:\s*(.*?)\s*</active_task>",
        r"Input:\s*(\[[^\n]*\])\s*$",
    ):
        match = re.search(pattern, input_prompt, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _build_task1_retry_prompt(base_prompt: str, previous_output: str | None = None, *, duplicate_seen: bool = False) -> str:
    feedback_lines = [
        "Reminder:",
        "1. Return only <label>INTEGER</label>.",
        "2. Do not output explanations, steps, or extra text.",
    ]
    if duplicate_seen:
        feedback_lines.append("3. The input contains a repeated integer, so the answer must be 0.")
    if previous_output:
        feedback_lines.append(f"Previous invalid output: {previous_output.strip()}")
    return f"{base_prompt}\n\n### Retry\n" + "\n".join(feedback_lines) + "\nOutput: <label>"

def _has_duplicates(text: str) -> bool:
    nums = re.findall(r"[-+]?\d+", text or "")
    if not nums: return False
    vals = [int(x) for x in nums]
    return len(set(vals)) != len(vals)

def _parse_task1_input_list(text: str) -> list[int] | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return value
    return None

def _parse_tagged_int_list(text: str, tag: str) -> list[int] | None:
    if not text:
        return None
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    if not content:
        return None

    # First try the strict Python-list form: [1, 2, 3]
    try:
        value = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        value = None
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return value

    # Then accept a looser comma-separated form inside the tag: 1, 2, 3
    nums = re.findall(r"[-+]?\d+", content)
    if nums:
        return [int(x) for x in nums]
    return None

def _parse_task1_label(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"<label>\s*(\d+)\s*</label>", text, flags=re.DOTALL)
    return match.group(1) if match else None

def _is_non_decreasing(nums: list[int]) -> bool:
    return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))

def _is_valid_sorted_version(original: list[int], candidate: list[int] | None) -> bool:
    if candidate is None or len(original) != len(candidate):
        return False
    return Counter(original) == Counter(candidate) and _is_non_decreasing(candidate)

def _describe_sorted_candidate_issue(original: list[int], candidate: list[int] | None) -> str:
    if candidate is None:
        return "The proposed sorted list could not be parsed as a valid integer list."
    if len(candidate) != len(original):
        return (
            f"The proposed sorted list has {len(candidate)} integers, but the input has {len(original)} integers."
        )
    if not _is_non_decreasing(candidate):
        for idx in range(len(candidate) - 1):
            if candidate[idx] > candidate[idx + 1]:
                return (
                    "The proposed sorted list is not ascending because "
                    f"{candidate[idx]} appears before smaller value {candidate[idx + 1]}."
                )
    original_counter = Counter(original)
    candidate_counter = Counter(candidate)
    extras = list((candidate_counter - original_counter).elements())
    missing = list((original_counter - candidate_counter).elements())
    if extras or missing:
        extras_text = ", ".join(str(x) for x in extras[:4]) if extras else "none"
        missing_text = ", ".join(str(x) for x in missing[:4]) if missing else "none"
        return (
            "The proposed sorted list changed the multiset of integers. "
            f"Extra/changed values: {extras_text}. Missing values: {missing_text}."
        )
    return "The proposed sorted list is inconsistent with the input."

def _build_diffs_template(num_diffs: int) -> str:
    return "[" + ", ".join("?" for _ in range(num_diffs)) + "]"

def _build_task1_sort_prompt(text2annotate: str) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Task: sort the given integers in ascending order.\n"
        "Use exactly the integers from the input. Do not add, remove, duplicate, or modify any value.\n"
        "Negative signs must be preserved.\n"
        "Do not copy any example output. Build the answer only from the CURRENT input integers.\n"
        "Before answering, check that every output integer appears in the current input and that no current input integer is missing.\n"
        "Before answering, scan every adjacent pair in your output and ensure output[i] <= output[i+1] for all i.\n"
        "Return only the sorted list in the format <sorted>[a, b, c]</sorted>.\n\n"
        "Example:\n"
        "Input: [71, -93, 63, -41, -18, 18]\n"
        "Output: <sorted>[-93, -41, -18, 18, 63, 71]</sorted>\n\n"
        "Wrong output example:\n"
        "Input: [13, 91, 52, 97]\n"
        "Output: <sorted>[13, 91, 52, 97]</sorted>\n"
        "This is wrong because 91 appears before smaller value 52.\n\n"
        "Wrong output example:\n"
        "Input: [75, 82, -22, 85, 96, -38, -25]\n"
        "Output: <sorted>[-93, -41, -18, 18, 63, 71]</sorted>\n"
        "This is wrong because it copies numbers that are not in the current input.\n\n"
        f"Input: {text2annotate}\n"
        "Output: <sorted>"
    )

def _build_task1_sort_retry_prompt(
    text2annotate: str,
    original_nums: list[int],
    previous_output: str | None = None,
) -> str:
    expected_count = len(original_nums)
    input_csv = ", ".join(str(x) for x in original_nums)
    feedback = ""
    if previous_output:
        feedback = (
            "The previous sorted list was invalid.\n"
            "It either changed numbers, omitted numbers, added numbers, or was not in ascending order.\n"
        )
    return (
        "You are a careful mathematical assistant.\n"
        f"{feedback}"
        "Sort the integers again from scratch.\n"
        f"You must output exactly {expected_count} integers.\n"
        "Use every input integer exactly once.\n"
        "Do not create any new integer.\n"
        "Do not merge digits (for example, 11 must not become 111).\n"
        "Do not drop any negative sign.\n"
        "The output must be in ascending order.\n"
        "Before answering, scan every adjacent pair and ensure no larger number appears before a smaller number.\n"
        "Do not copy any previous example output or unrelated list.\n"
        "Every output number must come from the current input integers shown below.\n"
        "Return only <sorted>[...]</sorted>.\n\n"
        f"Input integers: {input_csv}\n"
        f"Input: {text2annotate}\n"
        "Output: <sorted>"
    )

def _build_task1_sort_repair_prompt(text2annotate: str, original_nums: list[int]) -> str:
    expected_count = len(original_nums)
    indexed_items = ", ".join(f"{i+1}:{x}" for i, x in enumerate(original_nums))
    return (
        "You are a careful mathematical assistant.\n"
        "Repair the sorted list using the original integers only.\n"
        f"There are exactly {expected_count} integers.\n"
        "Each output integer must be copied from the original input.\n"
        "Do not invent new integers.\n"
        "Do not merge digits.\n"
        "Do not omit any integer.\n"
        "Keep all negative signs.\n"
        "The final list must satisfy output[i] <= output[i+1] for every adjacent pair.\n"
        "Do not copy a different example list.\n"
        "Return only the ascending list in <sorted>[...]</sorted>.\n\n"
        f"Indexed input integers: {indexed_items}\n"
        f"Original input: {text2annotate}\n"
        "Output: <sorted>"
    )

def _build_task1_sort_backup_prompt(text2annotate: str, original_nums: list[int]) -> str:
    expected_count = len(original_nums)
    csv_items = ", ".join(str(x) for x in original_nums)
    return (
        "You are a careful mathematical assistant.\n"
        "Final attempt: produce a valid ascending permutation of the exact input integers.\n"
        f"The answer must contain exactly {expected_count} integers.\n"
        "It must be a permutation of the input integers and nothing else.\n"
        "Check carefully that every original integer appears once.\n"
        "Check carefully that the result is in ascending order.\n"
        "If any adjacent pair decreases, the answer is invalid.\n"
        "If even one output integer is not in the input, the answer is invalid.\n"
        "Return only <sorted>[...]</sorted>.\n\n"
        "Example:\n"
        "Input integers: 83, -95, 92, 20, -83, 49, 6\n"
        "Output: <sorted>[-95, -83, 6, 20, 49, 83, 92]</sorted>\n\n"
        f"Input integers: {csv_items}\n"
        f"Original input: {text2annotate}\n"
        "Output: <sorted>"
    )

def _build_task1_sort_verify_prompt(text2annotate: str, proposed_sorted_raw: str) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Audit the proposed sorted list.\n"
        "Check all of the following:\n"
        "1. It uses exactly the integers from the original input.\n"
        "2. No integer is added, removed, duplicated, merged, or changed.\n"
        "3. All negative signs are preserved.\n"
        "4. The list is in ascending order.\n"
        "5. If any adjacent pair decreases, the answer must fail.\n"
        "If all checks pass, output exactly <verify>pass</verify>.\n"
        "Otherwise output exactly <verify>fail</verify>.\n"
        "Do not output explanations.\n\n"
        "Example 1:\n"
        "Input: [71, -93, 63, -41, -18, 18]\n"
        "Proposed: <sorted>[-93, -41, -18, 18, 63, 71]</sorted>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 2:\n"
        "Input: [-21, 100, 79, 10, -37, 16, -34, 62, 5, -97]\n"
        "Proposed: <sorted>[-97, -37, -34, -21, 5, 10, 16, 52, 62, 79, 100]</sorted>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 3:\n"
        "Input: [87, -94, -52, 35, 95, -93, 70]\n"
        "Proposed: <sorted>[-94, -52, -93, 35, 70, 87, 95]</sorted>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 4:\n"
        "Input: [-26, 14, 36, 10]\n"
        "Proposed: <sorted>[-26, 10, 14, 36]</sorted>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 5:\n"
        "Input: [13, 91, 52, 97]\n"
        "Proposed: <sorted>[13, 91, 52, 97]</sorted>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 6:\n"
        "Input: [-27, -62, -71, -76, -25, 81, -54]\n"
        "Proposed: <sorted>[-76, -75, -71, -62, -27, -25, 81]</sorted>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 7:\n"
        "Input: [78, -72, -79, -74]\n"
        "Proposed: <sorted>[-79, -78, -74, 78]</sorted>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 8:\n"
        "Input: [75, 82, -22, 85, 96, -38, -25]\n"
        "Proposed: <sorted>[-93, -41, -18, 18, 63, 71]</sorted>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 9:\n"
        "Input: [-85, 47, 39, 74, -66, -12, -82, 23, -30]\n"
        "Proposed: <sorted>[-85, -66, -82, -30, -12, 23, 39, 47, 74]</sorted>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 10:\n"
        "Input: [-33, 24, 65, 18]\n"
        "Proposed: <sorted>[-33, 18, 24, 65]</sorted>\n"
        "Output: <verify>pass</verify>\n\n"
        f"Input: {text2annotate}\n"
        f"Proposed: {proposed_sorted_raw}\n"
        "Output: <verify>"
    )

def _build_task1_answer_prompt(text2annotate: str, sorted_nums: list[int]) -> str:
    num_diffs = len(sorted_nums) - 1
    template = _build_diffs_template(num_diffs)
    return (
        "You are a careful mathematical assistant.\n"
        "Task: compute the adjacent differences from an already sorted integer list.\n"
        "For sorted [a, b, c, d], the output must be [b-a, c-b, d-c].\n"
        "Important: each value is NEXT minus CURRENT.\n"
        "Because the list is sorted ascending, every output value must be a non-negative integer.\n"
        "Output the differences list only.\n"
        "Each item inside <diffs> must be a final integer result only, never an operand copied from the sorted list.\n"
        "Do not repeat the sorted list.\n"
        "Do not output words.\n"
        "Do not output equations.\n"
        "Do not output minus expressions such as a-b.\n"
        "Do not output explanations.\n"
        f"There must be exactly {num_diffs} integers inside the tag.\n"
        "If even one item is negative, or if even one copied operand appears, the whole answer is invalid.\n"
        f"Return exactly this shell with integers filled in: <diffs>{template}</diffs>\n\n"
        "Example 1:\n"
        "Sorted: [-93, -41, -18, 18, 63, 71]\n"
        "Output: <diffs>[52, 23, 36, 45, 8]</diffs>\n\n"
        "Example 2:\n"
        "Sorted: [-84, -62, -59, -52, -31, 78, 79]\n"
        "Output: <diffs>[22, 3, 7, 21, 109, 1]</diffs>\n\n"
        "Wrong output 1: <diffs>[-84, -62, -59, -52, -31, 78, 79]</diffs>\n"
        "Wrong output 2: <diffs>[-93 - (-41) = -52, ...]</diffs>\n"
        "Wrong output 3: <diffs>The adjacent differences are [52, 23, 36, 45, 8]</diffs>\n\n"
        "Wrong output 4: <diffs>[-59, -88, 29, -3, -59, 56, 62, -3, 65]</diffs>\n"
        "This is wrong because it copies operands from the sorted list instead of outputting only the gap values.\n\n"
        "Wrong output 5: <diffs>[-41, 44, 37, 47]</diffs>\n"
        "This is wrong because gaps from an ascending sorted list cannot be negative.\n\n"
        f"Input: {text2annotate}\n"
        f"Sorted: {sorted_nums}\n"
        "Output: <diffs>"
    )

def _build_task1_answer_retry_prompt(text2annotate: str, sorted_nums: list[int], previous_output: str | None = None) -> str:
    num_diffs = len(sorted_nums) - 1
    template = _build_diffs_template(num_diffs)
    feedback = ""
    if previous_output:
        feedback = (
            "The previous response was invalid.\n"
            "Common mistakes:\n"
            "1. You repeated the sorted list instead of the differences.\n"
            "2. You wrote equations or explanations instead of plain integers.\n"
            "3. You subtracted in the wrong direction and produced negative values.\n"
        )
    return (
        "You are a careful mathematical assistant.\n"
        f"{feedback}"
        "Recompute from the sorted list only.\n"
        "For each adjacent pair, compute NEXT minus CURRENT.\n"
        "Return plain evaluated non-negative integers only.\n"
        "Each item must be the final gap value only; do not include the left number, right number, or any equation fragment.\n"
        "Do not copy the sorted list.\n"
        "Do not output words.\n"
        "Do not output equations.\n"
        "If any item is negative, the answer is invalid.\n"
        f"There must be exactly {num_diffs} integers.\n"
        f"Return exactly this shell: <diffs>{template}</diffs>\n\n"
        "Example 1:\n"
        "Sorted: [-84, -62, -59, -52, -31, 78, 79]\n"
        "Output: <diffs>[22, 3, 7, 21, 109, 1]</diffs>\n\n"
        "Example 2:\n"
        "Sorted: [-63, 29, 40]\n"
        "Output: <diffs>[92, 11]</diffs>\n\n"
        "Wrong output 1: <diffs>[-63, 29, 40]</diffs>\n"
        "Wrong output 2: <diffs>[-63 - 29 = -92, 29 - 40 = -11]</diffs>\n"
        "Wrong output 3: <diffs>The differences are [92, 11]</diffs>\n\n"
        "Wrong output 4: <diffs>[-10, -39, 21, 33, 4, 14]</diffs>\n"
        "This is wrong because negative operands were copied instead of pure adjacent gaps.\n\n"
        "Wrong output 5: <diffs>[18, 50, 10, 30, 2]</diffs>\n"
        "This is wrong because the number of gaps is too small.\n\n"
        f"Input: {text2annotate}\n"
        f"Sorted: {sorted_nums}\n"
        "Output: <diffs>"
    )


def _build_task1_answer_repair_prompt(text2annotate: str, sorted_nums: list[int]) -> str:
    num_diffs = len(sorted_nums) - 1
    template = _build_diffs_template(num_diffs)
    return (
        "You are a careful mathematical assistant.\n"
        "Final repair attempt for adjacent differences.\n"
        "The sorted list is already correct.\n"
        "You must output only the adjacent differences.\n"
        "For each adjacent pair, compute NEXT minus CURRENT.\n"
        "Because the list is ascending, every output must be non-negative.\n"
        "Output only the final gap integers, never copied operands and never arithmetic expressions.\n"
        "Do not repeat the sorted list.\n"
        "Do not output equations.\n"
        "Do not output explanations.\n"
        "If any item is negative or any operand is copied, the answer is invalid.\n"
        f"There must be exactly {num_diffs} integers.\n"
        f"Return exactly this shell: <diffs>{template}</diffs>\n\n"
        "Example 1:\n"
        "Sorted: [-93, -41, -18, 18, 63, 71]\n"
        "Output: <diffs>[52, 23, 36, 45, 8]</diffs>\n\n"
        "Example 2:\n"
        "Sorted: [-55, 20, 43, 51, 66, 77]\n"
        "Output: <diffs>[75, 23, 8, 15, 11]</diffs>\n\n"
        "Example 3:\n"
        "Sorted: [-63, 29, 40]\n"
        "Output: <diffs>[92, 11]</diffs>\n\n"
        "Wrong output 1: <diffs>[-55, 20, 43, 51, 66, 77]</diffs>\n"
        "Wrong output 2: <diffs>[-52, 23, 49, 81, 43]</diffs>\n"
        "Wrong output 3: <diffs>[-63, 29, 40]</diffs>\n\n"
        "Wrong output 4: <diffs>[-97, 10, 1, 13, 15, 5, 43, 3, 3, 21]</diffs>\n"
        "This is wrong because it mixes copied operands with gap values.\n\n"
        "Wrong output 5: <diffs>[-86, 47, -23, 65, 9, 46, 49]</diffs>\n"
        "This is wrong because negative values show the subtraction direction or copied operands is wrong.\n\n"
        f"Input: {text2annotate}\n"
        f"Sorted: {sorted_nums}\n"
        "Output: <diffs>"
    )

def _build_task1_diffs_verify_prompt(text2annotate: str, sorted_nums: list[int], diffs_raw: str) -> str:
    expected_count = len(sorted_nums) - 1
    return (
        "You are a careful mathematical assistant.\n"
        "Audit the proposed adjacent-differences list.\n"
        "Check all of the following:\n"
        "1. It contains exactly the adjacent differences for the given sorted list.\n"
        f"2. It contains exactly {expected_count} integers.\n"
        "3. Every integer is non-negative.\n"
        "4. It is not the sorted list copied again.\n"
        "5. It does not contain copied operands or equation fragments.\n"
        "If all checks pass, output exactly <verify>pass</verify>.\n"
        "Otherwise output exactly <verify>fail</verify>.\n"
        "Do not output explanations.\n\n"
        "Example 1:\n"
        "Sorted: [-63, 29, 40]\n"
        "Proposed: <diffs>[92, 11]</diffs>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 2:\n"
        "Sorted: [-63, 29, 40]\n"
        "Proposed: <diffs>[92, 15]</diffs>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 3:\n"
        "Sorted: [-84, -62, -59, -52, -31, 78, 79]\n"
        "Proposed: <diffs>[-84, -62, -59, -52, -31, 78, 79]</diffs>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 4:\n"
        "Sorted: [-88, -59, -3, 62, 70, 75]\n"
        "Proposed: <diffs>[-59 - (-88) = 29, -3 - (-59) = 56, 62 - (-3) = 65, 70 - 62 = 8, 75 - 70 = 5]</diffs>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 5:\n"
        "Sorted: [-98, -88, -49, -28, 5, 9, 23, 10]\n"
        "Proposed: <diffs>[-10, -39, 21, 33, 4, 14]</diffs>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 6:\n"
        "Sorted: [-45, -4, 8, 45, 92]\n"
        "Proposed: <diffs>[-41, 44, 37, 47]</diffs>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 7:\n"
        "Sorted: [-45, -4, 8, 45, 92]</sorted>\n"
        "Proposed: <diffs>[41, 12, 37, 47]</diffs>\n"
        "Output: <verify>pass</verify>\n\n"
        f"Input: {text2annotate}\n"
        f"Sorted: {sorted_nums}\n"
        f"Proposed: {diffs_raw}\n"
        "Output: <verify>"
    )

def _is_valid_task1_answer_output(text: str, sorted_nums: list[int]) -> bool:
    diffs = _parse_tagged_int_list(text, "diffs")
    if diffs is None:
        return False
    if len(diffs) != len(sorted_nums) - 1:
        return False
    if any(x < 0 for x in diffs):
        return False
    return True

def _build_task1_pairmin_prompt(left: int, right: int) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Task: return the smaller of two non-negative integers.\n"
        "Output only one integer in the format <label>INTEGER</label>.\n"
        "The answer must be exactly one of the two input integers.\n"
        "Rule:\n"
        "1. If A <= B, output A.\n"
        "2. If B < A, output B.\n"
        "3. The returned integer must be less than or equal to both A and B.\n\n"
        "Examples:\n"
        "A=22, B=3\n"
        "Output: <label>3</label>\n\n"
        "A=109, B=1\n"
        "Output: <label>1</label>\n\n"
        "A=22, B=4\n"
        "Output: <label>4</label>\n\n"
        "A=3, B=1\n"
        "Output: <label>1</label>\n\n"
        "A=7, B=7\n"
        "Output: <label>7</label>\n\n"
        f"A={left}, B={right}\n"
        "Output: <label>"
    )

def _build_task1_pairmin_retry_prompt(left: int, right: int, previous_output: str | None = None) -> str:
    feedback = ""
    if previous_output:
        feedback = (
            "The previous response was invalid.\n"
            "You may have returned the larger number by mistake.\n"
        )
    return (
        "You are a careful mathematical assistant.\n"
        f"{feedback}"
        "Return the smaller of A and B.\n"
        "Decision rule:\n"
        "1. If A <= B, output A.\n"
        "2. If B < A, output B.\n"
        "3. Never output the larger number.\n"
        "4. Output only <label>INTEGER</label>.\n\n"
        "Examples:\n"
        "A=22, B=4 -> <label>4</label>\n"
        "A=3, B=1 -> <label>1</label>\n"
        "A=7, B=7 -> <label>7</label>\n\n"
        f"A={left}, B={right}\n"
        "Output: <label>"
    )

def _build_task1_pairmin_repair_prompt(left: int, right: int, smaller: int) -> str:
    del smaller
    return (
        "You are a careful mathematical assistant.\n"
        "The previous answer chose the wrong number.\n"
        "Return the smaller of A and B.\n"
        "Re-evaluate from scratch.\n"
        "Output only that number in the format <label>NUMBER</label>.\n\n"
        f"A={left}, B={right}\n"
        "Output: <label>"
    )

def _build_task1_pairmin_verify_prompt(left: int, right: int, answer_raw: str) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Audit whether the proposed answer is the smaller of A and B.\n"
        "If the proposed answer equals the smaller value, output exactly <verify>pass</verify>.\n"
        "Otherwise output exactly <verify>fail</verify>.\n"
        "Do not output explanations.\n\n"
        "Example 1:\n"
        "A=8, B=31\n"
        "Proposed: <label>8</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 2:\n"
        "A=8, B=31\n"
        "Proposed: <label>31</label>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 3:\n"
        "A=7, B=7\n"
        "Proposed: <label>7</label>\n"
        "Output: <verify>pass</verify>\n\n"
        f"A={left}, B={right}\n"
        f"Proposed: {answer_raw}\n"
        "Output: <verify>"
    )

def _build_task1_final_verify_prompt(diffs: list[int], answer_raw: str) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Audit whether the proposed label is the minimum value in the gaps list.\n"
        "If correct, output exactly <verify>pass</verify>.\n"
        "If incorrect, output exactly <verify>fail</verify>.\n"
        "Do not output explanations.\n\n"
        "Example 1:\n"
        "Gaps: [29, 56, 65, 8, 15]\n"
        "Proposed: <label>8</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 2:\n"
        "Gaps: [47, 20, 29, 8, 31]\n"
        "Proposed: <label>31</label>\n"
        "Output: <verify>fail</verify>\n\n"
        f"Gaps: {diffs}\n"
        f"Proposed: {answer_raw}\n"
        "Output: <verify>"
    )

def _parse_task1_verify(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"<verify>\s*(pass|fail)\s*</verify>", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None

def _build_task1_gap_prompt(left: int, right: int) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Task: compute one adjacent gap from a sorted list.\n"
        "Because the list is sorted ascending, the gap is RIGHT minus LEFT.\n"
        "If LEFT is negative and RIGHT is positive, subtracting a negative becomes addition.\n"
        "Compute RIGHT - LEFT and output only the final non-negative integer in the format <label>INTEGER</label>.\n"
        "Do not output arithmetic expressions, words, or explanations.\n\n"
        "Examples:\n"
        "LEFT=-93, RIGHT=-41\n"
        "Output: <label>52</label>\n\n"
        "LEFT=-56, RIGHT=9\n"
        "Output: <label>65</label>\n\n"
        "LEFT=-83, RIGHT=6\n"
        "Output: <label>89</label>\n\n"
        "LEFT=63, RIGHT=71\n"
        "Output: <label>8</label>\n\n"
        f"LEFT={left}, RIGHT={right}\n"
        "Output: <label>"
    )

def _build_task1_gap_retry_prompt(left: int, right: int, previous_output: str | None = None) -> str:
    feedback = ""
    if previous_output:
        feedback = (
            "The previous response was invalid because it was not a single non-negative integer.\n"
        )
    return (
        "You are a careful mathematical assistant.\n"
        f"{feedback}"
        "Recompute the gap.\n"
        "Use GAP = RIGHT - LEFT.\n"
        "If LEFT is negative and RIGHT is positive, the gap is RIGHT + abs(LEFT).\n"
        "Return only <label>INTEGER</label>.\n\n"
        "Example:\n"
        "LEFT=-48, RIGHT=8\n"
        "Output: <label>56</label>\n\n"
        f"LEFT={left}, RIGHT={right}\n"
        "Output: <label>"
    )

def _build_task1_gap_repair_prompt(left: int, right: int, expected_gap: int) -> str:
    del expected_gap
    return (
        "You are a careful mathematical assistant.\n"
        "The previous gap was incorrect.\n"
        "For a sorted list, GAP = RIGHT - LEFT.\n"
        "When LEFT is negative and RIGHT is positive, do not subtract magnitudes in the wrong direction.\n"
        "Subtracting a negative increases the result.\n"
        "Recompute the gap carefully from scratch.\n"
        "Return only that number in the format <label>NUMBER</label>.\n\n"
        "Example:\n"
        "LEFT=-61, RIGHT=13 -> <label>74</label>\n\n"
        f"LEFT={left}, RIGHT={right}\n"
        "Output: <label>"
    )

def _build_task1_gap_verify_prompt(left: int, right: int, answer_raw: str) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Audit whether the proposed gap equals RIGHT - LEFT exactly.\n"
        "If correct, output exactly <verify>pass</verify>.\n"
        "If incorrect, output exactly <verify>fail</verify>.\n"
        "Do not output explanations.\n\n"
        "Example 1:\n"
        "LEFT=70, RIGHT=75\n"
        "Proposed: <label>5</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 2:\n"
        "LEFT=70, RIGHT=75\n"
        "Proposed: <label>15</label>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 3:\n"
        "LEFT=88, RIGHT=90\n"
        "Proposed: <label>2</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 4:\n"
        "LEFT=61, RIGHT=66\n"
        "Proposed: <label>15</label>\n"
        "Output: <verify>fail</verify>\n\n"
        "Example 5:\n"
        "LEFT=-12, RIGHT=12\n"
        "Proposed: <label>24</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 6:\n"
        "LEFT=77, RIGHT=90\n"
        "Proposed: <label>13</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 7:\n"
        "LEFT=62, RIGHT=79\n"
        "Proposed: <label>17</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 8:\n"
        "LEFT=-17, RIGHT=15\n"
        "Proposed: <label>32</label>\n"
        "Output: <verify>pass</verify>\n\n"
        "Example 9:\n"
        "LEFT=-5, RIGHT=71\n"
        "Proposed: <label>76</label>\n"
        "Output: <verify>pass</verify>\n\n"
        f"LEFT={left}, RIGHT={right}\n"
        f"Proposed: {answer_raw}\n"
        "Output: <verify>"
    )


def _task1_call_gap_with_retries(left: int, right: int, call_llm) -> tuple[int | None, str]:
    prompts = [
        _build_task1_gap_prompt(left, right),
        _build_task1_gap_retry_prompt(left, right, None),
        _build_task1_gap_repair_prompt(left, right, expected_gap=0),
    ]
    last_raw = None
    for idx, prompt in enumerate(prompts):
        gap_raw = call_llm(prompt, max_t=24, stop_token="</label>")
        last_raw = gap_raw
        gap_prediction = count_answer(gap_raw, task_id=1)
        if gap_prediction is None:
            if idx == 0:
                prompts[1] = _build_task1_gap_retry_prompt(left, right, gap_raw)
            continue
        verify_raw = call_llm(_build_task1_gap_verify_prompt(left, right, gap_raw), max_t=16, stop_token="</verify>")
        if _parse_task1_verify(verify_raw) == "pass":
            return int(gap_prediction), f"{gap_raw}\n{verify_raw}"
        if idx == 0:
            prompts[1] = _build_task1_gap_retry_prompt(left, right, gap_raw)
        last_raw = f"{gap_raw}\n{verify_raw}"
    return None, last_raw


def _task1_call_pairmin_with_retries(left: int, right: int, call_llm) -> tuple[int | None, str]:
    prompts = [
        _build_task1_pairmin_prompt(left, right),
        _build_task1_pairmin_retry_prompt(left, right, None),
        _build_task1_pairmin_repair_prompt(left, right, left),
    ]
    last_raw = None
    for idx, prompt in enumerate(prompts):
        pair_raw = call_llm(prompt, max_t=20, stop_token="</label>")
        last_raw = pair_raw
        pair_prediction = count_answer(pair_raw, task_id=1)
        if pair_prediction is None or int(pair_prediction) not in (left, right):
            continue
        verify_prompt = _build_task1_pairmin_verify_prompt(left, right, pair_raw)
        verify_raw = call_llm(verify_prompt, max_t=16, stop_token="</verify>")
        if _parse_task1_verify(verify_raw) == "pass":
            return int(pair_prediction), f"{pair_raw}\n{verify_raw}"
        if idx == 0:
            prompts[1] = _build_task1_pairmin_retry_prompt(left, right, pair_raw)
        elif idx == 1:
            prompts[2] = _build_task1_pairmin_repair_prompt(left, right, left)
        last_raw = f"{pair_raw}\n{verify_raw}"
    return None, last_raw

def _build_task1_retry_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    retry_context = ""
    if previous_output:
        retry_context = (
            "The previous response was invalid because it included reasoning, drifted from the input numbers, "
            "or did not end with a clean label.\n"
        )

    return (
        "You are a careful mathematical assistant.\n"
        "Recompute from scratch.\n"
        f"{retry_context}"
        "Checklist:\n"
        "1. Copy the input integers exactly.\n"
        "2. Sort them in ascending order.\n"
        "3. Take the minimum difference between adjacent sorted integers.\n"
        "4. If any integer repeats, answer 0.\n"
        "5. The final answer is never negative.\n"
        "6. If all sorted integers are distinct, the final answer must be a positive integer, not 0.\n"
        "7. Do not copy a raw input integer unless it is truly the minimum adjacent gap after sorting.\n"
        "8. Do not use copied operands, equation fragments, or unrelated example numbers.\n"
        "7. Do not output the word INTEGER.\n"
        "8. Return only the integer inside the label.\n\n"
        "Example:\n"
        "Input: [31, 77, 0]\n"
        "Output: <label>31</label>\n\n"
        "Example:\n"
        "Input: [-17, 29, 12, -70, 23, -14, -53, -84, -82, 41]\n"
        "Output: <label>2</label>\n\n"
        "Example:\n"
        "Input: [-33, 24, 65, 18]\n"
        "Output: <label>6</label>\n\n"
        f"Input: {text2annotate}\n"
        "Output: <label>"
    )

def count_answer(text: str, task_id: int | None = None):
    if not text: return None
    
    # 提取最后一个 <label> 标签中的内容
    matches = re.findall(r"<label>\s*(.*?)\s*</label>", text, flags=re.DOTALL)
    if matches:
        ans = matches[-1].strip()
        if task_id == 1:
            # 确保只返回纯数字
            num_match = re.fullmatch(r"\d+", ans)
            return num_match.group(0) if num_match else None
        return ans

    if task_id == 1:
        partial_label_match = re.fullmatch(r"\s*(\d+)\s*</label>\s*", text, flags=re.DOTALL)
        if partial_label_match:
            return partial_label_match.group(1)
    
    # Task 1 不再盲目回退到全文最后一个数字，避免把推理过程中的无关数字当答案
    if task_id == 1:
        tail_match = re.search(r"(?:answer|result)\s*[:=]?\s*(\d+)\s*$", text.strip(), flags=re.IGNORECASE)
        return tail_match.group(1) if tail_match else None
    return None

def annotate_nvidia(input_prompt: str, task_id: int | None = None, debug: bool = False, text2annotate: str | None = None):
    URL = "http://0.0.0.0:2026/v1/completions"

    def _call_llm(p: str, max_t=256, stop_token="</label>"):
        data = {
            "model": "../Qwen3-4B",
            "prompt": p,
            "max_tokens": max_t,
            "temperature": 0,
            "stop": [stop_token], 
        }
        resp = requests.post(URL, json=data, timeout=300)
        text = resp.json()["choices"][0]["text"]
        if stop_token == "</label>" and p.rstrip().endswith("<label>") and "<label>" not in text:
            return f"<label>{text}</label>"
        if stop_token == "</verify>" and p.rstrip().endswith("<verify>") and "<verify>" not in text:
            return f"<verify>{text}</verify>"
        if stop_token == "</sorted>" and p.rstrip().endswith("<sorted>") and "<sorted>" not in text:
            return f"<sorted>{text}</sorted>"
        if stop_token == "</diffs>" and p.rstrip().endswith("<diffs>") and "<diffs>" not in text:
            return f"<diffs>{text}</diffs>"
        return text + stop_token

    def _llm_sort_exact(nums: list[int]) -> tuple[list[int] | None, str]:
        if not nums:
            return [], "<sorted>[]</sorted>"

        subset_text = str(nums)
        sort_raw = _call_llm(_build_task1_sort_prompt(subset_text), max_t=128, stop_token="</sorted>")
        sorted_subset = _parse_tagged_int_list(sort_raw, "sorted")
        verify_raw = None
        attempts = 0
        while attempts < 3:
            if sorted_subset is not None:
                verify_raw = _call_llm(
                    _build_task1_sort_verify_prompt(subset_text, sort_raw),
                    max_t=16,
                    stop_token="</verify>",
                )
                if _parse_task1_verify(verify_raw) == "pass":
                    return sorted_subset, f"{sort_raw}\n{verify_raw}"
            if attempts < 2:
                retry_prompt = _build_task1_sort_retry_prompt(subset_text, nums, sort_raw)
            else:
                retry_prompt = _build_task1_sort_repair_prompt(subset_text, nums)
            sort_raw = _call_llm(retry_prompt, max_t=128, stop_token="</sorted>")
            sorted_subset = _parse_tagged_int_list(sort_raw, "sorted")
            attempts += 1

        return sorted_subset, f"{sort_raw}\n{verify_raw}" if verify_raw else sort_raw

    if task_id == 1:
        parsed_task_description, parsed_text2annotate = _task1_parse_shell(input_prompt)
        if text2annotate is None and parsed_text2annotate is not None:
            text2annotate = parsed_text2annotate

        if text2annotate is None:
            return (None, "Invalid task 1 input") if debug else None

        original_nums = _parse_task1_input_list(text2annotate)
        if original_nums is None:
            return (None, "Invalid task 1 input") if debug else None

        sort_raw = _call_llm(_build_task1_sort_prompt(text2annotate), max_t=128, stop_token="</sorted>")
        sorted_nums = _parse_tagged_int_list(sort_raw, "sorted")
        sort_verify_raw = None

        sort_attempts = 0
        while sort_attempts < 4:
            if sorted_nums is not None:
                sort_verify_raw = _call_llm(
                    _build_task1_sort_verify_prompt(text2annotate, sort_raw),
                    max_t=16,
                    stop_token="</verify>",
                )
                if _parse_task1_verify(sort_verify_raw) == "pass":
                    break
            if sort_attempts < 2:
                sort_retry_prompt = _build_task1_sort_retry_prompt(text2annotate, original_nums, sort_raw)
            elif sort_attempts == 2:
                sort_retry_prompt = _build_task1_sort_repair_prompt(text2annotate, original_nums)
            else:
                sort_retry_prompt = _build_task1_sort_backup_prompt(text2annotate, original_nums)
            sort_raw = _call_llm(sort_retry_prompt, max_t=128, stop_token="</sorted>")
            sorted_nums = _parse_tagged_int_list(sort_raw, "sorted")
            sort_attempts += 1

        if sorted_nums is None or _parse_task1_verify(sort_verify_raw) != "pass":
            whole_result = (
                f"# sort\n{sort_raw}"
                + (f"\n{sort_verify_raw}" if sort_verify_raw else "")
                + "\n<label>INVALID_SORT</label>"
            )
            fallback_prompt = _build_task1_retry_prompt(text2annotate, whole_result)
            fallback_raw = _call_llm(fallback_prompt, max_t=32, stop_token="</label>")
            whole_result = f"{whole_result}\n# fallback\n{fallback_raw}"
            prediction = count_answer(fallback_raw, task_id=task_id)
            return (prediction, whole_result) if debug else prediction

        if len(sorted_nums) < 2:
            prediction = "0"
            whole_result = f"# sort\n{sort_raw}\n# diffs\n<diffs>[]</diffs>\n# min\n<label>0</label>"
            return (prediction, whole_result) if debug else prediction

        gap_values: list[int] = []
        gap_trace_lines: list[str] = []
        gap_failed = False
        for idx in range(len(sorted_nums) - 1):
            left = sorted_nums[idx]
            right = sorted_nums[idx + 1]
            gap_value, gap_raw = _task1_call_gap_with_retries(left, right, _call_llm)
            gap_trace_lines.append(f"{left} -> {right}: {gap_raw}")
            if gap_value is None:
                gap_failed = True
                break
            gap_values.append(gap_value)

        if gap_failed:
            diffs_prompt = _build_task1_answer_prompt(text2annotate, sorted_nums)
            diffs_raw = _call_llm(diffs_prompt, max_t=96, stop_token="</diffs>")
            diffs = _parse_tagged_int_list(diffs_raw, "diffs")
            diffs_verify_raw = None
            if diffs is not None:
                diffs_verify_raw = _call_llm(
                    _build_task1_diffs_verify_prompt(text2annotate, sorted_nums, diffs_raw),
                    max_t=16,
                    stop_token="</verify>",
                )
            if diffs is None or _parse_task1_verify(diffs_verify_raw) != "pass":
                diffs_retry_prompt = _build_task1_answer_retry_prompt(text2annotate, sorted_nums, diffs_raw)
                diffs_raw = _call_llm(diffs_retry_prompt, max_t=96, stop_token="</diffs>")
                diffs = _parse_tagged_int_list(diffs_raw, "diffs")
                diffs_verify_raw = None
                if diffs is not None:
                    diffs_verify_raw = _call_llm(
                        _build_task1_diffs_verify_prompt(text2annotate, sorted_nums, diffs_raw),
                        max_t=16,
                        stop_token="</verify>",
                    )
            if diffs is None or _parse_task1_verify(diffs_verify_raw) != "pass":
                diffs_repair_prompt = _build_task1_answer_repair_prompt(text2annotate, sorted_nums)
                diffs_raw = _call_llm(diffs_repair_prompt, max_t=96, stop_token="</diffs>")
                diffs = _parse_tagged_int_list(diffs_raw, "diffs")
                diffs_verify_raw = None
                if diffs is not None:
                    diffs_verify_raw = _call_llm(
                        _build_task1_diffs_verify_prompt(text2annotate, sorted_nums, diffs_raw),
                        max_t=16,
                        stop_token="</verify>",
                    )
            if diffs is not None and _parse_task1_verify(diffs_verify_raw) == "pass":
                gap_values = diffs
                gap_trace_lines.append(f"fallback_diffs: {diffs_raw}\n{diffs_verify_raw}")
                gap_failed = False

        if gap_failed or not gap_values:
            whole_result = (
                f"# sort\n{sort_raw}\n"
                f"# diffs\n" + "\n".join(gap_trace_lines) + "\n<label>INVALID_DIFFS</label>"
            )
            fallback_prompt = _build_task1_retry_prompt(text2annotate, whole_result)
            fallback_raw = _call_llm(fallback_prompt, max_t=32, stop_token="</label>")
            whole_result = f"{whole_result}\n# fallback\n{fallback_raw}"
            prediction = count_answer(fallback_raw, task_id=task_id)
            return (prediction, whole_result) if debug else prediction

        pair_trace_lines: list[str] = []
        current_candidates = gap_values[:]
        while len(current_candidates) > 1:
            next_round: list[int] = []
            for idx in range(0, len(current_candidates), 2):
                left = current_candidates[idx]
                if idx + 1 >= len(current_candidates):
                    next_round.append(left)
                    pair_trace_lines.append(f"carry: <label>{left}</label>")
                    continue
                right = current_candidates[idx + 1]
                pair_value, pair_raw = _task1_call_pairmin_with_retries(left, right, _call_llm)
                pair_trace_lines.append(f"{left} vs {right}: {pair_raw}")
                if pair_value is None:
                    break
                next_round.append(pair_value)
            else:
                current_candidates = next_round
                continue
            current_candidates = []
            break

        prediction = str(current_candidates[0]) if current_candidates else None
        min_raw = f"<label>{prediction}</label>" if prediction is not None else "<label>INVALID_MIN</label>"
        final_verify_raw = None
        if prediction is not None:
            final_verify_raw = _call_llm(_build_task1_final_verify_prompt(gap_values, min_raw), max_t=16, stop_token="</verify>")
            if _parse_task1_verify(final_verify_raw) != "pass":
                prediction = None
                min_raw = "<label>INVALID_MIN</label>"
        whole_result = (
            f"# sort\n{sort_raw}\n"
            f"# diffs\n<diffs>{gap_values}</diffs>\n"
            + ("\n".join(gap_trace_lines) if gap_trace_lines else "")
            + "\n# min\n"
            + ("\n".join(pair_trace_lines) + "\n" if pair_trace_lines else "")
            + min_raw
            + (f"\n{final_verify_raw}" if final_verify_raw else "")
        )

        if prediction is None:
            fallback_prompt = _build_task1_retry_prompt(text2annotate, whole_result)
            fallback_raw = _call_llm(fallback_prompt, max_t=32, stop_token="</label>")
            whole_result = f"{whole_result}\n# fallback\n{fallback_raw}"
            prediction = count_answer(fallback_raw, task_id=task_id)

        return (prediction, whole_result) if debug else prediction

    whole_result = _call_llm(input_prompt, max_t=32 if task_id == 1 else 256)
    prediction = count_answer(whole_result, task_id=task_id)
    return (prediction, whole_result) if debug else prediction


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--max_input_length", type=int, default=30_500)
    parser.add_argument("--log_path_prefix", type=str, default="./outputs/")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen3-4B")
    parser.add_argument("--debug_first_n", type=int, default=0)
    parser.add_argument(
        "--trace_line_numbers",
        type=str,
        default=DEFAULT_TRACE_LINE_NUMBERS,
        help="Comma-separated 1-based line numbers to dump full reasoning into a sidecar jsonl file. Disabled by default.",
    )
    parser.add_argument(
        "--only_line_numbers",
        type=str,
        default=DEFAULT_ONLY_LINE_NUMBERS,
        help="Comma-separated 1-based line numbers to run exclusively.",
    )
    return parser.parse_args()


def _parse_line_number_set(raw: str) -> set[int]:
    result = set()
    if not raw:
        return result
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value > 0:
            result.add(value)
    return result


def evaluate(
    task_id: int,
    qwen_tokenizer: AutoTokenizer,
    max_input_length: int,
    log_path_prefix: str,
    debug_first_n: int,
    trace_line_numbers: set[int],
    only_line_numbers: set[int],
):
    if task_id != 1:
        raise ValueError(f"main_task1.py only supports task_id=1, got {task_id}")

    with open(TASK1_FILE, "r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    icl_examples = task_dict.get("examples", [])[:100]
    test_samples = task_dict["test_samples"]

    os.makedirs(log_path_prefix, exist_ok=True)
    version = 1
    output_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}.jsonl")
    while os.path.exists(output_file):
        version += 1
        output_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}.jsonl")

    effective_trace_line_numbers = set(trace_line_numbers)
    trace_file = None
    if effective_trace_line_numbers:
        trace_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}-trace.jsonl")

    print(f"Starting Task {task_id}: {task_name}")
    print(f"Output will be saved to: {output_file}")
    if trace_file:
        print(f"Trace output will be saved to: {trace_file}")
    else:
        print("Trace output is disabled. Pass --trace_line_numbers to enable sidecar trace output.")
    if only_line_numbers:
        print(f"Only running line numbers: {sorted(only_line_numbers)}")
    else:
        print("Running all line numbers.")

    for idx, test_sample in enumerate(tqdm(test_samples)):
        line_number = idx + 1
        if only_line_numbers and line_number not in only_line_numbers:
            continue

        test_record = {"test_sample_id": test_sample["id"]}
        text2annotate = test_sample["input"]
        should_trace = line_number in effective_trace_line_numbers

        prompt = build_prompt(task_id, task_description, text2annotate)
        examples_str = select_examples(icl_examples, task_description, text2annotate)
        input_prompt = prompt.replace("[[EXAMPLES]]", examples_str)

        tokenized = qwen_tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False)
        if tokenized["input_ids"].shape[1] > max_input_length:
            test_record["prediction"] = None
            raw_output = "INPUT_TOO_LONG" if should_trace else None
        else:
            if should_trace or (debug_first_n and idx < debug_first_n) or line_number in only_line_numbers:
                prediction, raw_output = annotate_nvidia(
                    input_prompt,
                    task_id=task_id,
                    debug=True,
                    text2annotate=text2annotate,
                )
                test_record["prediction"] = prediction
                if (debug_first_n and idx < debug_first_n) or line_number in only_line_numbers:
                    print(f"\n[DEBUG {line_number}] Input: {text2annotate}")
                    print(f"[RAW]: {raw_output}")
                    print(f"[PRED]: {prediction}\n" + "-" * 50)
            else:
                raw_output = None
                test_record["prediction"] = annotate_nvidia(
                    input_prompt,
                    task_id=task_id,
                    debug=False,
                    text2annotate=text2annotate,
                )

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_record) + "\n")

        if should_trace and trace_file:
            trace_record = {
                "line_number": line_number,
                "test_sample_id": test_sample["id"],
                "input": text2annotate,
                "prediction": test_record["prediction"],
                "raw_output": raw_output,
            }
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    evaluate(
        args.task_id,
        qwen_tokenizer,
        args.max_input_length,
        args.log_path_prefix,
        args.debug_first_n,
        _parse_line_number_set(args.trace_line_numbers),
        _parse_line_number_set(args.only_line_numbers),
    )
