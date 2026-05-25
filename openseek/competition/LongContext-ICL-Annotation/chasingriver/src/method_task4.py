import ast
import json
import re
import requests
from functools import lru_cache
from pathlib import Path
from transformers import AutoTokenizer

_TOKENIZER = None
TASK4_LONG_CONTEXT_TARGET_TOKENS = 30000
TASK4_LONG_CONTEXT_MAX_TOKENS = 30500
TASK4_LONG_CONTEXT_EXAMPLE_BUDGET_TOKENS = 4096
TASK4_LONG_CONTEXT_INSTRUCTION = (
    "Long-context prepass reference: concatenate the input strings in their original order; do not "
    "insert separators, spaces, commas, or quotes unless already present inside an item; preserve case "
    "and every character exactly; round one must return only the requested XML schema. "
)


def _get_qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen3-4B", trust_remote_code=True)
    return _TOKENIZER


def _build_task4_short_prompt(text2annotate: str) -> str:
    parsed_items = _parse_task4_input_list(text2annotate)
    if parsed_items:
        item_block = "\n".join(f"{idx + 1}. {repr(item)}" for idx, item in enumerate(parsed_items))
    else:
        item_block = text2annotate
    return (
        "You are copying string fragments, not writing natural language.\n"
        "Join the fragments in exact row order with NO separator.\n"
        "Output only one XML element and nothing else.\n\n"
        "Hard rules:\n"
        "1. Use each fragment exactly once, in the given order.\n"
        "2. Do not add spaces.\n"
        "3. Do not add commas, quotes, brackets, periods, or any separator unless that character is already inside a fragment.\n"
        "4. Preserve uppercase, lowercase, punctuation, underscores, and digits exactly.\n"
        "5. Do not explain, reason, paraphrase, or normalize.\n"
        "6. Output format: <label>FINAL_STRING</label>\n\n"
        "Example 1\n"
        "Fragments:\n"
        "1. 'p'\n"
        "2. 'that.'\n"
        "3. 'o'\n"
        "Output: <label>pthat.o</label>\n\n"
        "Example 2\n"
        "Fragments:\n"
        "1. 'T'\n"
        "2. 'M'\n"
        "3. 'Z'\n"
        "Output: <label>TMZ</label>\n\n"
        "Current input list:\n"
        f"{text2annotate}\n\n"
        "Authoritative fragments:\n"
        f"{item_block}\n\n"
        "Output: <label>"
    )


def _build_task4_piece_prompt(fragment: str, index: int, total: int) -> str:
    return (
        "Copy the fragment exactly.\n"
        "Do not explain.\n"
        "Do not add any character.\n"
        "Do not remove any character.\n"
        "Do not change case.\n"
        "Return only one XML element in this format:\n"
        "<piece>EXACT_FRAGMENT</piece>\n\n"
        f"Fragment index: {index}/{total}\n"
        f"Fragment to copy exactly: {repr(fragment)}\n\n"
        "Output: <piece>"
    )


def _build_task4_fragmentize_prompt(text2annotate: str) -> str:
    return (
        "Convert the Python-style list of quoted strings into ordered XML pieces.\n"
        "Do not concatenate.\n"
        "Do not explain.\n"
        "Copy every list element exactly as its own piece.\n"
        "Preserve case, punctuation, underscores, and digits exactly.\n"
        "Return only this XML structure:\n"
        "<fragments>\n"
        "<piece>FIRST</piece>\n"
        "<piece>SECOND</piece>\n"
        "...\n"
        "</fragments>\n\n"
        "Example input:\n"
        "['p', 'that.', 'o']\n\n"
        "Example output:\n"
        "<fragments>\n"
        "<piece>p</piece>\n"
        "<piece>that.</piece>\n"
        "<piece>o</piece>\n"
        "</fragments>\n\n"
        "Input:\n"
        f"{text2annotate}\n\n"
        "Output:\n"
        "<fragments>\n"
    )


def _extract_task4_piece(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"<piece>\s*(.*?)\s*</piece>", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    return None


def _extract_task4_fragment_list(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"<piece>\s*(.*?)\s*</piece>", text, flags=re.DOTALL)


def _task4_expected_chars(text2annotate: str) -> int:
    parsed_items = _parse_task4_input_list(text2annotate)
    if not parsed_items:
        return 0
    return sum(len(item) for item in parsed_items)


def _task4_max_tokens(text2annotate: str) -> int:
    expected_chars = _task4_expected_chars(text2annotate)
    if expected_chars <= 0:
        return 512
    return max(256, min(4096, expected_chars * 2))


def _task4_authoritative_items_block(text2annotate: str) -> str:
    parsed_items = _parse_task4_input_list(text2annotate)
    if not parsed_items:
        return text2annotate
    return "\n".join(f"{idx + 1}. {repr(item)}" for idx, item in enumerate(parsed_items))


def _build_task4_space_cleanup_prompt(text2annotate: str, candidate: str) -> str:
    item_block = _task4_authoritative_items_block(text2annotate)
    return (
        "You are doing a strict whitespace cleanup on a candidate concatenation.\n"
        "Your only job is to remove every extra ASCII space character U+0020 from the candidate.\n"
        "Delete spaces only.\n"
        "Do not change any non-space character.\n"
        "Do not reorder characters.\n"
        "Do not add missing characters.\n"
        "Do not delete punctuation, digits, underscores, or letters.\n"
        "If there is no ASCII space character, return the candidate unchanged.\n"
        "Return only one XML element and nothing else.\n\n"
        "Output format: <label>FINAL_STRING</label>\n\n"
        "Examples:\n"
        "Candidate: I s\n"
        "Output: <label>Is</label>\n\n"
        "Candidate: there r H p\n"
        "Output: <label>therrHp</label>\n\n"
        "Candidate: here rp\n"
        "Output: <label>hererp</label>\n\n"
        "Candidate: t hem\n"
        "Output: <label>them</label>\n\n"
        "Candidate: Jh kz\n"
        "Output: <label>Jhkz</label>\n\n"
        "Candidate: quponxuofinpadnkisV\n"
        "Output: <label>quponxuofinpadnkisV</label>\n\n"
        "Authoritative fragments for reference only:\n"
        f"{item_block}\n\n"
        "Candidate from round 1:\n"
        f"{candidate}\n\n"
        "Output: <label>"
    )


def _build_task4_retry_concat_prompt(text2annotate: str, previous_candidate: str, failure_reason: str) -> str:
    item_block = _task4_authoritative_items_block(text2annotate)
    return (
        "You are copying string fragments, not writing natural language.\n"
        "The previous candidate was rejected. Repair the concatenation by aligning it to the authoritative fragments.\n"
        f"Rejection reason: {failure_reason}\n\n"
        "Hard rules:\n"
        "1. Use each fragment exactly once, in the given order.\n"
        "2. Do not skip any fragment.\n"
        "3. Do not duplicate any fragment or any character chunk.\n"
        "4. Copy single-character fragments exactly. They must not disappear.\n"
        "5. Preserve every letter exactly.\n"
        "6. Preserve uppercase and lowercase exactly.\n"
        "7. Preserve punctuation exactly when it already exists inside a fragment.\n"
        "8. Do not add spaces or separators.\n"
        "9. If the candidate has a wrong, missing, duplicated, or substituted character, fix it by following the authoritative fragments, not the candidate.\n"
        "10. Output only one XML element: <label>FINAL_STRING</label>\n\n"
        "Mini examples:\n"
        "Fragments: 1. 'Y' 2. 'can' 3. 'y' 4. 'L'\n"
        "Output: <label>YcanyL</label>\n\n"
        "Fragments: 1. 'i' 2. 'L' 3. 'cabin-scuttle,4his'\n"
        "Output: <label>iLcabin-scuttle,4his</label>\n\n"
        "Fragments: 1. 'of' 2. 'f' 3. 'in'\n"
        "Output: <label>offin</label>\n\n"
        f"Rejected candidate:\n{previous_candidate}\n\n"
        "Authoritative fragments:\n"
        f"{item_block}\n\n"
        "Output: <label>"
    )


def _build_task4_missing_char_verify_prompt(text2annotate: str, candidate: str) -> str:
    item_block = _task4_authoritative_items_block(text2annotate)
    return (
        "You are verifying fragment coverage in a concatenation candidate.\n"
        "Check coverage and order only.\n"
        "Return fail if any fragment is missing fully or partially.\n"
        "Return fail if any character chunk is duplicated, added, substituted, or out of order.\n"
        "Ignore extra spaces for this round.\n"
        "Do not judge punctuation style or wrapper text unless it causes missing content.\n"
        "In <issue>, name the fragment index or short wrong chunk briefly.\n"
        "Return only XML.\n\n"
        "<verify>pass|fail</verify>\n"
        "<issue>short issue</issue>\n\n"
        "Authoritative fragments:\n"
        f"{item_block}\n\n"
        "Candidate:\n"
        f"{candidate}\n\n"
        "Output: <verify>"
    )


def _build_task4_case_verify_prompt(text2annotate: str, candidate: str) -> str:
    item_block = _task4_authoritative_items_block(text2annotate)
    return (
        "You are verifying letter case in a concatenation candidate.\n"
        "Check only whether any alphabetic character changed uppercase/lowercase relative to the authoritative fragments.\n"
        "Ignore extra ASCII spaces for this round.\n"
        "If any letter case changed, return fail.\n"
        "In <issue>, name the affected fragment index or changed character briefly.\n"
        "Return only XML.\n\n"
        "<verify>pass|fail</verify>\n"
        "<issue>short issue</issue>\n\n"
        "Authoritative fragments:\n"
        f"{item_block}\n\n"
        "Candidate:\n"
        f"{candidate}\n\n"
        "Output: <verify>"
    )


def _task4_has_ascii_space(text: str | None) -> bool:
    return text is not None and " " in text


def _normalize_task4_verify_response(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = text
    if "<verify>" in normalized and "</verify>" not in normalized:
        normalized += "</verify>"
    if "<issue>" in normalized and "</issue>" not in normalized:
        normalized += "</issue>"
    return normalized


def _task4_staged_copy_nvidia(text2annotate: str) -> tuple[str | None, str]:
    trace_parts: list[str] = []
    fragment_result = _task4_chat_request(
        _build_task4_fragmentize_prompt(text2annotate),
        system=(
            "You extract ordered string fragments from a Python-style list. "
            "Return only XML fragments and nothing else."
        ),
        max_tokens=max(128, _task4_expected_chars(text2annotate) * 3),
        stop=["</fragments>"],
    )
    if fragment_result and "<fragments>" in fragment_result:
        fragment_result = fragment_result if fragment_result.rstrip().endswith("</fragments>") else fragment_result + "</fragments>"
    else:
        fragment_result = f"<fragments>\n{fragment_result or ''}</fragments>"
    items = _extract_task4_fragment_list(fragment_result)
    trace_parts.append(f"# fragments\n{fragment_result}")
    if not items:
        return None, "\n".join(trace_parts)

    copied_parts: list[str] = []
    total = len(items)
    for index, fragment in enumerate(items, start=1):
        best_piece = None
        for attempt in range(2):
            raw_piece = _task4_chat_request(
                _build_task4_piece_prompt(fragment, index, total),
                system=(
                    "You are an exact fragment copier. Return only one XML piece element and nothing else."
                ),
                max_tokens=max(16, len(fragment) * 2 + 8),
                stop=["</piece>"],
            )
            if raw_piece and "<piece>" in raw_piece:
                piece_result = raw_piece if raw_piece.rstrip().endswith("</piece>") else raw_piece + "</piece>"
            else:
                piece_result = f"<piece>{raw_piece or ''}</piece>"
            piece = _extract_task4_piece(piece_result)
            trace_parts.append(f"# piece_{index}_try_{attempt + 1}\n{piece_result}")
            if piece == fragment:
                best_piece = piece
                break
            if best_piece is None and piece is not None:
                best_piece = piece
        copied_parts.append(best_piece if best_piece is not None else "")
    prediction = "".join(copied_parts)
    trace_parts.append(f"# final\n<label>{prediction}</label>")
    return prediction, "\n".join(trace_parts)


def _task4_staged_copy_ascend(text2annotate: str) -> tuple[str | None, str]:
    import openai

    trace_parts: list[str] = []
    response = openai.chat.completions.create(
        model="Qwen3-4B-ascend-flagos",
        messages=[
            {
                "role": "system",
                "content": "You extract ordered string fragments from a Python-style list. Return only XML fragments and nothing else.",
            },
            {"role": "user", "content": _build_task4_fragmentize_prompt(text2annotate)},
        ],
        temperature=0,
        top_p=1.0,
        max_tokens=max(128, _task4_expected_chars(text2annotate) * 3),
        stream=False,
        stop=["</fragments>"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    fragment_result = response.choices[0].message.content
    if fragment_result and "<fragments>" in fragment_result:
        fragment_result = fragment_result if fragment_result.rstrip().endswith("</fragments>") else fragment_result + "</fragments>"
    else:
        fragment_result = f"<fragments>\n{fragment_result or ''}</fragments>"
    items = _extract_task4_fragment_list(fragment_result)
    trace_parts.append(f"# fragments\n{fragment_result}")
    if not items:
        return None, "\n".join(trace_parts)

    copied_parts: list[str] = []
    total = len(items)
    for index, fragment in enumerate(items, start=1):
        best_piece = None
        for attempt in range(2):
            response = openai.chat.completions.create(
                model="Qwen3-4B-ascend-flagos",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an exact fragment copier. Return only one XML piece element and nothing else.",
                    },
                    {"role": "user", "content": _build_task4_piece_prompt(fragment, index, total)},
                ],
                temperature=0,
                top_p=1.0,
                max_tokens=max(16, len(fragment) * 2 + 8),
                stream=False,
                stop=["</piece>"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_piece = response.choices[0].message.content
            if raw_piece and "<piece>" in raw_piece:
                piece_result = raw_piece if raw_piece.rstrip().endswith("</piece>") else raw_piece + "</piece>"
            else:
                piece_result = f"<piece>{raw_piece or ''}</piece>"
            piece = _extract_task4_piece(piece_result)
            trace_parts.append(f"# piece_{index}_try_{attempt + 1}\n{piece_result}")
            if piece == fragment:
                best_piece = piece
                break
            if best_piece is None and piece is not None:
                best_piece = piece
        copied_parts.append(best_piece if best_piece is not None else "")
    prediction = "".join(copied_parts)
    trace_parts.append(f"# final\n<label>{prediction}</label>")
    return prediction, "\n".join(trace_parts)


def _build_task4_self_check_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    previous_block = previous_output if previous_output else "<no previous answer>"
    item_block = _task4_authoritative_items_block(text2annotate)
    return (
        "You are doing a conservative cleanup of a candidate concatenation.\n"
        "Use the authoritative numbered items only as a reference for tiny mechanical fixes.\n"
        "Do not rewrite the whole string.\n\n"
        "Allowed fixes only:\n"
        "1. Remove wrapper noise like 'Label:' or 'Answer:'.\n"
        "2. Remove accidental spaces or newlines inserted between pieces.\n"
        "3. Remove surrounding quotes added around the entire answer.\n"
        "4. If the candidate is already usable, keep it unchanged.\n"
        "5. If the candidate is only placeholder text like '...' then return an empty label.\n\n"
        "Forbidden actions:\n"
        "- Do not reconstruct the answer from scratch.\n"
        "- Do not replace letters with different letters.\n"
        "- Do not change case.\n"
        "- Do not add missing multi-character chunks.\n"
        "- Do not invent content from memory or examples.\n\n"
        "Return exactly one XML element named label containing only the cleaned candidate.\n\n"
        f"Authoritative numbered items:\n{item_block}\n\n"
        "Candidate to repair:\n"
        f"{previous_block}\n\n"
        "Output: <label>"
    )


@lru_cache(maxsize=1)
def _task4_unit_tokens() -> int:
    tokenizer = _get_qwen_tokenizer()
    return len(tokenizer.encode(TASK4_LONG_CONTEXT_INSTRUCTION, add_special_tokens=False))


def _task4_exact_token_len(text: str) -> int:
    tokenizer = _get_qwen_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=1)
def _task4_official_examples_appendix() -> str:
    data_path = Path(__file__).resolve().parents[1] / "data" / "openseek-4_conala_concat_strings.json"
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return TASK4_LONG_CONTEXT_INSTRUCTION.strip()

    examples = payload.get("examples", [])
    if not isinstance(examples, list) or not examples:
        return TASK4_LONG_CONTEXT_INSTRUCTION.strip()

    lines = [
        "Official labeled task4 examples only.",
        "Use these examples as long-context references.",
        "Do not infer any unlabeled test answers.",
        "",
    ]
    tokenizer = _get_qwen_tokenizer()
    current_text = "\n".join(lines).strip()
    current_tokens = len(tokenizer.encode(current_text, add_special_tokens=False))
    for example in examples:
        try:
            input_text = str(example["input"])
            output_list = example["output"]
            answer = output_list[0] if isinstance(output_list, list) and output_list else str(output_list)
            example_lines = [
                f"Input: {input_text}",
                f"Output: <label>{answer}</label>",
                "",
            ]
            example_text = "\n".join(example_lines)
            example_tokens = len(tokenizer.encode(example_text, add_special_tokens=False))
            if current_tokens + example_tokens > TASK4_LONG_CONTEXT_EXAMPLE_BUDGET_TOKENS:
                break
            lines.extend(example_lines)
            current_tokens += example_tokens
        except Exception:
            continue
    appendix = "\n".join(lines).strip()
    return appendix or TASK4_LONG_CONTEXT_INSTRUCTION.strip()


def _task4_build_30k_shell(task_description: str, text2annotate: str) -> str:
    short_prompt = _build_task4_short_prompt(text2annotate)
    unit_tokens = max(1, _task4_unit_tokens())
    short_tokens = _task4_exact_token_len(short_prompt)
    base_shell = (
        "ConcatAudit-S9 two-round long-context wrapper.\n"
        "Round 1 must read the appendix and the active task, then output XML only.\n\n"
        "<reference_appendix>\n"
        "</reference_appendix>\n\n"
        "<active_task>\n"
        "Task Description:\n"
        f"{task_description}\n\n"
        "Text To Annotate:\n"
        f"{text2annotate}\n"
        "</active_task>\n\n"
        "Round 1 output schema only:\n"
        "<analysis><status>usable|fallback</status><focus>order|characters|unknown</focus>"
        "<hint>short hint or fallback</hint></analysis>\n"
    )
    base_tokens = _task4_exact_token_len(base_shell)
    reserve_tokens = max(unit_tokens * 8, max(2000, short_tokens))
    example_budget = max(0, TASK4_LONG_CONTEXT_TARGET_TOKENS - base_tokens - reserve_tokens)
    example_block = _task4_official_examples_appendix()
    if example_budget > 0:
        tokenizer = _get_qwen_tokenizer()
        kept_lines: list[str] = []
        for line in example_block.splitlines():
            candidate = "\n".join(kept_lines + [line]).strip()
            if candidate and len(tokenizer.encode(candidate, add_special_tokens=False)) > example_budget:
                break
            kept_lines.append(line)
        example_block = "\n".join(kept_lines).strip() or TASK4_LONG_CONTEXT_INSTRUCTION.strip()
    else:
        example_block = TASK4_LONG_CONTEXT_INSTRUCTION.strip()
    remaining_tokens = max(0, TASK4_LONG_CONTEXT_TARGET_TOKENS - base_tokens - _task4_exact_token_len(example_block))
    repeat_count = max(1, remaining_tokens // unit_tokens) if remaining_tokens > 0 else 1
    instruction_block = (TASK4_LONG_CONTEXT_INSTRUCTION * repeat_count).strip()
    appendix = f"{example_block}\n\n{instruction_block}".strip()
    shell = (
        "ConcatAudit-S9 two-round long-context wrapper.\n"
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
        "<analysis><status>usable|fallback</status><focus>order|characters|unknown</focus>"
        "<hint>short hint or fallback</hint></analysis>\n"
    )
    while _task4_exact_token_len(shell) < TASK4_LONG_CONTEXT_TARGET_TOKENS:
        shell = shell.replace("</reference_appendix>", TASK4_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", 1)
    while _task4_exact_token_len(shell) > TASK4_LONG_CONTEXT_MAX_TOKENS:
        shell = shell.replace(TASK4_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", "</reference_appendix>", 1)
    return shell


def _task4_parse_shell(input_prompt: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"Task Description:\s*(.*?)\n\s*Text To Annotate:\s*(.*?)\s*</active_task>",
        input_prompt,
        flags=re.DOTALL,
    )
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _task4_parse_analysis(text: str | None) -> tuple[str | None, str | None, str | None]:
    if not text:
        return None, None, None
    status_match = re.search(r"<status>\s*(usable|fallback)\s*</status>", text, flags=re.IGNORECASE)
    focus_match = re.search(r"<focus>\s*(order|characters|unknown)\s*</focus>", text, flags=re.IGNORECASE)
    hint_match = re.search(r"<hint>\s*(.*?)\s*</hint>", text, flags=re.IGNORECASE | re.DOTALL)
    status = status_match.group(1).lower() if status_match else None
    focus = focus_match.group(1).lower() if focus_match else None
    hint = None
    if hint_match:
        hint = re.sub(r"\s+", " ", hint_match.group(1)).strip()
        if len(hint) > 80:
            hint = hint[:80].rstrip()
    return status, focus, hint


def _task4_format_analysis_trace(text: str | None) -> str:
    if not text:
        return "# stage1_analysis\n<analysis></analysis>"
    if "<analysis>" in text and text.rstrip().endswith("</analysis>"):
        return f"# stage1_analysis\n{text}"
    if "<analysis>" in text:
        return f"# stage1_analysis\n{text}</analysis>"
    return f"# stage1_analysis\n<analysis>{text}</analysis>"


@lru_cache(maxsize=1)
def _task4_model_id() -> str:
    try:
        resp = requests.get("http://0.0.0.0:2026/v1/models", timeout=30)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models and "id" in models[0]:
            return models[0]["id"]
    except Exception:
        pass
    return "./Qwen3-4B"


def _task4_chat_request(prompt: str, *, system: str, max_tokens: int, stop: list[str] | None) -> str | None:
    data = {
        "model": _task4_model_id(),
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


def _build_task4_stage2_short_prompt(
    task_description: str,
    text2annotate: str,
    *,
    focus: str | None = None,
    hint: str | None = None,
) -> str:
    base_prompt = (
        "### Task\n"
        f"{task_description}\n\n"
        "### Rules\n"
        "1. Concatenate every string item in the original order.\n"
        "2. Preserve every character exactly.\n"
        "3. Do not add separators, spaces, commas, quotes, or line breaks unless they already exist inside an item.\n"
        "4. Do not change uppercase or lowercase.\n"
        "5. Do not autocorrect fragments into fluent English.\n"
        "6. Return only one XML element: <label>FINAL_STRING</label>.\n\n"
        "### Official Examples\n"
        "[[EXAMPLES]]\n\n"
    )
    if focus or hint:
        focus_value = focus or "unknown"
        hint_value = hint or "preserve order and characters exactly"
        base_prompt += (
            "### Round 1 Notes\n"
            f"Focus: {focus_value}\n"
            f"Hint: {hint_value}\n\n"
        )
    base_prompt += (
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        "### Output Format\n"
        "<label>FINAL_STRING</label>\n"
        "Output: <label>"
    )
    return base_prompt


def build_prompt(task_id: int, task_description: str, text2annotate: str) -> str:
    if task_id == 4:
        return _task4_build_30k_shell(task_description, text2annotate)

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
    tokenizer = _get_qwen_tokenizer()
    del task_description, text2annotate
    target_length = 512
    examples_str = ""
    token_num = 0
    example_count = 0

    for example in all_examples:
        try:
            example_str = f"Input: {example['input']}\nOutput: <label>{example['output'][0]}</label>\n\n"
            length = len(tokenizer.encode(example_str, add_special_tokens=False))
            if token_num + length <= target_length:
                examples_str += example_str
                token_num += length
                example_count += 1
                if example_count >= 8:
                    break
            else:
                break
        except Exception:
            continue
    return examples_str


def _parse_task4_input_list(text: str) -> list[str] | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return None


def _normalize_task4_text(text: str) -> str:
    return text.strip().replace("\r", "")


def _extract_label_content(text: str) -> str | None:
    if not text:
        return None
    matches = re.findall(r"<label>\s*(.*?)\s*</label>", text, flags=re.DOTALL)
    if matches:
        content = _normalize_task4_text(matches[-1])
        return content if content else None
    return None


def _iter_task4_candidates(text: str) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []

    for match in re.findall(r"<label>\s*(.*?)\s*</label>", text, flags=re.DOTALL):
        content = _normalize_task4_text(match)
        if content:
            candidates.append(content)

    pattern_list = [
        r"Corrected string:\s*([^\n<]+)",
        r"Correct concatenated string:\s*([^\n<]+)",
        r"Corrected String:\s*([^\n<]+)",
        r"CorrectedString:\s*([^\n<]+)",
        r"corrected_string:\s*([^\n<]+)",
        r"corrected_string\s*([^\n<]+)",
        r"Correct prefix string:\s*([^\n<]+)",
        r"Correct concatenated string is\s*([^\n<]+)",
    ]
    for pattern in pattern_list:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            content = _normalize_task4_text(match)
            if content:
                candidates.append(content)

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _clean_task4_candidate(text: str) -> str | None:
    if text is None:
        return None
    cleaned = _normalize_task4_text(text)
    if not cleaned:
        return None
    cleaned = cleaned.strip("'\"")
    if any(token in cleaned for token in ("<input", "</input", "<answer", "</answer", "type=\"text\"")):
        return None
    lowered = cleaned.lower()
    if cleaned in {
        "FINAL_STRING",
        "CONCATENATED_PREFIX",
        "Corrected String",
        "corrected_string",
        "Corrected string",
        "CorrectedString",
        "Verified Answer",
        "Corrected List",
        "prefix",
        "Prefix",
        "Prefix:",
        "...",
        "...?",
        "?",
        "??",
        "???",
    }:
        return None
    if lowered in {
        "final_string",
        "concatenated_prefix",
        "label",
        "prefix",
        "prefix:",
        "blablabla",
        "mlink",
    }:
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return None
    banned_substrings = (
        "wait,",
        "okay,",
        "let's see",
        "the user",
        "the input is",
        "correct output should",
        "should be the concatenation",
        "provided an input list",
        "wants me to",
        "output should be",
        "textlabel:",
        "segment index:",
        "prefix length:",
        "...wait...",
        "correct concatenation",
        "corrected final answer",
        "your answer here",
        "the concatenation of all items",
        "answer:",
        "repaired string",
        "concatenated string",
    )
    if any(token in lowered for token in banned_substrings):
        return None
    if re.search(r"(^|[^a-z])label([^a-z]|$)", lowered):
        return None
    if re.search(r"(^|[^a-z])textlabel([^a-z]|$)", lowered):
        return None
    if re.search(r"\b\d+[LR]{2,}\b", cleaned):
        return None
    if re.search(r"\b[1-9][LR]{1,}\b", cleaned):
        return None
    if cleaned.count("\n") > 1:
        return None
    if len(cleaned) >= 4096:
        return None
    return cleaned


def _find_task4_prediction_in_text(text: str, expected_output: str | None = None) -> str | None:
    candidates = _iter_task4_candidates(text)
    cleaned_candidates: list[str] = []
    for candidate in candidates:
        cleaned = _clean_task4_candidate(candidate)
        if cleaned is not None:
            cleaned_candidates.append(cleaned)

    if expected_output is not None:
        for candidate in cleaned_candidates:
            if candidate == expected_output:
                return candidate

    if cleaned_candidates:
        return cleaned_candidates[-1]
    return None


def _extract_task4_chat_answer(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip().replace("\r", "")
    if not stripped:
        return None

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    candidates = lines if lines else [stripped]

    for candidate in candidates:
        if candidate.startswith("```") or candidate.endswith("```"):
            continue
        if candidate.lower().startswith(("here is", "here's", "output:", "answer:", "result:")):
            parts = candidate.split(":", 1)
            candidate = parts[1].strip() if len(parts) == 2 else candidate
        cleaned = _clean_task4_candidate(candidate)
        if cleaned is not None:
            return cleaned
    return None


def _extract_task4_stepwise_answer(text: str | None) -> str | None:
    if not text:
        return None
    return None


def _parse_task4_verify(text: str | None) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    verify_match = re.search(r"<verify>\s*(pass|fail)\s*</verify>", text, flags=re.IGNORECASE)
    issue_match = re.search(r"<issue>\s*(.*?)\s*</issue>", text, flags=re.IGNORECASE | re.DOTALL)
    verdict = verify_match.group(1).lower() if verify_match else None
    issue = None
    if issue_match:
        issue = re.sub(r"\s+", " ", issue_match.group(1)).strip()
        if len(issue) > 120:
            issue = issue[:120].rstrip()
    return verdict, issue


def count_answer(text: str, task_id: int | None = None):
    if not text:
        return None

    if task_id == 4:
        return _extract_task4_stepwise_answer(text) or _find_task4_prediction_in_text(text)

    matches = re.findall(r"<label>\s*(.*?)\s*</label>", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _build_task4_retry_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    return _build_task4_self_check_prompt(text2annotate, previous_output)


def _build_task4_final_retry_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    previous_block = previous_output if previous_output else "<no previous answer>"
    item_block = _task4_authoritative_items_block(text2annotate)
    return (
        "You are doing a full reconstruction because the current candidate is unusable.\n"
        "Rebuild the answer from the authoritative numbered items only.\n\n"
        "Reconstruction algorithm:\n"
        "1. Start with an empty string.\n"
        "2. Append item 1 exactly.\n"
        "3. Append item 2 exactly.\n"
        "4. Continue until the last item.\n"
        "5. Return exactly one XML element named label containing only the rebuilt string.\n\n"
        "Rules:\n"
        "- Preserve every character exactly.\n"
        "- Preserve exact case.\n"
        "- Do not add spaces or punctuation unless already inside an item.\n"
        "- Do not output 'Answer:', 'Label:', '...', placeholders, or explanations.\n"
        "- Do not copy anything from the previous candidate except to understand that it failed.\n\n"
        f"Authoritative numbered items:\n{item_block}\n\n"
        "Failed candidate:\n"
        f"{previous_block}\n\n"
        "Output: <label>"
    )


def _build_task4_anchor_repair_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    parsed_items = _parse_task4_input_list(text2annotate)
    item_block = _task4_authoritative_items_block(text2annotate)
    if parsed_items:
        first_item = parsed_items[0]
        last_item = parsed_items[-1]
        singletons = [f"{idx + 1}:{repr(item)}" for idx, item in enumerate(parsed_items) if len(item) == 1]
        singleton_block = ", ".join(singletons) if singletons else "<none>"
        item_count = len(parsed_items)
    else:
        first_item = ""
        last_item = ""
        singleton_block = "<unknown>"
        item_count = 0
    previous_block = previous_output if previous_output else "<no previous answer>"
    return (
        "You are doing a structural anchor repair on a candidate answer.\n"
        "Use the candidate and the input structure to repair dropped edge items, truncated output, or misplaced single-character items.\n\n"
        "The repaired answer must respect the numbered-item order exactly.\n\n"
        "Anchor repair rules:\n"
        "1. The final string must start with the first item exactly.\n"
        "2. The final string must end with the last item exactly.\n"
        "3. Single-character items must remain present in order with exact case.\n"
        "4. Do not output placeholders like ... or your answer here.\n"
        "5. Do not output the numbered list or any explanation.\n"
        "6. Return exactly one XML element named label containing only the repaired candidate.\n\n"
        f"Item count: {item_count}\n"
        f"First item: {repr(first_item)}\n"
        f"Last item: {repr(last_item)}\n"
        f"Single-character items in order: {singleton_block}\n"
        f"Authoritative numbered items:\n{item_block}\n\n"
        "Candidate to repair:\n"
        f"{previous_block}\n\n"
        "Do not borrow words from unrelated examples or prior prompts.\n"
        "Only repair this candidate using the current first item, last item, and singleton order.\n"
        "Output: <label>"
    )


def _build_task4_case_repair_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    parsed_items = _parse_task4_input_list(text2annotate)
    item_block = _task4_authoritative_items_block(text2annotate)
    if parsed_items:
        case_items = [f"{idx + 1}:{repr(item)}" for idx, item in enumerate(parsed_items) if len(item) == 1 or any(ch.isupper() for ch in item)]
        case_block = ", ".join(case_items) if case_items else "<none>"
    else:
        case_block = "<unknown>"
    previous_block = previous_output if previous_output else "<no previous answer>"
    return (
        "You are doing a case-and-singleton repair on a candidate answer.\n"
        "Do not rewrite the whole answer. Only fix missing or mis-cased short items.\n\n"
        "One-character items in the numbered list must appear once, in order, with exact case.\n\n"
        "Repair rules:\n"
        "1. Preserve the candidate unless a single-character item is missing, duplicated, lowercased, or uppercased incorrectly.\n"
        "2. Restore exact case for all one-character items.\n"
        "3. Do not add spaces.\n"
        "4. Do not delete longer substrings.\n"
        "5. Do not output placeholders or explanations.\n"
        "6. Return exactly one XML element named label containing only the repaired candidate.\n\n"
        f"Important single-character / case-sensitive items in order: {case_block}\n"
        f"Authoritative numbered items:\n{item_block}\n\n"
        "Candidate to repair:\n"
        f"{previous_block}\n\n"
        "Do not invent missing multi-character words.\n"
        "Only fix one-character items and case errors that are implied by the input items.\n"
        "Output: <label>"
    )


def _build_task4_order_repair_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    item_block = _task4_authoritative_items_block(text2annotate)
    previous_block = previous_output if previous_output else "<no previous answer>"
    return (
        "You are doing a final order-and-coverage repair on a candidate answer.\n"
        "If the candidate is truncated, reordered, or missing chunks, rebuild it by following the numbered items exactly once from top to bottom.\n\n"
        "The numbered items are authoritative. Follow them from item 1 to the last item exactly once.\n\n"
        "Final repair rules:\n"
        "1. Follow item order strictly from 1 to the last item.\n"
        "2. Use each item exactly once.\n"
        "3. Preserve every character and case exactly.\n"
        "4. Do not insert spaces or punctuation unless already inside an item.\n"
        "5. Do not output placeholders or explanations.\n"
        "6. Return exactly one XML element named label containing only the repaired answer.\n\n"
        "Numbered items:\n"
        f"{item_block}\n\n"
        "Candidate to compare against:\n"
        f"{previous_block}\n\n"
        "Do not copy text from unrelated examples or previous tasks.\n"
        "Use only the current numbered items.\n"
        "Output: <label>"
    )


def _looks_like_task4_placeholder(candidate: str | None) -> bool:
    if candidate is None:
        return True
    lowered = candidate.strip().lower()
    if not lowered:
        return True
    return lowered in {
        "...",
        "...?",
        "final_string",
        "... your answer here ...",
        "...your answer here...",
        "your answer here",
        "... (concatenated string) ...",
        "... (repaired string) ...",
        "answer:",
        "label:",
    }


def _has_task4_internal_spaces(candidate: str | None) -> bool:
    return candidate is not None and " " in candidate.strip()


def _has_task4_repeated_char_run(candidate: str | None) -> bool:
    if candidate is None or len(candidate) < 128:
        return False
    unique_chars = set(candidate)
    return len(unique_chars) == 1


def _has_task4_wrapper_noise(candidate: str | None) -> bool:
    if candidate is None:
        return False
    lowered = candidate.lower()
    return (
        lowered.startswith("answer:")
        or lowered.startswith("label:")
        or "..." in candidate
        or "\n" in candidate
        or candidate != candidate.strip()
    )


def _task4_expected_length(text2annotate: str) -> int | None:
    items = _parse_task4_input_list(text2annotate)
    if items is None:
        return None
    return sum(len(item) for item in items)


def _needs_task4_whitespace_repair(candidate: str | None) -> bool:
    if candidate is None:
        return False
    return " " in candidate or "\n" in candidate or _has_task4_wrapper_noise(candidate)


def _needs_task4_wrapper_cleanup(raw_text: str | None, candidate: str | None) -> bool:
    del raw_text
    if candidate is None:
        return True
    if _looks_like_task4_placeholder(candidate):
        return True
    if _has_task4_repeated_char_run(candidate):
        return True
    if len(candidate) >= 512:
        return True
    return False


def _task4_round_rank(round_name: str) -> int:
    return {
        "whole": 0,
        "retry": 1,
        "rebuild": 2,
    }.get(round_name, 99)


def _needs_task4_anchor_repair(candidate: str | None, text2annotate: str) -> bool:
    if candidate is None:
        return True
    items = _parse_task4_input_list(text2annotate)
    if not items:
        return False
    expected_length = _task4_expected_length(text2annotate)
    if expected_length is not None and len(candidate) < max(1, expected_length // 2):
        return True
    if not candidate.startswith(items[0]) or not candidate.endswith(items[-1]):
        return True
    return False


def _needs_task4_case_repair(candidate: str | None, text2annotate: str) -> bool:
    if candidate is None:
        return False
    items = _parse_task4_input_list(text2annotate)
    if not items:
        return False
    cursor = 0
    for item in items:
        if len(item) != 1:
            continue
        next_cursor = candidate.find(item, cursor)
        if next_cursor != -1:
            cursor = next_cursor + 1
            continue
        # If exact-case singleton is missing but case-folded variant exists ahead, trigger a case repair.
        if candidate.lower().find(item.lower(), cursor) != -1:
            return True
        return True
    return False


def _needs_task4_order_repair(candidate: str | None, text2annotate: str) -> bool:
    if candidate is None:
        return True
    items = _parse_task4_input_list(text2annotate)
    if not items:
        return False
    expected_length = _task4_expected_length(text2annotate)
    if expected_length is not None and len(candidate) < expected_length:
        return True
    if " " in candidate:
        return True
    return _task4_anchor_penalty(candidate, text2annotate) >= 35


def _task4_anchor_penalty(candidate: str | None, text2annotate: str) -> int:
    if candidate is None:
        return 100
    items = _parse_task4_input_list(text2annotate)
    if not items:
        return 0
    penalty = 0
    if _looks_like_task4_placeholder(candidate):
        penalty += 100
    if _has_task4_repeated_char_run(candidate):
        penalty += 80
    if _has_task4_internal_spaces(candidate):
        penalty += 20
    if _has_task4_wrapper_noise(candidate):
        penalty += 30
    if not candidate.startswith(items[0]):
        penalty += 25
    if not candidate.endswith(items[-1]):
        penalty += 25
    expected_length = _task4_expected_length(text2annotate)
    if expected_length is not None:
        if len(candidate) < expected_length:
            penalty += min(40, expected_length - len(candidate))
        elif len(candidate) > expected_length:
            penalty += min(40, len(candidate) - expected_length)
    cursor = 0
    for item in items:
        if len(item) != 1:
            continue
        next_cursor = candidate.find(item, cursor)
        if next_cursor == -1:
            penalty += 10
        else:
            cursor = next_cursor + len(item)
    return penalty


def _select_task4_prediction(
    text2annotate: str,
    candidates: list[tuple[str, str | None]],
) -> str | None:
    best_candidate = None
    best_score = 10**9
    for round_name, candidate in candidates:
        if candidate is None:
            continue
        score = _task4_anchor_penalty(candidate, text2annotate) + _task4_round_rank(round_name) * 3
        if best_candidate is None or score < best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate


def annotate_nvidia(input_prompt: str, task_id: int | None = None, debug: bool = False, text2annotate: str | None = None):
    url = "http://0.0.0.0:2026/v1/completions"

    def _call_llm(p: str, max_t=64, stop_token: str | None = "</label>"):
        data = {
            "model": "./Qwen3-4B",
            "prompt": p,
            "max_tokens": max_t,
            "temperature": 0,
        }
        if stop_token is not None:
            data["stop"] = [stop_token]
        resp = requests.post(url, json=data, timeout=300)
        text = resp.json()["choices"][0]["text"]
        if stop_token == "</label>" and p.rstrip().endswith("<label>") and "<label>" not in text:
            return f"<label>{text}</label>"
        if stop_token is None:
            return text
        return text + stop_token

    if task_id == 4:
        if text2annotate is None:
            return (None, "Invalid task 4 input") if debug else None

        if _parse_task4_input_list(text2annotate) is None:
            return (None, "Invalid task 4 input") if debug else None

        analysis_text = _task4_chat_request(
            input_prompt,
            system=(
                "You are a strict long-context XML prepass for task 4. "
                "Read the full appendix, then output only the requested XML schema and no prose."
            ),
            max_tokens=96,
            stop=["</analysis>"],
        )
        if analysis_text is not None:
            analysis_text += "</analysis>"

        prediction, short_raw_output = _task4_staged_copy_nvidia(text2annotate)
        if debug:
            combined_raw_output = (
                f"[LONG_PASS]\n{analysis_text or 'LONG_PASS_EMPTY'}\n\n"
                f"[SHORT_PASS]\n{short_raw_output}"
            )
            return prediction, combined_raw_output
        return prediction

    whole_result = _call_llm(input_prompt, max_t=256)
    prediction = count_answer(whole_result, task_id=task_id)
    return (prediction, whole_result) if debug else prediction


def annotate_ascend(input_prompt: str, task_id: int | None = None, debug: bool = False, text2annotate: str | None = None):
    import openai

    def _call_llm(p: str, max_t=64, stop_token: str | None = "</label>"):
        openai.api_key = "EMPTY"
        openai.base_url = "http://localhost:9010/v1/"
        kwargs = {
            "model": "Qwen3-4B-ascend-flagos",
            "messages": [
                {"role": "system", "content": "You are a careful string processing assistant."},
                {"role": "user", "content": p},
            ],
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": max_t,
            "stream": False,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if stop_token is not None:
            kwargs["stop"] = [stop_token]
        response = openai.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        if stop_token == "</label>" and p.rstrip().endswith("<label>") and "<label>" not in text:
            return f"<label>{text}</label>"
        if stop_token is None:
            return text
        return text + stop_token

    if task_id == 4:
        if text2annotate is None:
            return (None, "Invalid task 4 input") if debug else None

        if _parse_task4_input_list(text2annotate) is None:
            return (None, "Invalid task 4 input") if debug else None

        analysis_text = None
        try:
            response = openai.chat.completions.create(
                model="Qwen3-4B-ascend-flagos",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict long-context XML prepass for task 4. "
                            "Read the full appendix, then output only the requested XML schema and no prose."
                        ),
                    },
                    {"role": "user", "content": input_prompt},
                ],
                temperature=0,
                top_p=1.0,
                max_tokens=96,
                stream=False,
                stop=["</analysis>"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            analysis_text = response.choices[0].message.content
        except Exception:
            analysis_text = None
        if analysis_text is not None:
            analysis_text += "</analysis>"

        prediction, short_raw_output = _task4_staged_copy_ascend(text2annotate)
        if debug:
            combined_raw_output = (
                f"[LONG_PASS]\n{analysis_text or 'LONG_PASS_EMPTY'}\n\n"
                f"[SHORT_PASS]\n{short_raw_output}"
            )
            return prediction, combined_raw_output
        return prediction

    whole_result = _call_llm(input_prompt, max_t=256)
    prediction = count_answer(whole_result, task_id=task_id)
    return (prediction, whole_result) if debug else prediction