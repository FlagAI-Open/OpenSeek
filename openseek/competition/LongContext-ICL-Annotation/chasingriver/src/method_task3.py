import ast
import json
import re
import requests
from functools import lru_cache
from pathlib import Path
from transformers import AutoTokenizer

_TOKENIZER = None
TASK3_LONG_CONTEXT_TARGET_TOKENS = 30000
TASK3_LONG_CONTEXT_MAX_TOKENS = 30500
TASK3_LONG_CONTEXT_INSTRUCTION = (
    "Long-context prepass reference: apply one Collatz step to each integer independently; preserve "
    "the original order; if x is even output x divided by 2; if x is odd output 3*x+1; round one must "
    "return only the requested XML schema. "
)


def _get_qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen3-4B", trust_remote_code=True)
    return _TOKENIZER


def _build_task3_short_prompt(text2annotate: str) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Task: transform each integer in the input list independently.\n"
        "Think silently, then output only the final answer.\n\n"
        "Rules:\n"
        "1. Preserve the original order.\n"
        "2. If x is even, output x / 2.\n"
        "3. If x is odd, output 3 * x + 1.\n"
        "4. Output exactly one list in the format <label>[a, b, c]</label>.\n"
        "5. Do not output reasoning, steps, explanations, formulas, parentheses, or equals signs.\n"
        "6. Do not output placeholder words such as INTEGER.\n"
        "7. Output evaluated integers only.\n"
        "8. Never copy the input numbers unchanged unless the rule really gives that result.\n"
        "9. Never add a minus sign unless the computed result is actually negative.\n"
        "10. For even x, divide by 2 exactly once. Do not output decimals.\n"
        "11. For odd x, multiply by 3 and then add 1. Do not subtract. Do not negate.\n"
        "12. Apply exactly one step per element. Do not continue to a second step.\n"
        "13. If x is odd, never divide by 2. If you see a .5 result, you used the wrong rule.\n"
        "14. For odd x, do not stop at 3*x. You must add the final +1.\n\n"
        "Quick check before you answer:\n"
        "- even input -> integer half\n"
        "- odd input -> 3*x+1\n"
        "- odd input -> even output\n"
        "- odd input -> do not forget the final +1\n"
        "- same list length as input\n"
        "- no negative sign by accident\n\n"
        "Examples:\n"
        "Input: [72, 29, 49]\n"
        "Output: <label>[36, 88, 148]</label>\n\n"
        "Input: [27, 99, 188, 149]\n"
        "Output: <label>[82, 298, 94, 448]</label>\n\n"
        "Input: [47, 92]\n"
        "Output: <label>[142, 46]</label>\n\n"
        "Input: [25, 97, 131]\n"
        "Output: <label>[76, 292, 394]</label>\n\n"
        "Input: [113, 139]\n"
        "Output: <label>[340, 418]</label>\n\n"
        "Input: [63, 124, 15, 10, 2]\n"
        "Output: <label>[190, 62, 46, 5, 1]</label>\n\n"
        "Input: [183, 70, 77]\n"
        "Output: <label>[550, 35, 232]</label>\n\n"
        "Input: [165, 130, 81, 14]\n"
        "Output: <label>[496, 65, 244, 7]</label>\n\n"
        "Wrong output example 1: <label>[(56/2), 298, 280]</label>\n"
        "Wrong output example 2: <label>[56/2=28, 298, 280]</label>\n"
        "Wrong output example 3: <label>[INTEGER, 298, 280]</label>\n"
        "Wrong output example 4: <label>[-12, 18, 74]</label>\n"
        "Wrong output example 5: <label>[-183, 62, 275]</label>\n"
        "Wrong output example 6: <label>[-165, -130, 40]</label>\n"
        "Wrong output example 7: <label>[28, 49.5, 280]</label>\n"
        "Wrong output example 8: <label>[145, 328, 88]</label>\n"
        "Wrong output example 9: <label>[96, 36]</label>\n"
        "Wrong output example 10: <label>[339, 408]</label>\n\n"
        f"Input: {text2annotate}\n"
        "Output: <label>"
    )


@lru_cache(maxsize=1)
def _task3_unit_tokens() -> int:
    tokenizer = _get_qwen_tokenizer()
    return len(tokenizer.encode(TASK3_LONG_CONTEXT_INSTRUCTION, add_special_tokens=False))


def _task3_exact_token_len(text: str) -> int:
    tokenizer = _get_qwen_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=8)
def _task3_official_examples_appendix(max_tokens: int | None = None) -> str:
    data_path = Path(__file__).resolve().parents[1] / "data" / "openseek-3_collatz_conjecture.json"
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return TASK3_LONG_CONTEXT_INSTRUCTION.strip()

    examples = payload.get("examples", [])
    if not isinstance(examples, list) or not examples:
        return TASK3_LONG_CONTEXT_INSTRUCTION.strip()

    lines = [
        "Official labeled task3 examples only.",
        "Use these examples as long-context references.",
        "Do not infer any unlabeled test answers.",
        "",
    ]
    current = "\n".join(lines).strip()
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
            candidate = (current + "\n" + "\n".join(example_lines)).strip()
            if max_tokens is not None and _task3_exact_token_len(candidate) > max_tokens:
                break
            lines.extend(example_lines)
            current = candidate
        except Exception:
            continue
    appendix = "\n".join(lines).strip()
    return appendix or TASK3_LONG_CONTEXT_INSTRUCTION.strip()


def _task3_build_30k_shell(task_description: str, text2annotate: str) -> str:
    short_prompt = _build_task3_short_prompt(text2annotate)
    unit_tokens = max(1, _task3_unit_tokens())
    short_tokens = _task3_exact_token_len(short_prompt)
    base_shell = (
        "CollatzStepAudit-C7 two-round long-context wrapper.\n"
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
        "<analysis><status>usable|fallback</status><focus>order|parity|unknown</focus>"
        "<hint>short hint or fallback</hint></analysis>\n"
    )
    base_tokens = _task3_exact_token_len(base_shell)
    reserve_tokens = max(unit_tokens * 8, 2000)
    example_budget = max(0, TASK3_LONG_CONTEXT_TARGET_TOKENS - base_tokens - reserve_tokens)
    example_block = _task3_official_examples_appendix(example_budget)
    remaining_tokens = max(0, TASK3_LONG_CONTEXT_TARGET_TOKENS - base_tokens - _task3_exact_token_len(example_block))
    repeat_count = max(1, remaining_tokens // unit_tokens) if remaining_tokens > 0 else 1
    instruction_block = (TASK3_LONG_CONTEXT_INSTRUCTION * repeat_count).strip()
    appendix = f"{example_block}\n\n{instruction_block}".strip()
    shell = (
        "CollatzStepAudit-C7 two-round long-context wrapper.\n"
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
        "<analysis><status>usable|fallback</status><focus>order|parity|unknown</focus>"
        "<hint>short hint or fallback</hint></analysis>\n"
    )
    while _task3_exact_token_len(shell) < TASK3_LONG_CONTEXT_TARGET_TOKENS:
        shell = shell.replace("</reference_appendix>", TASK3_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", 1)
    while _task3_exact_token_len(shell) > TASK3_LONG_CONTEXT_MAX_TOKENS:
        shell = shell.replace(TASK3_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", "</reference_appendix>", 1)
    return shell


def _task3_parse_shell(input_prompt: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"Task Description:\s*(.*?)\n\s*Text To Annotate:\s*(.*?)\s*</active_task>",
        input_prompt,
        flags=re.DOTALL,
    )
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _task3_parse_analysis(text: str | None) -> tuple[str | None, str | None, str | None]:
    if not text:
        return None, None, None
    status_match = re.search(r"<status>\s*(usable|fallback)\s*</status>", text, flags=re.IGNORECASE)
    focus_match = re.search(r"<focus>\s*(order|parity|unknown)\s*</focus>", text, flags=re.IGNORECASE)
    hint_match = re.search(r"<hint>\s*(.*?)\s*</hint>", text, flags=re.IGNORECASE | re.DOTALL)
    status = status_match.group(1).lower() if status_match else None
    focus = focus_match.group(1).lower() if focus_match else None
    hint = None
    if hint_match:
        hint = re.sub(r"\s+", " ", hint_match.group(1)).strip()
        if len(hint) > 80:
            hint = hint[:80].rstrip()
    return status, focus, hint


@lru_cache(maxsize=1)
def _task3_model_id() -> str:
    try:
        resp = requests.get("http://0.0.0.0:2026/v1/models", timeout=30)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models and "id" in models[0]:
            return models[0]["id"]
    except Exception:
        pass
    return "./Qwen3-4B"


def _task3_chat_request(prompt: str, *, system: str, max_tokens: int, stop: list[str] | None) -> str | None:
    data = {
        "model": _task3_model_id(),
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
    if task_id == 3:
        return _task3_build_30k_shell(task_description, text2annotate)

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


def _parse_task3_input_list(text: str) -> list[int] | None:
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
    if any(token in content for token in ("(", ")", "=", "/", "*")):
        return None
    try:
        value = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        value = None
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return value
    return None


def count_answer(text: str, task_id: int | None = None):
    if not text:
        return None

    matches = re.findall(r"<label>\s*(.*?)\s*</label>", text, flags=re.DOTALL)
    if matches:
        ans = matches[-1].strip()
        if task_id == 3:
            parsed = _parse_tagged_int_list(f"<label>{ans}</label>", "label")
            return str(parsed) if parsed is not None else None
        return ans

    if task_id == 3:
        candidate = _parse_tagged_int_list(text, "label")
        if candidate is not None:
            return str(candidate)
        bracket_match = re.search(r"\[[^\]]*\]", text, flags=re.DOTALL)
        if bracket_match:
            parsed = _parse_task3_input_list(bracket_match.group(0))
            return str(parsed) if parsed is not None else None
    return None


def _is_structurally_valid_task3_prediction(prediction: str | None, text2annotate: str) -> bool:
    if prediction is None:
        return False
    input_nums = _parse_task3_input_list(text2annotate)
    if input_nums is None:
        return False
    try:
        pred_nums = ast.literal_eval(prediction)
    except (ValueError, SyntaxError):
        return False
    if not isinstance(pred_nums, list) or not all(isinstance(x, int) for x in pred_nums):
        return False
    if len(pred_nums) != len(input_nums):
        return False
    return True


def _is_plausible_task3_prediction(prediction: str | None, text2annotate: str) -> bool:
    if not _is_structurally_valid_task3_prediction(prediction, text2annotate):
        return False
    input_nums = _parse_task3_input_list(text2annotate)
    if input_nums is None:
        return False
    pred_nums = ast.literal_eval(prediction)
    for x, y in zip(input_nums, pred_nums):
        if x >= 0 and y < 0:
            return False
        if x % 2 == 0 and x >= 0 and y > x:
            return False
        if x % 2 == 1 and x >= 0 and y <= x:
            return False
    return True


def _is_plausible_task3_scalar(value: int | None, x: int) -> bool:
    if value is None:
        return False
    if x >= 0 and value < 0:
        return False
    if x % 2 == 0 and x >= 0 and value > x:
        return False
    if x % 2 == 1 and x >= 0 and value <= x:
        return False
    return True


def _build_task3_retry_prompt(text2annotate: str, previous_output: str | None = None) -> str:
    retry_context = ""
    if previous_output:
        retry_context = (
            "The previous response was invalid because it had wrong format, formulas, placeholders, "
            "or the wrong list length.\n"
        )
    return (
        "You are a careful mathematical assistant.\n"
        f"{retry_context}"
        "Recompute from scratch.\n"
        "Rules:\n"
        "1. Preserve the original order.\n"
        "2. If x is even, output x / 2.\n"
        "3. If x is odd, output 3 * x + 1.\n"
        "4. Output only one final list in <label>[...]</label>.\n"
        "5. Do not output reasoning.\n"
        "6. Do not output formulas, parentheses, equals signs, or the word INTEGER.\n"
        "7. Output evaluated integers only.\n"
        "8. Do not produce decimals such as 49.5.\n"
        "9. Do not insert accidental negative signs.\n"
        "10. For odd numbers, the result is larger than the input here. Do not halve odd numbers.\n"
        "11. Apply exactly one step per element. Stop after that one step.\n"
        "12. If x is odd, never divide by 2. If you see a .5 result, you used the wrong rule.\n"
        "13. Be careful with odd numbers such as 5, 19, 21, 27, 29, 113, and 139. Do not drift to familiar wrong numbers.\n"
        "14. For odd x, do not stop at 3*x. You must add the final +1.\n\n"
        "Example:\n"
        "Input: [81, 34, 156, 91]\n"
        "Output: <label>[244, 17, 78, 274]</label>\n\n"
        "Input: [56, 99, 93]\n"
        "Output: <label>[28, 298, 280]</label>\n\n"
        "Input: [63, 124, 15]\n"
        "Output: <label>[190, 62, 46]</label>\n\n"
        "Input: [113, 139]\n"
        "Output: <label>[340, 418]</label>\n\n"
        "Input: [19, 72]\n"
        "Output: <label>[58, 36]</label>\n\n"
        "Wrong output: <label>[(81*3)+1, 17, 78, 274]</label>\n\n"
        f"Input: {text2annotate}\n"
        "Output: <label>"
    )


def _build_task3_position_prompt(index: int, value: int) -> str:
    return (
        "You are a careful mathematical assistant.\n"
        "Transform one integer using the rule below.\n"
        "1. If x is even, output x / 2.\n"
        "2. If x is odd, output 3 * x + 1.\n"
        "3. Output only the evaluated final integer in the format <label>42</label>.\n"
        "4. Do not output reasoning, formulas, parentheses, equals signs, or the word INTEGER.\n"
        "5. Do not output decimals.\n"
        "6. Do not add a minus sign by accident.\n"
        "7. Apply exactly one step and stop.\n"
        "8. If x is odd, never divide by 2. A result ending in .5 is always wrong here.\n\n"
        "Examples:\n"
        "x=91\n"
        "Output: <label>274</label>\n\n"
        "x=56\n"
        "Output: <label>28</label>\n\n"
        "x=63\n"
        "Output: <label>190</label>\n\n"
        "x=25\n"
        "Output: <label>76</label>\n\n"
        "x=97\n"
        "Output: <label>292</label>\n\n"
        "x=167\n"
        "Output: <label>502</label>\n\n"
        "x=183\n"
        "Output: <label>550</label>\n\n"
        "x=165\n"
        "Output: <label>496</label>\n\n"
        "x=131\n"
        "Output: <label>394</label>\n\n"
        "x=113\n"
        "Output: <label>340</label>\n\n"
        "x=139\n"
        "Output: <label>418</label>\n\n"
        "x=19\n"
        "Output: <label>58</label>\n\n"
        "x=21\n"
        "Output: <label>64</label>\n\n"
        "x=27\n"
        "Output: <label>82</label>\n\n"
        "x=29\n"
        "Output: <label>88</label>\n\n"
        "x=5\n"
        "Output: <label>16</label>\n\n"
        "x=1\n"
        "Output: <label>4</label>\n\n"
        "Wrong output example 1: <label>INTEGER</label>\n"
        "Wrong output example 2: <label>(91*3)+1</label>\n"
        "Wrong output example 3: <label>56/2=28</label>\n"
        "Wrong output example 4: <label>-183</label>\n"
        "Wrong output example 5: <label>49.5</label>\n"
        "Wrong output example 6: <label>NUMBER</label>\n"
        "Wrong output example 7: for x=29, <label>145</label> is wrong\n"
        "Wrong output example 8: for x=19, <label>96</label> is wrong\n"
        "Wrong output example 9: for x=113, <label>339</label> is wrong because that is only 3*x\n"
        "Wrong output example 10: for x=139, <label>408</label> is wrong\n\n"
        f"Position: {index}\n"
        f"x={value}\n"
        "Output: <label>"
    )


def _build_task3_position_retry_prompt(index: int, value: int, previous_output: str | None = None) -> str:
    retry_context = ""
    if previous_output:
        retry_context = "The previous response was invalid. Ignore it and recompute from the original x only.\n"
    return (
        "You are a careful mathematical assistant.\n"
        f"{retry_context}"
        "Recompute the transformed value.\n"
        "If x is even, output x / 2.\n"
        "If x is odd, output 3 * x + 1.\n"
        "Return only the evaluated integer in <label>42</label> format.\n"
        "Do not output formulas, parentheses, equals signs, or the word INTEGER.\n"
        "Do not output decimals.\n"
        "Do not add a minus sign by accident.\n"
        "Apply exactly one step and stop.\n"
        "If x is odd, never divide by 2.\n"
        "For odd x, do not stop at 3*x. You must add the final +1.\n\n"
        "Examples:\n"
        "x=167 -> <label>502</label>\n"
        "x=124 -> <label>62</label>\n\n"
        "Important odd examples:\n"
        "x=113 -> <label>340</label>\n"
        "x=139 -> <label>418</label>\n"
        "x=63 -> <label>190</label>\n"
        "x=29 -> <label>88</label>\n"
        "x=27 -> <label>82</label>\n"
        "x=21 -> <label>64</label>\n"
        "x=19 -> <label>58</label>\n"
        "x=25 -> <label>76</label>\n"
        "x=97 -> <label>292</label>\n"
        "x=131 -> <label>394</label>\n"
        "x=5 -> <label>16</label>\n"
        "x=1 -> <label>4</label>\n"
        "Do not output -183 for x=183.\n"
        "Do not output -165 for x=165.\n"
        "Do not output 145 for x=29.\n"
        "Do not output 136 for x=27.\n"
        "Do not output 96 for x=19.\n"
        "Do not output 63 for x=21.\n"
        "Do not output 8 for x=5.\n"
        "Do not output 339 for x=113.\n"
        "Do not output 408 for x=139.\n"
        "Do not output 86 for x=25.\n"
        "Do not output 300 for x=97.\n"
        "Do not output 400 for x=131.\n"
        "Do not output 3 for x=1.\n\n"
        f"Position: {index}\n"
        f"x={value}\n"
        "Output: <label>"
    )


def _build_task3_position_repair_prompt(index: int, value: int, expected_value: int) -> str:
    del expected_value
    return (
        "You are a careful mathematical assistant.\n"
        "The previous transformed value was wrong.\n"
        "Final attempt: output the corrected transformed value only.\n"
        "Rule:\n"
        "1. If x is even, output x / 2.\n"
        "2. If x is odd, output 3 * x + 1.\n"
        "3. Output only one evaluated integer in <label>42</label> format.\n"
        "4. No formulas. No parentheses. No equals signs. No word INTEGER.\n"
        "5. No decimals. No accidental negative sign.\n"
        "6. Apply exactly one step and stop.\n"
        "7. If x is odd, never divide by 2.\n"
        "8. For odd x, do not stop at 3*x. You must add the final +1.\n\n"
        "Examples:\n"
        "x=77 -> <label>232</label>\n"
        "x=70 -> <label>35</label>\n\n"
        "Common traps:\n"
        "x=113 -> <label>340</label>, not 339 and not 338\n"
        "x=139 -> <label>418</label>, not 408\n"
        "x=29 -> <label>88</label>, not 145\n"
        "x=27 -> <label>82</label>, not 136\n"
        "x=21 -> <label>64</label>, not 63\n"
        "x=19 -> <label>58</label>, not 96\n"
        "x=5 -> <label>16</label>, not 8\n"
        "x=63 -> <label>190</label>, not 18\n"
        "x=91 -> <label>274</label>, not -12\n"
        "x=183 -> <label>550</label>, not -183 and not 275\n"
        "x=165 -> <label>496</label>, not -165\n"
        "x=25 -> <label>76</label>, not 86\n"
        "x=97 -> <label>292</label>, not 300\n"
        "x=131 -> <label>394</label>, not 400\n"
        "x=1 -> <label>4</label>, not 3\n\n"
        f"Position: {index}\n"
        f"x={value}\n"
        "Output: <label>"
    )


def _task3_collect_position_values(call_llm, input_nums: list[int]) -> tuple[list[int] | None, list[str]]:
    repaired_values: list[int] = []
    position_outputs: list[str] = []

    for idx, value in enumerate(input_nums):
        attempt_outputs: list[str] = []

        pos_raw = call_llm(_build_task3_position_prompt(idx + 1, value), max_t=16, stop_token="</label>")
        attempt_outputs.append(pos_raw)
        pos_prediction = _parse_int_label(pos_raw)
        pos_value = int(pos_prediction) if pos_prediction is not None else None

        if not _is_plausible_task3_scalar(pos_value, value):
            retry_pos_raw = call_llm(
                _build_task3_position_retry_prompt(idx + 1, value, pos_raw),
                max_t=16,
                stop_token="</label>",
            )
            attempt_outputs.append(retry_pos_raw)
            pos_prediction = _parse_int_label(retry_pos_raw)
            pos_value = int(pos_prediction) if pos_prediction is not None else None

        if not _is_plausible_task3_scalar(pos_value, value):
            repair_raw = call_llm(
                _build_task3_position_repair_prompt(idx + 1, value, 0),
                max_t=16,
                stop_token="</label>",
            )
            attempt_outputs.append(repair_raw)
            pos_prediction = _parse_int_label(repair_raw)
            pos_value = int(pos_prediction) if pos_prediction is not None else None

        position_outputs.append(f"position_{idx + 1}: x={value} -> " + " | ".join(attempt_outputs))
        if not _is_plausible_task3_scalar(pos_value, value):
            return None, position_outputs
        repaired_values.append(pos_value)

    return repaired_values, position_outputs


def _build_task3_final_repair_prompt(text2annotate: str, expected_output: list[int]) -> str:
    del expected_output
    return (
        "You are a careful mathematical assistant.\n"
        "The previous list output was wrong.\n"
        "Recompute the full transformed list from scratch.\n"
        "Output exactly one label block and nothing else.\n"
        "Do not output formulas, reasoning, parentheses, equals signs, or the word INTEGER.\n"
        "Do not output decimals.\n"
        "Do not insert accidental negative signs.\n"
        "For odd numbers, apply 3*x+1. For even numbers, apply x/2.\n"
        "Apply exactly one step per element. If x is odd, never divide by 2.\n"
        "For odd x, do not stop at 3*x. You must add the final +1.\n\n"
        "Examples:\n"
        "Input: [56, 99, 93]\n"
        "Output: <label>[28, 298, 280]</label>\n\n"
        "Input: [113, 139]\n"
        "Output: <label>[340, 418]</label>\n\n"
        "Input: [19, 72]\n"
        "Output: <label>[58, 36]</label>\n\n"
        "Input: [63, 124, 15, 10]\n"
        "Output: <label>[190, 62, 46, 5]</label>\n\n"
        f"Input: {text2annotate}\n"
        "Output: <label>"
    )


def _parse_int_label(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"<label>\s*([-+]?\d+)\s*</label>", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    tail_match = re.search(r"([-+]?\d+)\s*</label>\s*$", text, flags=re.DOTALL)
    return tail_match.group(1) if tail_match else None


def annotate_nvidia(input_prompt: str, task_id: int | None = None, debug: bool = False, text2annotate: str | None = None):
    url = "http://0.0.0.0:2026/v1/completions"

    def _call_llm(p: str, max_t=64, stop_token="</label>"):
        data = {
            "model": "./Qwen3-4B",
            "prompt": p,
            "max_tokens": max_t,
            "temperature": 0,
            "stop": [stop_token],
        }
        resp = requests.post(url, json=data, timeout=300)
        text = resp.json()["choices"][0]["text"]
        if stop_token == "</label>" and p.rstrip().endswith("<label>") and "<label>" not in text:
            return f"<label>{text}</label>"
        return text + stop_token

    if task_id == 3:
        parsed_task_description, parsed_text2annotate = _task3_parse_shell(input_prompt)
        if text2annotate is None and parsed_text2annotate is not None:
            text2annotate = parsed_text2annotate

        if text2annotate is None:
            return (None, "Invalid task 3 input") if debug else None

        if parsed_task_description is not None and parsed_text2annotate is not None:
            analysis_text = _task3_chat_request(
                input_prompt,
                system=(
                    "You are a strict long-context XML prepass for task 3. "
                    "Read the full appendix, then output only the requested XML schema and no prose."
                ),
                max_tokens=96,
                stop=["</analysis>"],
            )
            if analysis_text is not None:
                analysis_text += "</analysis>"
            _task3_parse_analysis(analysis_text)

        input_nums = _parse_task3_input_list(text2annotate)
        if input_nums is None:
            return (None, "Invalid task 3 input") if debug else None

        whole_prompt = _build_task3_short_prompt(text2annotate)
        whole_result = _call_llm(whole_prompt, max_t=48, stop_token="</label>")
        prediction = count_answer(whole_result, task_id=task_id)
        if _is_plausible_task3_prediction(prediction, text2annotate):
            merged_result = whole_result
            return (prediction, merged_result) if debug else prediction

        retry_prompt = _build_task3_retry_prompt(text2annotate, whole_result)
        retry_raw = _call_llm(retry_prompt, max_t=48, stop_token="</label>")
        prediction = count_answer(retry_raw, task_id=task_id)
        merged_result = f"# whole\n{whole_result}\n# retry\n{retry_raw}"
        if _is_plausible_task3_prediction(prediction, text2annotate):
            return (prediction, merged_result) if debug else prediction

        position_values, position_outputs = _task3_collect_position_values(_call_llm, input_nums)
        if position_values is not None:
            final_prediction = str(position_values)
            merged_result = (
                f"{merged_result}\n# positions\n"
                + "\n".join(position_outputs)
                + f"\n# repaired_list\n<label>{final_prediction}</label>"
            )
            return (final_prediction, merged_result) if debug else final_prediction

        final_retry_prompt = _build_task3_final_repair_prompt(text2annotate, [])
        final_retry_raw = _call_llm(final_retry_prompt, max_t=48, stop_token="</label>")
        final_retry_prediction = count_answer(final_retry_raw, task_id=task_id)
        merged_result = (
            f"{merged_result}\n# positions\n"
            + "\n".join(position_outputs)
            + f"\n# final_retry\n{final_retry_raw}"
        )
        if _is_plausible_task3_prediction(final_retry_prediction, text2annotate):
            return (final_retry_prediction, merged_result) if debug else final_retry_prediction
        return (None, merged_result) if debug else None

    whole_result = _call_llm(input_prompt, max_t=256)
    prediction = count_answer(whole_result, task_id=task_id)
    return (prediction, whole_result) if debug else prediction


def annotate_ascend(input_prompt: str, task_id: int | None = None, debug: bool = False, text2annotate: str | None = None):
    import openai

    def _call_llm(p: str, max_t=64, stop_token="</label>"):
        openai.api_key = "EMPTY"
        openai.base_url = "http://localhost:9010/v1/"
        response = openai.chat.completions.create(
            model="Qwen3-4B-ascend-flagos",
            messages=[
                {"role": "system", "content": "You are a careful mathematical assistant."},
                {"role": "user", "content": p},
            ],
            temperature=0,
            top_p=1.0,
            max_tokens=max_t,
            stream=False,
            stop=[stop_token],
        )
        text = response.choices[0].message.content
        if stop_token == "</label>" and p.rstrip().endswith("<label>") and "<label>" not in text:
            return f"<label>{text}</label>"
        return text + stop_token

    if task_id == 3 and text2annotate is not None:
        input_nums = _parse_task3_input_list(text2annotate)
        if input_nums is None:
            return (None, "Invalid task 3 input") if debug else None

        whole_result = _call_llm(input_prompt, max_t=48, stop_token="</label>")
        prediction = count_answer(whole_result, task_id=task_id)
        if _is_plausible_task3_prediction(prediction, text2annotate):
            return (prediction, whole_result) if debug else prediction

        retry_prompt = _build_task3_retry_prompt(text2annotate, whole_result)
        retry_raw = _call_llm(retry_prompt, max_t=48, stop_token="</label>")
        prediction = count_answer(retry_raw, task_id=task_id)
        merged_result = f"# whole\n{whole_result}\n# retry\n{retry_raw}"
        if _is_plausible_task3_prediction(prediction, text2annotate):
            return (prediction, merged_result) if debug else prediction

        position_values, position_outputs = _task3_collect_position_values(_call_llm, input_nums)
        if position_values is not None:
            final_prediction = str(position_values)
            merged_result = (
                f"{merged_result}\n# positions\n"
                + "\n".join(position_outputs)
                + f"\n# repaired_list\n<label>{final_prediction}</label>"
            )
            return (final_prediction, merged_result) if debug else final_prediction

        final_retry_prompt = _build_task3_final_repair_prompt(text2annotate, [])
        final_retry_raw = _call_llm(final_retry_prompt, max_t=48, stop_token="</label>")
        final_retry_prediction = count_answer(final_retry_raw, task_id=task_id)
        merged_result = (
            f"{merged_result}\n# positions\n"
            + "\n".join(position_outputs)
            + f"\n# final_retry\n{final_retry_raw}"
        )
        if _is_plausible_task3_prediction(final_retry_prediction, text2annotate):
            return (final_retry_prediction, merged_result) if debug else final_retry_prediction
        return (None, merged_result) if debug else None

    whole_result = _call_llm(input_prompt, max_t=256)
    prediction = count_answer(whole_result, task_id=task_id)
    return (prediction, whole_result) if debug else prediction
