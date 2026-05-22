"""
Method module: shared API call for all tasks.
"""
import os
import re


def _extract_answer(text: str) -> str:
    """Extract answer from model output: last <label>...</label> pair."""
    if not text:
        return ""

    last_label_pos = text.rfind('<label>')
    if last_label_pos != -1:
        remaining = text[last_label_pos + 7:]
        end_pos = remaining.find('</label>')
        if end_pos != -1:
            return _normalize_answer(remaining[:end_pos])
        else:
            clean = remaining.strip()
            if clean and len(clean) < 10000:
                return _normalize_answer(clean)
            return ""

    return ""


def _normalize_answer(text: str) -> str:
    """Post-process extracted answer to fix common formatting issues."""
    text = text.strip()
    if not text:
        return text
    if text.startswith('[') and text.endswith(']'):
        text = re.sub(r',\s*', ', ', text)
    return text


def annotate_ascend(
    input_prompt: str,
    temperature: float = 0.0,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 5000,
    stop_tokens: list = None,
) -> tuple:
    """Chat API. Returns (prediction, raw_output) tuple."""
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:2026/v1/")
    model = os.environ.get("MODEL_NAME", "/root/models/qwen/Qwen3-4B")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt},
    ]
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=False,
    )
    if stop_tokens:
        kwargs["stop"] = stop_tokens

    try:
        response = openai.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        whole_result = choice.content or choice.reasoning_content or ""
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        whole_result = ""

    prediction = _extract_answer(whole_result)
    return prediction, whole_result
