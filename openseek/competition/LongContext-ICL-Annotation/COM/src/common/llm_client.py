"""Shared LLM client for all 8 tasks.

Provides:
- ``annotate_nvidia(...)`` : OpenAI-SDK based vLLM (Qwen3-4B) inference wrapper
  with thinking-mode support, repetition penalty, top-k/min-p sampling and
  automatic ``<label>`` / code-block extraction.
- ``count_answer(...)``    : robust multi-level answer extraction from raw
  model output (`<label>...</label>` first, then fallbacks).

Endpoint and model id are taken from :mod:`common.paths` (env-overridable
via ``FLAGOS_VLLM_BASE_URL`` / ``FLAGOS_VLLM_MODEL_ID``); see
``env/llm_config.yaml`` for how the service is launched.
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Optional, Tuple, Union

from openai import OpenAI

from .paths import VLLM_BASE_URL as DEFAULT_BASE_URL
from .paths import VLLM_MODEL_ID as DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton client (vLLM OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
_client = OpenAI(api_key="EMPTY", base_url=DEFAULT_BASE_URL)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def count_answer(text: str) -> Optional[str]:
    """Extract the annotation result from raw model output (multi-level fallback)."""
    if not text or text.strip() == "None":
        logger.warning("count_answer: empty / 'None' input")
        return None

    logger.debug(f"count_answer: len={len(text)} preview='{text[:100]}...'")

    # Primary: <label>...</label>
    matches = re.findall(r"<label>\s*(.+?)\s*</label>", text, re.DOTALL)
    if matches:
        counter = Counter(matches)
        max_count = max(counter.values())
        # Pick the most-frequent answer; ties broken by first appearance
        winner = [c for c, cnt in counter.items() if cnt == max_count][0]
        result = re.sub(r"</?label>", "", winner).strip()
        if len(result) > 50000:
            logger.warning(f"answer too long ({len(result)} chars); reject")
            return None
        return result

    # Fallback 1: strip common prefixes
    cleaned = text.strip()
    prefixes = [
        "Answer:", "Result:", "Output:", "Label:",
        "answer:", "result:", "output:", "label:",
        "The answer is:", "The result is:",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            if cleaned and "\n" not in cleaned and len(cleaned) <= 500:
                logger.info(f"count_answer fallback-1: '{cleaned[:80]}'")
                return cleaned
            break
    if "\n" not in cleaned and cleaned and len(cleaned) <= 200:
        logger.info(f"count_answer fallback-1 (single-line): '{cleaned[:80]}'")
        return cleaned

    # Fallback 2: last non-empty line
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if lines:
        last = re.sub(r"</?label>", "", lines[-1]).strip()
        if last and len(last) <= 500:
            logger.info(f"count_answer fallback-2: '{last[:80]}...'")
            return last

    logger.warning(f"count_answer failed; raw preview: {text[:200]}")
    return None


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------
def annotate_nvidia(
    input_prompt: str,
    *,
    max_retries: int = 2,
    timeout: int = 120,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    seed: Optional[int] = None,
    repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
    return_raw: bool = False,
    model: str = DEFAULT_MODEL,
) -> Union[Optional[str], Tuple[Optional[str], str]]:
    """Run a single chat completion against the local vLLM endpoint.

    Args:
        enable_thinking: If True, vLLM exposes ``reasoning_content``
            (Qwen3 thinking mode); the routine will fall back to extracting
            ``<label>`` / code blocks from ``reasoning_content`` when
            ``content`` is empty.
        top_p / top_k / min_p / repetition_penalty: passed through ``extra_body``
            (vLLM-specific knobs); ``top_p`` is also accepted directly by SDK.
        return_raw: When True, return ``(prediction, raw_output)``.

    Returns:
        prediction (str | None) or (prediction, raw_output) when ``return_raw``.
    """
    messages = [{"role": "user", "content": input_prompt}]

    extra_body: dict = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = repetition_penalty
    if top_k is not None:
        extra_body["top_k"] = top_k
    if min_p is not None:
        extra_body["min_p"] = min_p

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": extra_body,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if seed is not None:
        kwargs["seed"] = seed

    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            response = _client.chat.completions.create(**kwargs, timeout=timeout)
            elapsed_ms = (time.time() - t0) * 1000

            message = response.choices[0].message
            whole_result = message.content or ""
            usage = response.usage
            reasoning = getattr(message, "reasoning_content", None)

            if reasoning:
                logger.info(f"thinking:\n{reasoning}")
            logger.debug(
                f"API: {elapsed_ms:.0f}ms, "
                f"prompt={usage.prompt_tokens if usage else '?'}, "
                f"completion={usage.completion_tokens if usage else '?'}"
            )
            logger.info(f"raw output:\n{whole_result}")

            # thinking-mode rescue: content empty → mine reasoning_content
            if not whole_result.strip() and reasoning:
                logger.info("content empty, scraping reasoning_content")
                tag_m = re.search(
                    r"<(?:label|answer)>\s*(.*?)\s*</(?:label|answer)>",
                    reasoning, re.DOTALL,
                )
                if tag_m:
                    whole_result = tag_m.group(1).strip()
                else:
                    code_blocks = re.findall(
                        r"```(?:python)?\s*\n(.*?)```", reasoning, re.DOTALL
                    )
                    if code_blocks:
                        whole_result = code_blocks[-1].strip()
                    else:
                        for ci, line in enumerate(reasoning.split("\n")):
                            sl = line.strip()
                            if (sl.startswith("import ") or sl.startswith("from ")
                                    or sl.startswith("@triton")):
                                whole_result = "\n".join(
                                    reasoning.split("\n")[ci:]
                                )
                                break

            raw_content = whole_result

            # Normalise alternative tag names → <label>
            whole_result = re.sub(
                r"<(answer|fixed_code|code|solution|output|result|response|python)>",
                "<label>", whole_result,
            )
            whole_result = re.sub(
                r"</(answer|fixed_code|code|solution|output|result|response|python)>",
                "</label>", whole_result,
            )

            label_m = re.search(
                r"<label>\s*(.+?)\s*</label>", whole_result, re.DOTALL
            )
            label_content = label_m.group(1).strip() if label_m else ""

            if label_content:
                # keep last <label> (recency bias for CoT outputs)
                if whole_result.count("<label>") > 1:
                    whole_result = whole_result[whole_result.rfind("<label>"):]
                if "</label>" not in whole_result:
                    whole_result += "</label>"
            else:
                # No usable <label>: try markdown code block / xml tag / import row
                extracted = None
                code_blocks = re.findall(
                    r"```(?:python)?\s*\n(.*?)```", whole_result, re.DOTALL
                )
                if code_blocks:
                    extracted = code_blocks[-1].strip()
                else:
                    unclosed = re.search(
                        r"```(?:python)?\s*\n(.+)", whole_result, re.DOTALL
                    )
                    if unclosed:
                        extracted = unclosed.group(1).strip()
                        if extracted.endswith("```"):
                            extracted = extracted[:-3].strip()
                if not extracted:
                    any_tag = re.search(
                        r"<(\w+)>\s*(.*?)\s*</\1>", whole_result, re.DOTALL
                    )
                    if any_tag and len(any_tag.group(2).strip()) > 10:
                        extracted = any_tag.group(2).strip()
                if not extracted:
                    for line in whole_result.split("\n"):
                        sl = line.strip()
                        if (sl.startswith("import ") or sl.startswith("from ")
                                or sl.startswith("@triton")):
                            idx = whole_result.index(line)
                            extracted = whole_result[idx:].strip()
                            break
                if extracted:
                    whole_result = f"<label>{extracted}</label>"
                else:
                    whole_result = "<label>" + whole_result.strip() + "</label>"

            prediction = count_answer(whole_result)
            if prediction is not None:
                logger.info(f"prediction: '{prediction[:80]}' ({elapsed_ms:.0f}ms)")
            else:
                logger.warning(f"count_answer=None; wrapped='{whole_result[:200]}'")

            if return_raw:
                return prediction, raw_content
            return prediction

        except Exception as e:
            logger.warning(f"API exception (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(1)

    logger.error("all retries exhausted")
    if return_raw:
        return None, ""
    return None
