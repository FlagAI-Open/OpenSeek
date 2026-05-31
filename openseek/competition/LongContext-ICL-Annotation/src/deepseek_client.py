"""DeepSeek OpenAI 兼容 API 客户端（供 test_samples 推理脚本共用）。"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from openai import OpenAI

DEEPSEEK_API_KEY = "sk-340c73f8209e4ba3ae44c7e6cf570506"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

_tls = threading.local()


def thread_local_client() -> OpenAI:
    client = getattr(_tls, "client", None)
    if client is None:
        _tls.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _tls.client


def _resolved_model(model: str | None = None) -> str:
    if model:
        return model
    override = os.environ.get("DEEPSEEK_MODEL_OVERRIDE", "").strip()
    return override or DEEPSEEK_MODEL


def deepseek_chat(
    input_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str | None:
    client = thread_local_client()
    completion = client.chat.completions.create(
        model=_resolved_model(model),
        messages=[{"role": "user", "content": input_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = completion.choices[0].message.content
    if content is None:
        return None
    return str(content).strip()


def annotate_via_count_answer(input_prompt: str, *, model: str | None = None) -> str | None:
    from method_hyb import count_answer

    raw = deepseek_chat(input_prompt, model=model)
    if raw is None:
        return None
    parsed = count_answer(raw)
    return None if parsed is None else str(parsed).strip()


def annotate_with_retry(
    input_prompt: str,
    *,
    retries: int = 3,
    retry_wait_seconds: float = 2.0,
    model: str | None = None,
    on_error: Callable[[int, int, Exception], None] | None = None,
) -> str:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = annotate_via_count_answer(input_prompt, model=model)
            return "" if result is None else result
        except Exception as e:  # noqa: BLE001
            last_err = e
            if on_error is not None:
                on_error(attempt, retries, e)
            elif attempt >= retries:
                print(f"[DeepSeek 推理失败] attempt={attempt}/{retries} error={e}")
            else:
                print(f"[DeepSeek 重试] attempt={attempt}/{retries} error={e}")
                time.sleep(retry_wait_seconds)
    if last_err is not None:
        print(f"[DeepSeek 放弃] error={last_err}")
    return ""


def install_deepseek_as_annotate_nvidia() -> None:
    """在导入 task6 流水线模块之前调用，将各处的 ``annotate_nvidia`` 替换为 DeepSeek。"""
    import method_hyb
    import postprocess_task6_v2 as p2
    import postprocess_task6_v3 as p3

    method_hyb.annotate_nvidia = annotate_via_count_answer  # type: ignore[assignment]
    p2.annotate_nvidia = annotate_via_count_answer  # type: ignore[assignment]
    p3.annotate_nvidia = annotate_via_count_answer  # type: ignore[assignment]
