#!/usr/bin/env python3
"""Autoresearch 用 LLM 客户端（DeepSeek / DashScope 兼容）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepseek_client import deepseek_chat


@dataclass
class ChatResponse:
    content: str
    reasoning_content: str | None = None


class LLMClient:
    def __init__(self, backend: str = "deepseek", *, model: str | None = None, base_url: str = "") -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8000,
    ) -> ChatResponse:
        if self.backend == "deepseek":
            text = self._chat_deepseek(messages, temperature=temperature, max_tokens=max_tokens)
            return ChatResponse(content=text or "", reasoning_content=None)
        if self.backend in ("dashscope", "local"):
            text = self._chat_dashscope(messages, temperature=temperature, max_tokens=max_tokens)
            return ChatResponse(content=text or "", reasoning_content=None)
        raise ValueError(f"unsupported backend: {self.backend}")

    def _chat_deepseek(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role}]\n{content}")
        prompt = "\n\n".join(parts)
        out = deepseek_chat(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)
        return out or ""

    def _chat_dashscope(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        from method_hyb import _nvidia_dashscope_chat_text

        prompt = "\n\n".join(f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages)
        return _nvidia_dashscope_chat_text(prompt) or ""


def create_client(backend: str, **kwargs: Any) -> LLMClient:
    return LLMClient(backend, model=kwargs.get("model"), base_url=kwargs.get("base_url", ""))
