from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import ModelConfig


class ModelClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._local_pipeline: Any | None = None
        self._local_tokenizer: Any | None = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self.config.provider == "mock":
            return json.dumps({"output": "unknown", "confidence": 0.0, "rationale": "mock provider"})
        if self.config.provider == "local_transformers":
            return self._complete_local(messages)
        if self.config.provider != "openai_compatible":
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        if not self.config.base_url or not self.config.api_key:
            raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY are required for openai_compatible provider")

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model request failed with HTTP {error.code}: {detail}") from error
        return str(data["choices"][0]["message"]["content"])

    def _load_local(self) -> None:
        if self._local_pipeline is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        except ImportError as error:
            raise RuntimeError(
                "local_transformers provider requires torch, transformers, accelerate, and bitsandbytes."
            ) from error

        model_id = self.config.local_model_path or "Qwen/Qwen3-4B"
        quantization_config = None
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quantization_config

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self._local_tokenizer = tokenizer
        self._local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def _complete_local(self, messages: list[dict[str, str]]) -> str:
        self._load_local()
        assert self._local_pipeline is not None
        assert self._local_tokenizer is not None
        prompt = self._local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        outputs = self._local_pipeline(
            prompt,
            max_new_tokens=self.config.max_tokens,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 0.01),
            return_full_text=False,
        )
        return str(outputs[0]["generated_text"]).strip()
