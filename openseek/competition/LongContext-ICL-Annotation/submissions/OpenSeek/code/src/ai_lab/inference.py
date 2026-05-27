import re
from typing import Any, Dict, Optional

import requests


class InferenceBackend:
    def generate_raw(self, prompt: str, task_type: Optional[str] = None) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, task_type: str) -> str:
        return normalize_prediction(self.generate_raw(prompt, task_type=task_type), task_type)


class HeuristicBackend(InferenceBackend):
    def __init__(self, fallback_prediction: str = "") -> None:
        self.fallback_prediction = fallback_prediction

    def generate_raw(self, prompt: str, task_type: Optional[str] = None) -> str:
        return self.fallback_prediction


class FlagScaleAPIBackend(InferenceBackend):
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        self.api_url = model_cfg["api_url"]
        self.api_model_name = model_cfg["api_model_name"]
        self.max_new_tokens = int(model_cfg["max_new_tokens"])
        self.max_new_tokens_by_task = dict(model_cfg.get("max_new_tokens_by_task", {}))
        self.temperature = float(model_cfg.get("temperature", 0.0))
        self.top_p = float(model_cfg.get("top_p", 1.0))
        self.timeout_seconds = int(model_cfg.get("timeout_seconds", 600))

    def _resolve_max_new_tokens(self, task_type: Optional[str]) -> int:
        if task_type and task_type in self.max_new_tokens_by_task:
            return int(self.max_new_tokens_by_task[task_type])
        return self.max_new_tokens

    def generate_raw(self, prompt: str, task_type: Optional[str] = None) -> str:
        payload = {
            "model": self.api_model_name,
            "prompt": prompt,
            "max_tokens": self._resolve_max_new_tokens(task_type),
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        response = requests.post(self.api_url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"]


class TransformersLocalBackend(InferenceBackend):
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        self.model_path = model_cfg["model_path"]
        self.tokenizer_path = model_cfg.get("tokenizer_path", self.model_path)
        self.device = model_cfg.get("device", "cuda")
        self.max_new_tokens = int(model_cfg["max_new_tokens"])
        self.max_new_tokens_by_task = dict(model_cfg.get("max_new_tokens_by_task", {}))
        self.temperature = float(model_cfg.get("temperature", 0.0))
        self.top_p = float(model_cfg.get("top_p", 1.0))
        self.repetition_penalty = float(model_cfg.get("repetition_penalty", 1.0))
        self.trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        self.dtype = str(model_cfg.get("dtype", "bfloat16"))
        self.load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
        self.bnb_4bit_compute_dtype = str(
            model_cfg.get("bnb_4bit_compute_dtype", "float16")
        )
        self.bnb_4bit_quant_type = str(model_cfg.get("bnb_4bit_quant_type", "nf4"))
        self.bnb_4bit_use_double_quant = bool(
            model_cfg.get("bnb_4bit_use_double_quant", True)
        )
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=self.trust_remote_code,
        )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        compute_dtype = dtype_map.get(self.bnb_4bit_compute_dtype, torch.float16)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = "auto" if self.device == "cuda" else None

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        if self.device != "cuda":
            self._model.to(self.device)
        self._model.eval()

    def _resolve_max_new_tokens(self, task_type: Optional[str]) -> int:
        if task_type and task_type in self.max_new_tokens_by_task:
            return int(self.max_new_tokens_by_task[task_type])
        return self.max_new_tokens

    def generate_raw(self, prompt: str, task_type: Optional[str] = None) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(prompt, return_tensors="pt")
        if self.device != "cuda":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        elif hasattr(self._model, "device"):
            encoded = {k: v.to(self._model.device) for k, v in encoded.items()}

        do_sample = self.temperature > 0.0
        generated = self._model.generate(
            **encoded,
            max_new_tokens=self._resolve_max_new_tokens(task_type),
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = generated[0][encoded["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_backend(model_cfg: Dict[str, Any], fallback_prediction: str = "") -> InferenceBackend:
    backend = model_cfg.get("backend", "heuristic")
    if backend == "heuristic":
        return HeuristicBackend(fallback_prediction=fallback_prediction)
    if backend == "flagscale_api":
        return FlagScaleAPIBackend(model_cfg)
    if backend == "transformers_local":
        return TransformersLocalBackend(model_cfg)
    raise ValueError(f"Unsupported model.backend: {backend}")


def normalize_prediction(text: Optional[str], task_type: str) -> str:
    if text is None:
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    if task_type == "code_generation":
        code_match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return cleaned

    labels = re.findall(r"<label>\s*(.*?)\s*</label>", cleaned, flags=re.DOTALL)
    if labels:
        return labels[-1].strip()

    first_line = cleaned.splitlines()[0].strip()
    return first_line
