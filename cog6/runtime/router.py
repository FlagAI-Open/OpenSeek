import requests
import json
import logging
import time

logger = logging.getLogger("COG-6")

class RuntimeRouter:
    def __init__(self, mode="ollama"):
        self.mode = mode
        self.endpoint = "http://localhost:11434/api/generate"
        self.model = "qwen2.5:3b"
        self.timeout = 120
        
        if self.mode == "flagscale":
            logger.info("[COG-6] FlagScale backend selected. Initializing vLLM endpoint configuration.")
            self.endpoint = "http://localhost:8000/v1/completions"

    def generate(self, prompt: str) -> str:
        if self.mode == "ollama":
            return self._call_ollama(prompt)
        elif self.mode == "flagscale":
            return self._call_flagscale(prompt)
        else:
            raise ValueError(f"Unknown runtime mode: {self.mode}")

    def _call_flagscale(self, prompt: str) -> str:
        logger.info("[COG-6] Sending request to FlagScale vLLM backend...")
        # Abstracted for Track 3 submission. Exact payload structure matches standard OpenAI API or vLLM native.
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.0
        }
        
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                logger.info("[COG-6] Response received from FlagScale...")
                return result.get("choices", [{}])[0].get("text", "").strip()
            except requests.exceptions.RequestException as e:
                logger.warning(f"[COG-6] FlagScale request error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    logger.error("[COG-6] Max retries reached for FlagScale. Inference failed.")
                    return ""
                logger.info("[COG-6] Retrying FlagScale inference...")
        return ""

    def _call_ollama(self, prompt: str) -> str:
        logger.info("[COG-6] Sending request to Ollama...")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                logger.info("[COG-6] Response received...")
                return result.get("response", "").strip()
            except requests.exceptions.Timeout:
                logger.warning(f"[COG-6] Timeout error on attempt {attempt + 1}.")
                if attempt == max_retries:
                    logger.error("[COG-6] Max retries reached. Inference failed.")
                    return ""
                logger.info("[COG-6] Retrying inference...")
            except requests.exceptions.RequestException as e:
                logger.warning(f"[COG-6] Request error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    logger.error("[COG-6] Max retries reached. Inference failed.")
                    return ""
                logger.info("[COG-6] Retrying inference...")
        return ""
