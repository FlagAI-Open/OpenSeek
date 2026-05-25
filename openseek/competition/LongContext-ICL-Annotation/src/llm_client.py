import requests
from pathlib import Path
from typing import List, Union, Optional

CONFIG_PATH = Path(__file__).resolve().parent / "llm_config.yaml"


def load_config(config_path: Path) -> dict[str, str]:
    base_url = "http://localhost:8000/v1"
    model = "Qwen3-4B"
    api_key = "EMPTY"

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "api:":
            continue
        if line.startswith("base_url:"):
            base_url = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("model:"):
            model = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("api_key:"):
            api_key = line.split(":", 1)[1].strip().strip('"').strip("'")

    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
    }


config = load_config(CONFIG_PATH)
API_URL = config["base_url"]
MODEL_NAME = config["model"]
API_KEY = config["api_key"]

class LLMClient:
    def __init__(self):
        self.api_url = API_URL
        self.model_name = MODEL_NAME
        self.api_key = API_KEY
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def post_chat_completion(
        self,
        messages: List[dict],
        max_tokens: int = 5000,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        timeout: int = 600
    ) -> str:
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop: data["stop"] = stop
        try:
            url = f"{self.api_url}/chat/completions"
            response = requests.post(url, json=data, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            res_json = response.json()
            return str(res_json["choices"][0].get("message", {}).get("content", "")) if "choices" in res_json else ""
        except Exception as e:
            print(f"LLM Chat API Error: {e}")
            raise

    def post_completion(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 5000,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        timeout: int = 600
    ) -> str:
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop: data["stop"] = stop
        try:
            url = f"{self.api_url}/completions"
            response = requests.post(url, json=data, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            res_json = response.json()
            return str(res_json["choices"][0].get("text", "")) if "choices" in res_json else ""
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/models", headers=self.headers, timeout=5)
            return response.status_code == 200
        except: return False

client = LLMClient()

def post_completion(prompt, **kwargs) -> str:
    return client.post_completion(prompt, **kwargs)

def post_chat(messages: List[dict], **kwargs) -> str:
    return client.post_chat_completion(messages, **kwargs)

def is_completion_server_available() -> bool:
    return client.is_available()
