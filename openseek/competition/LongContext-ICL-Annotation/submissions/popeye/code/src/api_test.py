import os

import requests


DEFAULT_URL = os.environ.get("OPENSEEK_VLLM_URL", "http://127.0.0.1:2026/v1/completions")
DEFAULT_MODEL = os.environ.get("OPENSEEK_MODEL_NAME", "../Qwen3-4B")


def main():
    prompts = [
        "Hello, FlagScale + vLLM!",
        "Translate 'Hello World' to Chinese.",
        "Write a short poem about autumn.",
    ]

    for prompt in prompts:
        data = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 256,
        }
        resp = requests.post(DEFAULT_URL, json=data, timeout=60)
        resp.raise_for_status()
        print(f"Prompt: {prompt}")
        print("Response:", resp.json(), "\n")
        print("*" * 50)


if __name__ == "__main__":
    main()
