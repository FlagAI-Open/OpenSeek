import requests

url = "http://0.0.0.0:2026/v1/completions"
prompts = [
    "Hello, FlagScale + vLLM!",
    "Translate 'Hello World' to Chinese.",
    "Write a short poem about autumn."
    # '用中文写一首短诗，诗句开头用<label>，结尾用</label>包裹起来'
]

for prompt in prompts:
    data = {
        "model": "../Qwen3-4B",
        "prompt": prompt,
        "max_tokens": 1000
    }
    resp = requests.post(url, json=data)
    print(f"Prompt: {prompt}")
    print("Response:", resp.json(), "\n")

    print("*"*50)
