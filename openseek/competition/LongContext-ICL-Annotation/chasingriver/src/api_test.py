import requests

BASE_URL = "http://127.0.0.1:2026"

prompts = [
    "Hello, FlagScale + vLLM!",
    "Translate 'Hello World' to Chinese.",
    "Write a short poem about autumn."
]

def get_model_id():
    r = requests.get(f"{BASE_URL}/v1/models", timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["id"]

def main():
    model_id = get_model_id()
    print("Using model:", model_id)

    url = f"{BASE_URL}/v1/completions"

    for prompt in prompts:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            # 可选：减少“自言自语/继续扩写”的概率
            "stop": ["\n\n", "</s>"]
        }
        resp = requests.post(url, json=payload, timeout=300)

        print(f"\nPrompt: {prompt}")
        print("HTTP status:", resp.status_code)

        if resp.status_code != 200:
            print("Raw response (first 500 chars):")
            print(resp.text[:500])
            resp.raise_for_status()

        data = resp.json()
        text = data["choices"][0].get("text", "")
        print("Answer:", text.strip())
        print("*" * 50)

if __name__ == "__main__":
    main()
