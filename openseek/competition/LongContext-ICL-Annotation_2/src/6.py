import json
import os
import re
from collections import Counter

import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-6_mnli_same_genre_classification.json"
OUTPUT_PATH = "../outputs/result/openseek-6-v1.jsonl"
TASK_ID = 6

def build_icl_prompt(dataset: dict, input_text: str) -> str:
    """
    Matched exactly from prompt.md logic for task_id == 6
    """
    #definition = dataset.get("Definition", [""])[0]
    #base_prompt = f"Task Definition:\n{definition}\n\n"

    # Rules for task 6
    rules = (
        
        #"Available genres include: government, letters, 9/11, slate, telephone, travel, verbatim, oup, fiction.\n"
        "If don't have the same words or synonymous word answer will be 'N'.The answer is 'Y' IF the two sentences contain the same words, synonymous words,even if there is one word output Y. "
    )
    base_prompt = f"Rules:\n{rules}\n\n"
    
    #base_prompt += "You are an expert data annotator. Think step-by-step in Analysis, then output the final answer clearly.\n\n"

    target_prompt = "The target input.\n"
    target_prompt += (
        f"--- Target ---\nInput: {input_text}\n"
        "According to the rule output ONLY 'Y' or 'N' at the end.\n"
        )
    
    return base_prompt + target_prompt

def count_answer(text: str):
    """
    Matched exactly from prompt.md logic for task_id == 6
    """
    if not text:
        return "N——"

    # Logic 1: when </think> exists, prioritize answer after it.
    if "</think>" in text:
        post_think = text.split("</think>")[-1].strip()
        post_matches = re.findall(r"\b([YN])\b", post_think, re.IGNORECASE)
        if post_matches:
            return post_matches[-1].upper()

    # Logic 2: when </think> does not exist, use regex majority vote.
    pattern = re.compile(
        r'(?:\bis\b|\bAnswer:?|\bOutput:?|\bshould\s+be\b|\bfinal\s+answer\b\**:?|\n)\s*[\'"]?\s*([YN])\b',
        re.IGNORECASE,
    )
    matches = pattern.findall(text)

    if not matches:
        return "N——"

    cleaned = [m.strip().upper() for m in matches if m.strip()]
    if not cleaned:
        return "N——"

    counts = Counter(cleaned)
    max_count = max(counts.values())
    candidates = set(k for k, v in counts.items() if v == max_count)
    for m in reversed(cleaned):
        if m in candidates:
            return m

    return "N"


def call_llm(prompt_text: str) -> str:
    """Call the local OpenAI-compatible model endpoint."""
    try:
        response = client.chat.completions.create(
            model="/root/ascend/log/debug/Qwen3-4B",
            messages=[
                {"role": "system", "content": "You are a highly capable to determine if the two sentences have same words or synonymous keywords."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.6,
            top_p=0.9,
            max_tokens=3000,
        )
        whole_result = response.choices[0].message.content
        print(f"\n[Model Response] Task {TASK_ID}:\n{whole_result}\n" + "=" * 50)
        return whole_result
    except Exception as e:
        print(f"Exception occurred for Task {TASK_ID}: {e}")
        return ""

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    test_samples = dataset.get("test_samples", [])

    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["test_sample_id"])
                except Exception:
                    continue

    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
        for sample in tqdm(test_samples):
            sample_id = sample['id']
            if sample_id in processed_ids:
                continue

            input_text = sample.get("input", "")
            prompt_text = build_icl_prompt(dataset, input_text)

            #print(f"\n{'='*20} PROMPT SENT TO MODEL {'='*20}")
            #print(prompt_text)
            #print(f"{'='*60}\n")

            response_raw = call_llm(prompt_text)
            prediction = count_answer(response_raw)

            result = {
                "test_sample_id": sample_id,
                "prediction": prediction
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

if __name__ == "__main__":
    main()
