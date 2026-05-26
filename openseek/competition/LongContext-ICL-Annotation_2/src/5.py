import json
import os
import re
import random
from typing import List, Dict
import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
OUTPUT_PATH = "../outputs/result/openseek-5-v1.jsonl"
TASK_ID = 5
SAD_SHOT_COUNT = 100
NOT_SAD_SHOT_COUNT = 50
MAX_RETRY_TIMES = 3


def remove_emoji(text: str) -> str:
    """Remove emoji and pictograph characters from input text."""
    if not text:
        return ""
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-c
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FAFF"  # chess and symbols extended-a
        "\U00002700-\U000027BF"  # dingbats
        "\U00002600-\U000026FF"  # misc symbols
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)

def build_custom_icl_prompt(task_id: int, dataset: dict, input_text: str) -> str:
    """
    Build prompt with grouped few-shot examples.
    """
    examples = dataset.get("examples", [])

    sad_examples = []
    not_sad_examples = []

    for ex in examples:
        label = ex["output"][0] if isinstance(ex["output"], list) else ex["output"]
        if label == "Sad":
            sad_examples.append(ex)
        elif label == "Not sad":
            not_sad_examples.append(ex)

    # Uniform sampling from the full grouped pools (without replacement).
    sad_examples = random.sample(sad_examples, min(SAD_SHOT_COUNT, len(sad_examples)))
    not_sad_examples = random.sample(not_sad_examples, min(NOT_SAD_SHOT_COUNT, len(not_sad_examples)))

    system_instr = (
        "Emotion binary classification task: Classify the input text into exactly one label: Sad or Not Sad. Classification Rules:\n"
        "Output Sad if the text expresses any negative emotions that make people feel bad, including but not limited to pain, anger, disappointment, fear, tiredness, criticism, sympathy, anger, sadness, disappointment, and insults.\n"
        "Output Not Sad if the text does not contain any negative emotions; this category includes positive emotions that make people feel good (such as happiness, excitement, praise, humor) and neutral content that makes people feel nothing (such as objective statements of facts).\n"
        "Special Note on Traps:\n"
        "Pay attention to the ambiguity of emojis. For example, the emoji 😭 can mean extreme sadness (label as Sad) or laughing to death/too envious (label as Not Sad); you must judge based on the context of the text (e.g., if the text contains U so lucky, 😭 here means envy, so label as Not Sad).\n"
        "Output only one answer: Sad / Not sad.\n\n"
    )

    examples_str = ""
    examples_str += "Sad examples:\n"
    for ex in sad_examples:
        examples_str += f"Tweet: {ex['input']}\nLabel: Sad\n\n"

    examples_str += "Not sad examples:\n"
    for ex in not_sad_examples:
        examples_str += f"Tweet: {ex['input']}\nLabel: Not sad\n\n"

    target_prompt = (
        #examples_str +
        f"Tweet: {input_text}\n"
        "Label: "
    )

    return system_instr + target_prompt

def call_llm(prompt_text: str) -> str:
    """Implement logic from prompt.md"""
    try:
        response = client.chat.completions.create(
            model="/root/ascend/log/debug/Qwen3-4B",
            messages=[
                {"role": "system", "content": "You are a highly empathetic identifier."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.5, # Lower temperature for more stable classification
            top_p=0.8,
            max_tokens=2500,
        )
        whole_result = response.choices[0].message.content
        print(f"\n[Model Response] Task {TASK_ID}:\n{whole_result}\n" + "="*50)
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

    # Load existing results to support resume
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['test_sample_id'])
                except:
                    continue

    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
        for sample in tqdm(test_samples):
            sample_id = sample['id']
            if sample_id in processed_ids:
                continue
            
            #input_text = remove_emoji(sample.get("input", ""))
            input_text = sample.get("input", "")
            prompt_text = build_custom_icl_prompt(TASK_ID, dataset, input_text)
            
            # Print the final prompt sent to the model for debugging
            #print(f"\n{'='*20} PROMPT SENT TO MODEL (TASK {TASK_ID}) {'='*20}")
            #print(prompt_text)
            #print(f"{'='*60}\n")
            
            # Retry the same sample until the model returns a completed think block.
            retry_count = 0
            while True:
                response_raw = call_llm(prompt_text)
                if "</think>" in response_raw:
                    post_think_text = response_raw.split("</think>")[-1].strip()
                    break
                retry_count += 1
                if retry_count > MAX_RETRY_TIMES:
                    print(
                        f"[Retry] Task {TASK_ID}, sample {sample_id}: "
                        f"retry limit {MAX_RETRY_TIMES} exceeded, using latest raw response"
                    )
                    post_think_text = response_raw.strip()
                    break
                print(f"[Retry] Task {TASK_ID}, sample {sample_id}: missing </think>, retrying ({retry_count})")

            # Prefer explicit final answer spans, and prioritize "not sad" before "sad".
            prediction = "Not"
            answer_pattern = re.compile(
                r'(?:\*\*\s*answer\s*\*\*|\banswer\b|\bfinal\s+answer\b)\s*[:：-]?\s*(not[\s_-]?sad|sad)\b',
                re.IGNORECASE,
            )
            answer_matches = answer_pattern.findall(post_think_text)

            if answer_matches:
                label = answer_matches[-1].strip().lower().replace("_", " ")
                prediction = "Not sad" if label == "not sad" else "Sad"
            else:
                # Fallback: use the last sentiment label mention in text.
                pattern_fallback = re.compile(r'\b(not[\s_-]?sad|sad)\b', re.IGNORECASE)
                matches = pattern_fallback.findall(post_think_text)
                if matches:
                    label = matches[-1].strip().lower().replace("_", " ")
                    prediction = "Not sad" if label == "not sad" else "Sad"
            
            result = {
                "test_sample_id": sample_id,
                "prediction": prediction
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

if __name__ == "__main__":
    main()