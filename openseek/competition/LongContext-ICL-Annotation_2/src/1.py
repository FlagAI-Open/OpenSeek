import json
import os
import re
import random
from collections import Counter

import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-1_closest_integers.json"
OUTPUT_PATH = "../outputs/result/openseek-1-v1.jsonl"
TASK_ID = 1
SHOT_COUNT = 8


def _extract_example_output_text(example: dict) -> str:
	output_value = example.get("output", "")
	if isinstance(output_value, list):
		return str(output_value[0]).strip() if output_value else ""
	return str(output_value).strip()


def format_few_shot_examples(examples: list) -> str:
	if not examples:
		return ""

	blocks = []
	for ex in examples:
		blocks.append(
			f"Input: {ex.get('input', '')}\n"
			f"Answer: {_extract_example_output_text(ex)}\n"
		)
	return "\n".join(blocks)


def build_icl_prompt(dataset: dict, input_text: str) -> str:
	"""Matched from prompt.md logic for task_id == 1."""
	definition = dataset.get("Definition", [""])[0]
	base_prompt = f"Task Definition:\n{definition}\n\n"

	base_prompt += (
		"You are an expert data annotator. Think step-by-step in Analysis, then output the final answer clearly.\n\n"
	)

	examples = dataset.get("examples", [])
	if examples:
		sampled_examples = random.sample(examples, min(SHOT_COUNT, len(examples)))

	target_prompt = "Now solve the target input.\n"
	target_prompt += (
		f"--- Target ---\nInput: {input_text}\n"
		"Analysis: Let's solve this step-by-step based on the task definition. "
		"First, I will compare integer pairs and their absolute differences, and finally determine that the single integer is "
	)

	return base_prompt + target_prompt


def count_answer(text: str):
	"""Matched from prompt.md logic for task_id == 1 with think-aware extraction."""
	if not text:
		return "no"

	# Logic 1: when </think> exists, extract answer after it.
	if "</think>" in text:
		post_think = text.split("</think>")[-1].strip()
		post_matches = re.findall(r"[-+]?\d+", post_think)
		return post_matches[-1] if post_matches else "no"

	# Logic 2: when </think> does not exist, use regex matching.
	pattern = re.compile(r"\bis\s+([-+]?\d+)", re.IGNORECASE)
	matches = pattern.findall(text)

	if not matches:
		return "no"

	cleaned = [m.strip() for m in matches if m.strip()]
	if not cleaned:
		return "no"

	counts = Counter(cleaned)
	max_count = max(counts.values())
	candidates = set(k for k, v in counts.items() if v == max_count)
	for m in reversed(cleaned):
		if m in candidates:
			return m

	return "no"


def call_llm(prompt_text: str) -> str:
	"""Call the local OpenAI-compatible model endpoint."""
	try:
		response = client.chat.completions.create(
			model="/root/ascend/log/debug/Qwen3-4B",
			messages=[
				{"role": "system", "content": "You are a highly capable data annotation AI."},
				{"role": "user", "content": prompt_text},
			],
			temperature=0.7,
			top_p=0.95,
			max_tokens=2000,
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

	with open(DATA_PATH, "r", encoding="utf-8") as f:
		dataset = json.load(f)
	test_samples = dataset.get("test_samples", [])

	if os.path.exists(OUTPUT_PATH):
		with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
			processed_ids = {json.loads(line)["test_sample_id"] for line in f if line.strip()}
	else:
		processed_ids = set()

	with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
		for sample in tqdm(test_samples):
			sample_id = sample["id"]
			if sample_id in processed_ids:
				continue

			input_text = sample.get("input", "")
			prompt_text = build_icl_prompt(dataset, input_text)
			response_raw = call_llm(prompt_text)
			prediction = count_answer(response_raw)

			result = {
				"test_sample_id": sample_id,
				"prediction": prediction if prediction else "no",
			}
			f.write(json.dumps(result, ensure_ascii=False) + "\n")
			f.flush()


if __name__ == "__main__":
	main()
