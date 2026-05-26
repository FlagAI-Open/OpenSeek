import json
import os
import re
from collections import Counter

import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-3_collatz_conjecture.json"
OUTPUT_PATH = "../outputs/result/openseek-3-v1.jsonl"
TASK_ID = 3


def build_icl_prompt(dataset: dict, input_text: str) -> str:
	"""
	Matched from prompt.md logic for task_id == 3.
	"""
	definition = dataset.get("Definition", [""])[0]
	base_prompt = f"Task Definition:\n{definition}\n\n"

	rules = (
		"Output ONLY the final sequence result.\\n"
		"Follow the Collatz conjecture progression strictly."
	)
	base_prompt += f"Rules:\n{rules}\n\n"

	base_prompt += "You are an expert data annotator. Think step-by-step in Analysis, then output the final answer clearly.\n\n"

	target_prompt = "Now solve the target input.\n"
	target_prompt += (
		f"--- Target ---\nInput: {input_text}\n"
		"Analysis: Let's solve this step-by-step based on the task definition. "
		"First, I will analyze each number in the list, apply the Collatz rule, and finally determine that the list of integers result is "
	)

	return base_prompt + target_prompt


def count_answer(text: str):
	"""
	Matched from prompt.md logic for task_id == 3:
	extract list-form answer like [1, 2, 3].
	"""
	if not text:
		return "no"

	# Logic 1: when </think> exists, extract answer after it.
	if "</think>" in text:
		post_think = text.split("</think>")[-1].strip()
		post_matches = re.findall(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]", post_think)
		return post_matches[-1] if post_matches else "no"

	# Logic 2: when </think> does not exist, use regex matching.
	pattern = re.compile(
		r"(?:\\bis\\b|\\bAnswer:?|\\bOutput:?|\\bshould\\s+be\\b)?\\s*(\\\\boxed\{)?(\[[\d\s,.-]*\])(\})?",
		re.IGNORECASE,
	)
	matches = pattern.findall(text)

	final_prediction = None
	if matches:
		cleaned = []
		for m in matches:
			# m is a tuple: (boxed_prefix, list_text, boxed_suffix)
			if isinstance(m, tuple) and len(m) >= 2:
				candidate = m[1]
			else:
				candidate = m

			boxed_match = re.search(r"\\\\boxed\{(.*?)\}", candidate)
			if boxed_match:
				cleaned.append(boxed_match.group(1))
			else:
				cleaned.append(candidate)

		cleaned = [c.strip().strip("'\" ").strip() for c in cleaned if c.strip()]

		# Keep only valid integer-list strings.
		cleaned = [
			c
			for c in cleaned
			if re.fullmatch(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]", c)
		]

		if cleaned:
			counts = Counter(cleaned)
			max_count = max(counts.values())
			candidates = set(k for k, v in counts.items() if v == max_count)
			for c in reversed(cleaned):
				if c in candidates:
					final_prediction = c
					break

	if isinstance(final_prediction, str):
		final_prediction = "".join(ch for ch in final_prediction if ord(ch) >= 32 or ch in "\n\r\t")

	return final_prediction if final_prediction is not None else "no"


def call_llm(prompt_text: str) -> str:
	"""Implement logic from existing task scripts."""
	try:
		response = client.chat.completions.create(
			model="/root/ascend/log/debug/Qwen3-4B",
			messages=[
				{"role": "system", "content": "You are a highly capable data annotation AI."},
				{"role": "user", "content": prompt_text},
			],
			temperature=0.6,
			top_p=0.9,
			max_tokens=2500,
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

	processed_ids = set()
	if os.path.exists(OUTPUT_PATH):
		with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				try:
					data = json.loads(line)
					processed_ids.add(data["test_sample_id"])
				except Exception:
					continue

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
				"prediction": prediction,
			}
			f.write(json.dumps(result, ensure_ascii=False) + "\n")
			f.flush()


if __name__ == "__main__":
	main()
