import json
import os
import re
from collections import Counter

import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-4_conala_concat_strings.json"
OUTPUT_PATH = "../outputs/result/openseek-4-v1.jsonl"
TASK_ID = 4
MAX_RETRY_TIMES = 3


def build_icl_prompt(task_id: int, task_file_path: str, input_text: str) -> str:
    """Build the task prompt directly in this file."""
    with open(task_file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    definition = dataset.get("Definition", [""])[0]
    answer_desc = "the concatenated string is"

    base_prompt = f"Task Definition:\n{definition}\n\n"
    base_prompt += (
        "Rules:\n"
        "- Output ONLY the concatenated string.\n"
        "- Do NOT include any spaces, quotes, or box formatting in the final value.\n"
        "- Read the input carefully and concatenate the all strings exactly.\n\n"
        "- When concatenating string lists, do not add, delete, or modify any characters.\n"
        "- Keep the same case as the original list elements.\n"
        "- Concatenate from front to back in order, without changing the sequence.\n\n"
    )
    base_prompt += "You are a highly capable in merging strings.\n\n"

    base_prompt += "Examples:\n"
    for ex in dataset.get("examples", [])[-3:]:
        base_prompt += f"Input: {ex['input']}\n"
        output_str = ex["output"][0] if isinstance(ex["output"], list) else ex["output"]
        base_prompt += f"The input's concatenate string is {output_str}\n\n"

    target_prompt = "Now solve the target input.\n"
    target_prompt += (
        f"--- Target ---\nInput: {input_text}\n"
        f"Directly output the char's concatenated string.\n"
    )

    return base_prompt + target_prompt


def count_answer(text: str, task_id: int = 4):
    """Extract the final concatenated string for task 4."""
    if not text:
        return "no"

    # Logic 1: when </think> exists, extract answer after it.
    if "</think>" in text:
        post_think = text.split("</think>")[-1].strip()
        lines = [ln.strip().strip("'\" ") for ln in post_think.splitlines() if ln.strip()]
        return lines[0].replace(" ", "") if lines else "no"

    # Logic 2: when </think> does not exist, use regex matching only.
    pattern = re.compile(
        r'(?:\bis\b|\bAnswer:?|\bOutput:?|\bshould\s+be\b|\bthe\s+input\'s\s+concatenate\s+string\s+is\b)\s*[:*]*\s*["\']?(.*?)["\']?(?:\.|"|\n|$)',
        re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(text)

    if not matches:
        return "no"

    cleaned = []
    for m in matches:
        boxed_match = re.search(r'\\boxed\{(.*?)\}', m)
        if boxed_match:
            cleaned.append(boxed_match.group(1))
        else:
            cleaned.append(m)

    cleaned = [m.strip().strip("[]'\" ").strip() for m in cleaned if m.strip()]
    cleaned = [m.replace(" ", "") for m in cleaned]

    if not cleaned:
        return "no"

    counts = Counter(cleaned)
    max_count = max(counts.values())
    candidates = set(k for k, v in counts.items() if v == max_count)
    for m in reversed(cleaned):
        if m in candidates:
            return m

    return "no"


def build_reflection_prompt(definition: str, input_text: str, candidate_answer: str) -> str:
    """Ask the model to verify a candidate answer and correct it if needed."""
    return (
        f"Task Definition:\n{definition}\n\n"
        f"--- Target ---\nInput: {input_text}\n"
        f"Candidate Answer: {candidate_answer}\n\n"
        "Task:\n"
        "You need to check whether the candidate answer completely matches in sequence and consistent.\n"
        "Check strictly according to the following steps:\n"
        "1.Compare character by character one by one in sequence.\n"
        "2.if any character has case error, wrong character, extra character or missing character,then fix it and make sure the answer is correct.\n"
        "3.Only output the final corrected answer.\n"

        f"Directly output the char's concatenated string answer.\n"
    )


def reflect_and_revise_answer(definition: str, sample_id: str, input_text: str, candidate_answer: str) -> str:
    """Run a second review pass and replace the candidate when the model gives a different answer."""
    review_prompt = build_reflection_prompt(definition, input_text, candidate_answer)
    review_raw = call_llm_until_think_complete(review_prompt, sample_id)
    revised_answer = count_answer(review_raw, TASK_ID)

    if revised_answer and revised_answer != "no" and revised_answer != candidate_answer:
        print(
            f"[Reflect] Task {TASK_ID}, sample {sample_id}: "
            f"updated answer from '{candidate_answer}' to '{revised_answer}'"
        )
        return revised_answer

    return candidate_answer


def call_llm(prompt_text: str) -> str:
    """Call the local OpenAI-compatible model endpoint."""
    try:
        response = client.chat.completions.create(
            model="/root/ascend/log/debug/Qwen3-4B",
            messages=[
                {"role": "system", "content": "You are a highly capable in merging strings."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.5,
            top_p=0.85,
            max_tokens=5000,
        )
        whole_result = response.choices[0].message.content
        print(f"\n[Model Response] Task {TASK_ID}:\n{whole_result}\n" + "=" * 50)
        return whole_result
    except Exception as e:
        print(f"Exception occurred for Task {TASK_ID}: {e}")
        return ""


def call_llm_until_think_complete(prompt_text: str, sample_id: str) -> str:
    """Retry call when </think> is missing, up to MAX_RETRY_TIMES."""
    retry_count = 0
    while True:
        response_raw = call_llm(prompt_text)
        if "</think>" in response_raw:
            return response_raw

        retry_count += 1
        if retry_count > MAX_RETRY_TIMES:
            print(
                f"[Retry] Task {TASK_ID}, sample {sample_id}: "
                f"retry limit {MAX_RETRY_TIMES} exceeded, use latest response"
            )
            return response_raw

        print(
            f"[Retry] Task {TASK_ID}, sample {sample_id}: "
            f"missing </think>, retrying ({retry_count})"
        )


def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    definition = dataset.get("Definition", [""])[0]
    test_samples = dataset.get("test_samples", [])
    all_sample_ids = {sample["id"] for sample in test_samples}

    results_dict = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    results_dict[data["test_sample_id"]] = data
                except Exception:
                    continue

    completed_before_run = all_sample_ids.issubset(set(results_dict.keys()))
    run_review_mode = False
    if completed_before_run:
        user_choice = input(
            "All test samples are already present in openseek-4-v1.jsonl. Enter review mode now? (y/n): "
        ).strip().lower()
        run_review_mode = user_choice == "y"
        if not run_review_mode:
            print("Skip review mode by user choice (n). Exiting.")
            return

    if not completed_before_run:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for sample in tqdm(test_samples):
                sample_id = sample["id"]
                if sample_id in results_dict:
                    continue

                input_text = sample.get("input", "")
                prompt_text = build_icl_prompt(TASK_ID, DATA_PATH, input_text)
                response_raw = call_llm_until_think_complete(prompt_text, sample_id)
                prediction = count_answer(response_raw, TASK_ID)

                result = {
                    "test_sample_id": sample_id,
                    "prediction": prediction if prediction else "no",
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results_dict[sample_id] = result
        print("\nAll 500 samples completed. Please run the script again to enter review mode.")
        return

    if run_review_mode:
        # Second pass: reflect on the first-round answers and revise if the model changes its judgment.
        final_results = {}
        for sample in test_samples:
            sample_id = sample["id"]
            input_text = sample.get("input", "")
            current_result = results_dict.get(sample_id)
            if not current_result:
                continue

            candidate_answer = current_result.get("prediction", "no")
            revised_answer = reflect_and_revise_answer(definition, sample_id, input_text, candidate_answer)
            if revised_answer != candidate_answer:
                current_result["prediction"] = revised_answer
                results_dict[sample_id] = current_result

            final_results[sample_id] = {
                "test_sample_id": sample_id,
                "prediction": revised_answer if revised_answer else candidate_answer,
            }

        print("\nEnsuring answers are correctly ordered and rewriting with reflection updates...")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for sample in test_samples:
                sample_id = sample["id"]
                if sample_id in final_results:
                    f.write(json.dumps(final_results[sample_id], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
