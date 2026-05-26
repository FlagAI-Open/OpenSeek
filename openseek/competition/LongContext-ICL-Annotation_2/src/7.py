import json
import os
import re
import random
from typing import List, Dict
import numpy as np
import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-7_jeopardy_answer_generation_all.json"
OUTPUT_PATH = "../outputs/result/openseek-7-v1.jsonl"
TASK_ID = 7
SHOT_COUNT = 8
MAX_THINK_RETRIES = 3

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE = "cpu"
EMBED_BATCH_SIZE = 8
EMBED_POOL_SIZE = 108
EMBED_DEBUG = True

_EMBEDDER = None
_EXAMPLE_INDEX = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    return _EMBEDDER


def _build_example_records(examples: list) -> list:
    records = []
    for example in examples:
        input_text = str(example.get("input", "")).strip()
        output_value = example.get("output", "")
        if isinstance(output_value, list):
            output_text = str(output_value[0]).strip() if output_value else ""
        else:
            output_text = str(output_value).strip()
        if not input_text or not output_text:
            continue
        records.append({
            "id": example.get("id", "unknown"),
            "input": input_text,
            "output": output_text,
        })
    return records


def _ensure_example_index(examples: list):
    global _EXAMPLE_INDEX
    if _EXAMPLE_INDEX is not None:
        return _EXAMPLE_INDEX

    records = _build_example_records(examples)
    if not records:
        _EXAMPLE_INDEX = (records, None)
        return _EXAMPLE_INDEX

    try:
        model = _get_embedder()
        texts = [r["input"] for r in records]
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        if EMBED_DEBUG:
            print(
                "[Embedding] Built example index: "
                f"records={len(records)}, embeddings_shape={embeddings.shape}"
            )
    except Exception as exc:
        print(f"[Embedding] Failed to build example index: {exc}")
        embeddings = None

    _EXAMPLE_INDEX = (records, embeddings)
    return _EXAMPLE_INDEX


def select_few_shot_examples(examples: list, count: int = SHOT_COUNT, query_text: str = "") -> list:
    """Randomly sample candidates, then rank by embedding similarity and keep top-k."""
    records, embeddings = _ensure_example_index(examples)
    if not records or count <= 0:
        return []

    pool_size = min(len(records), max(count, EMBED_POOL_SIZE))
    if pool_size == len(records):
        candidate_indices = list(range(len(records)))
    else:
        candidate_indices = random.sample(range(len(records)), k=pool_size)
    candidate = [(idx, records[idx]) for idx in candidate_indices]

    top_k = min(count, len(candidate))
    if query_text and embeddings is not None:
        try:
            if EMBED_DEBUG:
                print(
                    "[Embedding] Using embedding ranking: "
                    f"candidates={len(candidate)}, top_k={top_k}"
                )
            model = _get_embedder()
            query_vec = model.encode(
                [query_text],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            query_vec = np.asarray(query_vec, dtype="float32")[0]
            candidate_embeddings = embeddings[[idx for idx, _ in candidate]]
            scores = candidate_embeddings @ query_vec
            order = np.argsort(-scores)
            return [candidate[int(i)][1] for i in order[:top_k]]
        except Exception as exc:
            print(f"[Embedding] Failed to rank examples: {exc}")

    if EMBED_DEBUG:
        reason = "missing query_text" if not query_text else "embeddings unavailable"
        print(f"[Embedding] Falling back to random selection ({reason}).")

    return [r for _, r in candidate[:top_k]]


def format_few_shot_examples(examples: list) -> str:
    if not examples:
        return ""
    blocks = []
    for ex in examples:
        blocks.append(
            f"Input (Category & Clue): {ex['input']}\n"
            f"Answer: {ex['output']}\n"
        )
    return "\n".join(blocks)


def _extract_last_answer_line(text: str) -> str:
    """Extract the last meaningful line as final answer, preferring post-</think> region."""
    if not text:
        return ""

    region = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
    lines = [ln.strip() for ln in region.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Drop code fence-only lines if any
    lines = [ln for ln in lines if ln not in {"```", "```text", "```python"}]
    if not lines:
        return ""

    last_line = lines[-1]
    last_line = re.sub(r"^(the answer is:|answer:)\s*", "", last_line, flags=re.IGNORECASE)
    return last_line.strip().strip('"\'`.,;:! ').lower()


def count_answer(text: str, task_id: int = 7):
    """Extract the final answer for task 7, preferring content after </think>."""
    if not text:
        return "no"

    # If </think> present, extract from the suffix only
    if "</think>" in text:
        post_think = text.split("</think>")[-1].strip()
        # Try explicit 'The Answer is:' first
        match = re.search(r"The Answer is:\s*(.+)", post_think, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('."\'')

        # Prefer final answer line right after thinking; this avoids being misled by earlier analysis mentions.
        last_line_answer = _extract_last_answer_line(text)
        if last_line_answer:
            return last_line_answer

        # Otherwise try common answer patterns in the post-think text
        pattern = re.compile(r'(?:\bis\b|\bAnswer:|\bshould\s+be\b)\s*[:]*\s*["\']?(.*?)["\']?(?:\.|"|\n|$)', re.IGNORECASE)
        matches = pattern.findall(post_think)
        matches = [m.strip().lower() for m in matches if m.strip()]
        if matches:
            from collections import Counter
            counts = Counter(matches)
            max_count = max(counts.values())
            candidates = set(k for k, v in counts.items() if v == max_count)
            for m in reversed(matches):
                if m in candidates:
                    return m

        # Last resort: return the last non-empty line from post_think
        lines = [l for l in post_think.split('\n') if l.strip()]
        return lines[-1].strip().strip('."\'') if lines else "no"

    # If no </think>, search whole response similarly
    pattern = re.compile(r'(?:\bis\b|\bAnswer:|\bshould\s+be\b)\s*[:]*\s*["\']?(.*?)["\']?(?:\.|"|\n|$)', re.IGNORECASE)
    matches = pattern.findall(text)
    matches = [m.strip().lower() for m in matches if m.strip()]
    if matches:
        from collections import Counter
        counts = Counter(matches)
        max_count = max(counts.values())
        candidates = set(k for k, v in counts.items() if v == max_count)
        for m in reversed(matches):
            if m in candidates:
                return m

    # Fallback to last non-empty line
    lines = [l for l in text.split('\n') if l.strip()]
    return lines[-1].strip().strip('."\'') if lines else "no"


def build_reflection_prompt(dataset: dict, input_text: str, candidate_answer: str) -> str:
    """Build a second-pass prompt to verify whether candidate answer best matches the clue."""
    examples = dataset.get("examples", [])
    ref_examples = select_few_shot_examples(examples, SHOT_COUNT, query_text=input_text)
    ref_block = ""
    if ref_examples:
        ref_block = "Few-shot Examples:\n" + format_few_shot_examples(ref_examples) + "\n\n"
    return (
        ref_block +
        "Task Definition:\n"
        #f"{definition}\n\n"
        "Reflection Task:\n"
        "You are given a Jeopardy category+clue and a candidate answer.\n"
        #"Do not judge or explain whether it is correct.\n"
        "Directly correct the candidate to the best final answer for the clue.\n"
        "If the candidate is already correct, return it unchanged.\n"
        "Output only one final answer in lower case.\n\n"
        f"Input (Category & Clue): {input_text}\n"
        f"Candidate Answer: {candidate_answer}\n\n"
        "Give the final answer directly.\n"
    )


def reflect_and_revise_answer(dataset: dict, sample_id: str, input_text: str, candidate_answer: str) -> str:
    """Run reflection for task 7 and revise answer when model gives a better one."""
    review_prompt = build_reflection_prompt(dataset, input_text, candidate_answer)
    review_raw = call_llm_until_think_complete(review_prompt, sample_id)
    revised_answer = _extract_last_answer_line(review_raw) or count_answer(review_raw, TASK_ID)

    if revised_answer and revised_answer != "no":
        revised_answer = re.sub(r"^(the answer is:|answer:)", "", revised_answer, flags=re.IGNORECASE).strip().lower()

    normalized_candidate = (candidate_answer or "").strip().lower()
    if revised_answer and revised_answer != "no" and revised_answer != normalized_candidate:
        print(
            f"[Reflect] Task {TASK_ID}, sample {sample_id}: "
            f"updated answer from '{candidate_answer}' to '{revised_answer}'"
        )
        return revised_answer

    return candidate_answer


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    return re.sub(r"^(the answer is:|answer:)", "", str(answer), flags=re.IGNORECASE).strip().lower()


def persist_results_jsonl(results_dict: dict, test_samples: list) -> None:
    """Write the current ordered results back to the JSONL file immediately."""
    ordered_results = []
    for sample in test_samples:
        sample_id = sample["id"]
        if sample_id in results_dict:
            ordered_results.append(results_dict[sample_id])

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in ordered_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def call_llm_until_think_complete(prompt_text: str, sample_id: str) -> str:
    """Retry call when </think> is missing, up to MAX_THINK_RETRIES."""
    retry_count = 0
    while True:
        response_raw = call_llm(prompt_text)
        if "</think>" in response_raw:
            return response_raw

        retry_count += 1
        if retry_count > MAX_THINK_RETRIES:
            print(
                f"[Retry] Task {TASK_ID}, sample {sample_id}: "
                f"retry limit {MAX_THINK_RETRIES} exceeded, use latest response"
            )
            return response_raw

        print(
            f"[Retry] Task {TASK_ID}, sample {sample_id}: "
            f"missing </think>, retrying ({retry_count})"
        )

def build_custom_icl_prompt(task_id: int, dataset: dict, input_text: str) -> str:
    """
    Build a Jeopardy prompt with randomized few-shot examples and strict output anchoring.
    """
    definition = dataset.get("Definition", [""])[0]
    examples = dataset.get("examples", [])

    sampled_examples = select_few_shot_examples(examples, SHOT_COUNT, query_text=input_text)

    system_instr = (
        #"Task Definition:\n"
        #f"{definition}\n\n"
        "Answering Rules:\n"
        "You are a professional trivia assistant. Your task is to deduce the precise answer based on the provided Category and Clue.\n"
        
    )

    examples_str = "Few-shot Examples:\n" + format_few_shot_examples(sampled_examples) + "\n"

    target_prompt = (
        examples_str +
        
        "Treat the Clue as a fill-in-the-blank question.\n"
        
        "1. Identify demonstrative pronouns and interrogative anchors to locate the blank to be filled, or the specific answer not provided by the clue.\n"
        "2. The answer needs to be inferred and does not appear in the provided information.\n"
        "3. Based on the provided clues, category and related knowledge, combine the conditions to identify the answer and complete this fill-in-the-blank question.\n"
        
        #"4. Formatted Output: The final answer must be in **lowercase letters** and enclosed in square brackets `[]`. If there are multiple answers, separate them with commas.\n"
        
        
        
        "Target:\n"
        f"Input (Category & Clue): {input_text}\n\n"
        
        "Output answer:"
        "Follow the output format strictly: lowercase English entities only, comma-separated if needed, no explanation, no sentence, output the single correct answer.s.\n"
        #"If several answers fit, include all valid answers.\n"
        "Give the final answer directly.\n"
        
    )
    
    return system_instr + target_prompt


def call_llm(prompt_text: str) -> str:
    """Implement logic from prompt.md"""
    try:
        response = client.chat.completions.create(
            model="/root/ascend/log/debug/Qwen3-4B",
            messages=[
                {"role": "system", "content": "You are solving a clue-reasoning question."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3,
            top_p=0.98,
            max_tokens=5000,
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
    all_sample_ids = {sample['id'] for sample in test_samples}

    # Load existing results to support resume
    results_dict = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    results_dict[data['test_sample_id']] = data
                except:
                    continue

    completed_before_run = all_sample_ids.issubset(set(results_dict.keys()))
    run_review_mode = False
    if completed_before_run:
        user_choice = input(
            "All test samples are already present in openseek-7-v1.jsonl. Enter review mode now? (y/n): "
        ).strip().lower()
        run_review_mode = user_choice == "y"
        if not run_review_mode:
            print("Skip review mode by user choice (n).")
            return
                    
    if not completed_before_run:
        # Append missing ones
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            for sample in tqdm(test_samples):
                sample_id = sample['id']
                if sample_id in results_dict:
                    continue
                
                input_text = sample.get("input", "")
                prompt_text = build_custom_icl_prompt(TASK_ID, dataset, input_text)

                # Print the final prompt sent to the model for debugging
                #print(f"\n{'='*20} PROMPT SENT TO MODEL (TASK {TASK_ID}) {'='*20}")
                #print(prompt_text)
                #print(f"{'='*60}\n")
                
                # Use Task 4 style flow: wait for </think> and extract answer from post-think only
                response_raw = call_llm_until_think_complete(prompt_text, sample_id)
                prediction = count_answer(response_raw, TASK_ID)
                # Clean up potential verbose prefixes
                prediction = re.sub(r"^(the answer is:|answer:)", "", prediction, flags=re.IGNORECASE).strip()

                result = {
                    "test_sample_id": sample_id,
                    "prediction": prediction
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results_dict[sample_id] = result
        print("\nAll samples completed. Please run the script again to enter review mode.")
        return

    if run_review_mode:
        # Second pass: review each answer with clue + candidate and revise if needed.
        for sample in test_samples:
            sample_id = sample['id']
            input_text = sample.get("input", "")
            current_result = results_dict.get(sample_id)
            if not current_result:
                continue

            candidate_answer = normalize_answer(current_result.get("prediction", "no"))
            revised_answer = reflect_and_revise_answer(dataset, sample_id, input_text, candidate_answer)

            if revised_answer != candidate_answer:
                print(
                    f"[Update] Task {TASK_ID}, sample {sample_id}: "
                    f"writing revised answer '{revised_answer}' back to output jsonl"
                )

            results_dict[sample_id]["prediction"] = revised_answer if revised_answer else candidate_answer

        persist_results_jsonl(results_dict, test_samples)

    print("\nReview pass finished; results are already persisted during updates.")

if __name__ == "__main__":
    main()
