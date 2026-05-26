import json
import os
import re
import random
from typing import List, Dict
import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-7_jeopardy_answer_generation_all.json"
OUTPUT_PATH = "../outputs/result/openseek-7-v1.jsonl"
GLOBAL_DYNAMIC_RULES = None
TASK_ID = 7

def extract_dynamic_rules(dataset: dict) -> str:
    """
    Step 1: Uses the LLM to inspect the training examples and infer formatting rules dynamically.
    """
    rule_dir = r"/root/ascend/log/debug/LongContext-ICL-Annotation/outputs/rule"
    os.makedirs(rule_dir, exist_ok=True)
    rule_file_path = os.path.join(rule_dir, "openseek-7-rules.txt")
    
    # Check if final rules already exist
    if os.path.exists(rule_file_path):
        with open(rule_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "=== FINAL TOP 5 RULES ===" in content:
                print("Found existing final rules in file. Using them directly without regeneration.")
                final_rules = content.split("=== FINAL TOP 5 RULES ===")[-1].strip()
                if final_rules:
                    return final_rules

    print("Analyzing examples to extract global prompt rules (generating 10 times)...")
    all_examples = dataset.get("examples", [])
    sampled_examples = random.sample(all_examples, min(500, len(all_examples)))
    
    examples_str = ""
    for ex in sampled_examples:
        val = ex['output'][0] if isinstance(ex['output'], list) else ex['output']
        examples_str += f"Clue: {ex['input']}\nAnswer: {val}\n\n"
        
    prompt_text = (
        "Analyze the following Jeopardy clues and their answers:\n\n"
        f"{examples_str}\n"
        "Based on these examples, infer the underlying rules and patterns. Specifically, focus on:\n"
        "1. The relationship between the Input and the Output .\n"
        "2. The tendencies/preferences of the answers.\n"
        "3. The characteristics of the answers themselves.\n\n"
        "Output your inferred rules as a concise numbered list in English. Keep each rule to a single short sentence. Remind yourself to always answer directly without analysis."
    )
    
    all_extracted_rules = []
    times = 10
    with open(rule_file_path, "w", encoding="utf-8") as f:
        for i in range(times):
            print(f"Generating rule set {i+1}/{times}...")
            try:
                response = client.chat.completions.create(
                    model="/root/ascend/log/debug/Qwen3-4B",
                    messages=[
                        {"role": "system", "content": "You are a data pattern analyzer."},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=2000
                )
                raw = response.choices[0].message.content
                clean_rules = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                f.write(f"--- Rule Set {i+1} ---\n{clean_rules}\n\n")
                
                # Parse individual numbered rules
                for line in clean_rules.split('\n'):
                    match = re.match(r'^\d+\.\s*(.+)', line.strip())
                    if match:
                        all_extracted_rules.append(match.group(1).strip())
            except Exception as e:
                print(f"Failed on iteration {i+1}: {e}")

    # Select top 5 repeated rules using the LLM for semantic clustering
    print("Summarizing the 10 rule sets to find the 5 most common rules...")
    aggregation_prompt = (
        "Here are several sets of rules generated for Jeopardy answers. "
        "Many of these rules express the same concept in different words. "
        "Identify the 5 most common and repeated underlying rules across all these sets.\n\n"
        "Rule Sets:\n" + "\n".join(all_extracted_rules) + "\n\n"
        "Output ONLY a numbered list of the top 5 most common rules, written clearly and concisely as statements."
    )
    
    try:
        agg_response = client.chat.completions.create(
            model="/root/ascend/log/debug/Qwen3-4B",
            messages=[
                {"role": "system", "content": "You are a data pattern analyzer."},
                {"role": "user", "content": aggregation_prompt}
            ],
            temperature=0.3,
            top_p=0.95,
            max_tokens=1000
        )
        agg_raw = agg_response.choices[0].message.content
        final_rules_str = re.sub(r'<think>.*?</think>', '', agg_raw, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"Failed to cluster rules via LLM: {e}")
        final_rules_str = ""
        
    if not final_rules_str:
        final_rules_str = "1. Output only the final answer in lowercase."

    with open(rule_file_path, "a", encoding="utf-8") as f:
        f.write("=== FINAL TOP 5 RULES ===\n")
        f.write(final_rules_str + "\n")
        
    return final_rules_str

def build_custom_icl_prompt(task_id: int, dataset: dict, input_text: str, dynamic_rules: str) -> str:
    """
    Improved Jeopardy prompt building with dynamic extracted rules and strict output anchoring.
    """
    definition = dataset.get("Definition", [""])[0]

    system_instr = (
        "Task Definition:\n"
        f"{definition}\n\n"
        "Inferred Rules from Examples:\n"
        f"{dynamic_rules}\n\n"
        "Strict Formatting Rule:\n"
        "EXACT OUTPUT: You must end your response with: 'The Answer is: [Your Answer]'.\n\n"
    )

    target_prompt = (
        "Target:\n"
        f"Input (Category & Clue): {input_text}\n\n"
        "Step-by-step reasoning:\n"
        "1. Identify the 'category' from the input, which represents the overarching topic.\n"
        "2. Analyze the 'clue' part to extract specific factual hints and constraints.\n"
        "3. Synthesize the hints to infer the exact answer that strongly fits both the category topic and the specific clue conditions.\n"
        "You must end your response with: 'The Answer is: [Your Answer]'.\n"
        "The Answer is: "
    )
    
    return system_instr + target_prompt

def call_llm(prompt_text: str) -> str:
    """Implement logic from prompt.md"""
    try:
        response = client.chat.completions.create(
            model="/root/ascend/log/debug/Qwen3-4B",
            messages=[
                {"role": "system", "content": "You are a Jeopardy expert."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.9, 
            top_p=0.95,
            max_tokens=2000,
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
                    
    # Generate dynamic rules based on training set once per run
    global GLOBAL_DYNAMIC_RULES
    if GLOBAL_DYNAMIC_RULES is None:
        GLOBAL_DYNAMIC_RULES = extract_dynamic_rules(dataset)
        
    dynamic_rules = GLOBAL_DYNAMIC_RULES

    print(f"\n{'='*20} DYNAMICALLY INFERRED RULES {'='*20}")
    print(dynamic_rules)
    print(f"{'='*60}\n")

    # Append missing ones
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
        for sample in tqdm(test_samples):
            sample_id = sample['id']
            if sample_id in results_dict:
                continue
            
            input_text = sample.get("input", "")
            prompt_text = build_custom_icl_prompt(TASK_ID, dataset, input_text, dynamic_rules)

            # Print the final prompt sent to the model for debugging
            print(f"\n{'='*20} PROMPT SENT TO MODEL (TASK {TASK_ID}) {'='*20}")
            print(prompt_text)
            print(f"{'='*60}\n")
            
            response_raw = call_llm(prompt_text)
            
            post_think_text = response_raw.split("</think>")[-1].strip() if "</think>" in response_raw else response_raw.strip()
            
            # Focused matching: look for "The Answer is: ..." first
            match = re.search(r"Answer is:\s*(.*)", post_think_text, re.IGNORECASE)
            if match:
                prediction = match.group(1).strip().strip('."\'')
            else:
                # Fallback to extract from the thinking process or raw response
                from collections import Counter
                pattern = re.compile(r'(?:\bis\b|\bAnswer:|\bshould\s+be\b)\s+["\']?(.*?)["\']?(?:\.|\"|\n|$)', re.IGNORECASE)
                matches = pattern.findall(response_raw)
                valid_matches = [m.strip().lower() for m in matches if m.strip()]
                
                if valid_matches:
                    counts = Counter(valid_matches)
                    max_count = max(counts.values())
                    candidates = set(k for k, v in counts.items() if v == max_count)
                    # Count occurrences, if tied, take the last one that appeared
                    for m in reversed(valid_matches):
                        if m in candidates:
                            prediction = m
                            break
                else:
                    # Last resort
                    lines = [l for l in post_think_text.split('\n') if l.strip()]
                    prediction = lines[-1].strip().strip('."\'') if lines else "no"
            
            # Clean up potential "The " prefix in Jeopardy style if model over-outputs
            prediction = re.sub(r"^(The Answer is:|Answer:)", "", prediction, flags=re.IGNORECASE).strip()

            result = {
                "test_sample_id": sample_id,
                "prediction": prediction
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results_dict[sample_id] = result
            
    # Rewrite the whole file to ensure answers are strictly ordered exactly as in test_samples
    # and there are absolutely no empty lines in the middle
    print("\nEnsuring answers are correctly ordered and fixing any missing/empty spaces...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            sample_id = sample['id']
            if sample_id in results_dict:
                f.write(json.dumps(results_dict[sample_id], ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
