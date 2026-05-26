import json
import os
import random
import re

import openai
from tqdm import tqdm

# --- Configuration ---
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:2026/v1")

DATA_PATH = "../data/openseek-8_kernel_generation.json"
OUTPUT_PATH = "../outputs/result/openseek-8-v1.jsonl"
TASK_ID = 8
MAX_RETRY_TIMES = 5
STAGE_COUNT = 2
REFERENCE_COUNT = 8
REFERENCE_MIN_LEN = 300
REFERENCE_MAX_CHARS = 1200

TRAILING_NOTE = "After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate."


def clean_input_text(input_text: str) -> str:
    if not input_text:
        return ""
    cleaned = input_text.replace(TRAILING_NOTE, "")
    return cleaned.rstrip()

def extract_input_sections(input_text: str) -> dict:
    cleaned_input = clean_input_text(input_text).strip()

    # Use regex to safely extract sections
    desc_match = re.search(r'Functional Description:\s*(.*?)(?=\nWrapper Entry Information:|$)', cleaned_input, re.DOTALL)
    wrap_match = re.search(r'Wrapper Entry Information:\s*(.*?)(?=\nMath:|$)', cleaned_input, re.DOTALL)
    math_match = re.search(r'Math:\s*(.*?)(?=\nother:|$)', cleaned_input, re.DOTALL | re.IGNORECASE)
    other_match = re.search(r'other:\s*(.*)$', cleaned_input, re.DOTALL | re.IGNORECASE)

    desc_text = desc_match.group(1).strip() if desc_match else ""
    wrap_text = wrap_match.group(1).strip() if wrap_match else ""
    math_text = math_match.group(1).strip() if math_match else ""
    other_text = other_match.group(1).strip() if other_match else ""

    # Fallback for samples that don't include the explicit section headers.
    # In that case, treat the whole input as the functional description so Stage 1/2 prompts are still informative.
    if not desc_text and not wrap_text and not math_text:
        desc_text = cleaned_input

    return {
        "desc": desc_text,
        "wrap": wrap_text,
        "math": math_text,
        "other": other_text,
    }


def _extract_example_output_text(example: dict) -> str:
    output_value = example.get("output", "")
    if isinstance(output_value, list):
        return str(output_value[0]).strip() if output_value else ""
    return str(output_value).strip()


def _extract_example_input_text(example: dict) -> str:
    return str(example.get("input", "")).strip()


def select_output_references(examples: list, count: int = REFERENCE_COUNT) -> list:
    """Randomly select count I/O examples with suitable output length, favoring medium-size snippets."""
    records = []
    for example in examples:
        input_text = _extract_example_input_text(example)
        text = _extract_example_output_text(example)
        if not text:
            continue
        records.append({
            "id": example.get("id", "unknown"),
            "input": input_text,
            "output": text,
            "length": len(text),
        })

    if not records:
        return []

    lengths = sorted(r["length"] for r in records)
    median_len = lengths[len(lengths) // 2]
    low = max(REFERENCE_MIN_LEN, int(median_len * 0.5))
    high = int(median_len * 1.5)

    suitable = [r for r in records if low <= r["length"] <= high]
    if len(suitable) < count:
        suitable = [r for r in records if r["length"] >= REFERENCE_MIN_LEN]
    if len(suitable) < count:
        suitable = records

    random.shuffle(suitable)
    return suitable[: min(count, len(suitable))]


def format_output_references_for_prompt(references: list) -> str:
    if not references:
        return ""

    blocks = []
    for i, ref in enumerate(references, 1):
        input_text = ref.get("input", "")
        output_text = ref.get("output", "")
        if len(output_text) > REFERENCE_MAX_CHARS:
            output_text = output_text[:REFERENCE_MAX_CHARS].rstrip() + "\n# ... truncated ..."
        blocks.append(
            f"Few-shot Example {i} (id={ref['id']}, output_len={ref['length']}):\n"
            f"Input:\n"
            f"```text\n{input_text}\n```\n"
            f"Output:\n"
            f"```python\n{output_text}\n```"
        )

    return "\n\n".join(blocks)


def build_stage_prompt(stage: int, definition: str, sections: dict, previous_summary: str, output_reference_text: str) -> str:
    """Build 2-stage prompts:
    - Stage 1: integrate the three core sections and output runnable code (fenced python block).
    - Stage 2: given the initial code from stage1, modify and validate it against the sample's sections and return the final runnable code (fenced python block).
    """
    base = (
        "# Role\n"
        "You are a Senior GPU Kernel Engineer specializing in Triton and PyTorch autograd.\n\n"
    )

    if stage == 1:
        return (
            base
            + "# Stage 1 Objective\n"
            + "Integrate the following three core sections into a single runnable implementation (Triton + PyTorch autograd where applicable).\n"
            + "Produce only runnable code enclosed in a ```python fenced block```. Do not add explanations.\n\n"
            + ("# Few-shot Examples (Input+Output)\n" + output_reference_text + "\n\n" if output_reference_text else "")
            + "# Wrapper Entry Information\n"
            + f"{sections.get('wrap','')}\n\n"
            + "# Math\n"
            + f"{sections.get('math','')}\n\n"
            + "# Functional Description\n"
            + f"{sections.get('desc','')}\n\n"
            + "If any placeholder values are required, use reasonable defaults but mark them clearly in comments.\n"
        )

    # stage == 2: previous_summary now contains the initial code produced by stage1
    return (
        base
        + "# Stage 2 Objective\n"
        + "You are given an INITIAL implementation (from Stage 1) below. Your job: modify and validate that code specifically for the current test sample.\n"
        + "Make minimal, targeted fixes to ensure the code matches the task's Functional Description and the other provided information for this sample.\n"
        + "Use the sample's test-facing details as the primary validation signal. Add/adjust unit-test-like checks as comments if helpful, but output only final runnable code in a ```python fenced block```.\n\n"
        + "# INITIAL CODE (from Stage 1)\n"
        + "```python\n"
        + f"{previous_summary}\n"
        + "```\n\n"
        + ("# Few-shot Examples (Input+Output)\n" + output_reference_text + "\n\n" if output_reference_text else "")
        + "# Functional Description\n"
        + f"{definition}\n\n"
        + "# Other Information\n"
        + f"{sections.get('other','')}\n\n"
        + "Perform lightweight checks (e.g., matching func signature, inputs shape expectations). Fix code where necessary.\n"
        + "Output final runnable code only, enclosed in a single ```python block```.\n"
    )


def extract_summary_text(text: str) -> str:
    """Extract concise summary text from model response after thinking."""
    if not text:
        return ""

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Remove fenced blocks if any; stage 1 should be plain text.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    summary = " ".join(lines)
    if len(summary) > 400:
        summary = summary[:400].rstrip()
    return summary
def count_answer(text: str):
    """
    Matched exactly from prompt.md logic for task_id == 8
    """
    if not text:
        return "no"

    # Logic 1: when </think> exists, prioritize the post-think content.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Task 8: Kernel Generation (Code block)
    # 保留 ```python 格式的代码块匹配，并选用最后出现的一个（必须以 ``` 结尾闭合）
    block_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
    matches = block_pattern.findall(text)
    if matches:
        return matches[-1].strip()
    
    # 尝试匹配从 import 开始到 ``` 结束的内容（同样必须闭合）
    import_pattern = re.compile(r'(import\s+.*?)```', re.DOTALL)
    import_matches = import_pattern.findall(text)
    if import_matches:
        return import_matches[-1].strip()

    return "no"

def call_llm(prompt_text: str) -> str:
    """Call the local OpenAI-compatible model endpoint."""
    try:
        response = client.chat.completions.create(
            model="/root/ascend/log/debug/Qwen3-4B",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Senior GPU Kernel Engineer specializing in Triton and PyTorch autograd.",
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.6,
            top_p=0.85,
            max_tokens=8000,
        )
        whole_result = response.choices[0].message.content
        print(f"\n[Model Response] Task {TASK_ID}:\n{whole_result}\n" + "=" * 50)
        return whole_result
    except Exception as e:
        print(f"Exception occurred for Task {TASK_ID}: {e}")
        return ""


def call_llm_until_think_complete(prompt_text: str, sample_id: str, stage: int) -> str:
    retry_count = 0
    while True:
        response_raw = call_llm(prompt_text)
        if "</think>" in response_raw:
            return response_raw
        retry_count += 1
        if retry_count > MAX_RETRY_TIMES:
            print(
                f"[Retry] Task {TASK_ID}, sample {sample_id}, stage {stage}: "
                f"retry limit {MAX_RETRY_TIMES} exceeded, use latest response"
            )
            return response_raw
        print(
            f"[Retry] Task {TASK_ID}, sample {sample_id}, stage {stage}: "
            f"missing </think>, retrying ({retry_count})"
        )

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    definition = dataset.get("Definition", [""])[0]
    examples = dataset.get("examples", [])
    test_samples = dataset.get("test_samples", [])

    output_references = select_output_references(examples, REFERENCE_COUNT)
    output_reference_text = format_output_references_for_prompt(output_references)
    print(
        f"[Task {TASK_ID}] selected {len(output_references)} output references for prompting: "
        + ", ".join(r["id"] for r in output_references)
    )

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
            sections = extract_input_sections(input_text)

            #print(f"\n{'='*20} PROMPT SENT TO MODEL {'='*20}")
            #print(prompt_text)
            #print(f"{'='*60}\n")

            stage1_prompt = build_stage_prompt(1, definition, sections, "", output_reference_text)
            stage1_raw = call_llm_until_think_complete(stage1_prompt, sample_id, 1)
            # Stage 1 should return initial runnable code (fenced python). Extract it.
            initial_code = count_answer(stage1_raw)
            if not initial_code or initial_code == "no":
                # fallback: also try to extract summary if code extraction failed
                initial_code = extract_summary_text(stage1_raw)

            stage2_prompt = build_stage_prompt(2, definition, sections, initial_code, output_reference_text)
            stage2_raw = call_llm_until_think_complete(stage2_prompt, sample_id, 2)
            stage_code = count_answer(stage2_raw)

            prediction = stage_code if stage_code else "no"

            result = {
                "test_sample_id": sample_id,
                "prediction": prediction
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

if __name__ == "__main__":
    main()