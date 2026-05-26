import argparse
import ast
import json
import os
import re
from pathlib import Path

from main import TASK_FILES
from method import (
    DEFAULT_COMPLETION_URL,
    DEFAULT_REQUEST_TIMEOUT,
    TASK_MAX_TOKENS,
    TASK_STOPS,
    build_prompt,
    count_answer,
    get_served_model_name,
    sanitize_task8_code,
    select_examples,
)
from task8_retrieval import (
    build_task8_lexical_retrieval_context,
    extract_expected_signature,
    reorder_task8_examples_by_family_lexical_retrieval,
    reorder_task8_examples_by_lexical_retrieval,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Existing task8 jsonl file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write the updated task8 file.")
    parser.add_argument(
        "--only_ids",
        type=str,
        nargs="+",
        required=True,
        help="Explicit task8 sample ids to rerun with retrieval-based example selection.",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=100,
        help="Maximum number of task8 examples available for retrieval.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=128000,
        help="Maximum prompt token length before skipping a sample.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer path for prompt length checks.",
    )
    parser.add_argument(
        "--sample_attempts",
        type=int,
        default=1,
        help="Number of generation attempts per selected sample.",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="lexical",
        choices=["lexical", "family_lexical"],
        help="Retrieval strategy used to order task8 examples for reruns.",
    )
    return parser.parse_args()

def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task8_quality_score(code: str, expected_name: str | None) -> int:
    if not code:
        return -100

    score = 0
    if "@triton.jit" in code:
        score += 3
    if "import torch" in code and "import triton" in code:
        score += 2
    if re.search(r"(?m)^def\s+\w+\s*\(", code):
        score += 2
    if expected_name and re.search(rf"(?m)^def\s+{re.escape(expected_name)}\s*\(", code):
        score += 4
    if ".shape[" in code:
        score -= 1
    if any(token in code[:160] for token in ("Answer:", "<label>", "</label>", "<|")):
        score -= 6
    if len(code.strip()) < 200:
        score -= 3
    try:
        ast.parse(code)
        score += 3
    except SyntaxError:
        score -= 8
    return score


def request_single_completion(input_prompt: str) -> str:
    import requests

    payload = {
        "model": get_served_model_name(),
        "prompt": input_prompt,
        "max_tokens": TASK_MAX_TOKENS.get(8, 12000),
        "temperature": 0.0,
        "top_p": 1.0,
    }
    stop = TASK_STOPS.get(8)
    if stop:
        payload["stop"] = stop

    resp = requests.post(
        DEFAULT_COMPLETION_URL,
        json=payload,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def main():
    args = parse_args()
    task_dict = json.loads(TASK_FILES[8].read_text(encoding="utf-8"))
    examples = task_dict["examples"][: args.examples_limit]
    retrieval_context = build_task8_lexical_retrieval_context(examples)
    test_samples = {sample["id"]: sample for sample in task_dict["test_samples"]}

    rows = load_rows(Path(args.input_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(args.input_file).name

    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    target_ids = set(args.only_ids)
    rerun_count = 0
    updated_count = 0

    for row in rows:
        sample_id = row["test_sample_id"]
        if sample_id not in target_ids:
            continue

        sample = test_samples[sample_id]
        expected_name, _ = extract_expected_signature(sample["input"])
        text2annotate = sample["input"]
        if args.retrieval_mode == "family_lexical":
            ordered_examples = reorder_task8_examples_by_family_lexical_retrieval(
                examples,
                text2annotate,
                retrieval_context,
            )
        else:
            ordered_examples = reorder_task8_examples_by_lexical_retrieval(
                examples,
                text2annotate,
                retrieval_context,
            )
        prompt = build_prompt(task_dict["Definition"][0], text2annotate, task_id=8)
        examples_str = select_examples(
            ordered_examples,
            task_dict["Definition"][0],
            text2annotate,
            task_id=8,
        )
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + "\n\n")

        if tokenizer is not None:
            tokenized_input = tokenizer(input_prompt, return_tensors="pt")
            if tokenized_input["input_ids"].shape[1] > args.max_input_length:
                continue

        prediction = None
        for _ in range(max(1, args.sample_attempts)):
            try:
                raw = request_single_completion(input_prompt)
            except Exception:
                raw = ""
            prediction = count_answer(raw, task_id=8)
            if prediction is not None:
                break

        rerun_count += 1
        current_prediction = sanitize_task8_code(row.get("prediction") or "")
        candidate_prediction = sanitize_task8_code(prediction or "")
        if (
            candidate_prediction
            and candidate_prediction != row.get("prediction")
            and task8_quality_score(candidate_prediction, expected_name)
            > task8_quality_score(current_prediction, expected_name)
        ):
            row["prediction"] = candidate_prediction
            updated_count += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_file": str(args.input_file),
                "output_file": str(output_path),
                "requested_ids": len(target_ids),
                "rerun_count": rerun_count,
                "updated_count": updated_count,
                "retrieval_mode": args.retrieval_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
