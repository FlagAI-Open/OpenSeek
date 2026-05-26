import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from method_task6 import annotate_nvidia as annotate
from method_task6 import build_prompt, select_examples

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "Qwen3-4B")

DEFAULT_TRACE_LINE_NUMBERS = ""
DEFAULT_ONLY_LINE_NUMBERS = ""

TASK_FILES = {
    1: "./data/openseek-1_closest_integers.json",
    2: "./data/openseek-2_count_nouns_verbs.json",
    3: "./data/openseek-3_collatz_conjecture.json",
    4: "./data/openseek-4_conala_concat_strings.json",
    5: "./data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    6: "./data/openseek-6_mnli_same_genre_classification.json",
    7: "./data/openseek-7_jeopardy_answer_generation_all.json",
    8: "../data/openseek-8_kernel_generation.json",
}


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--max_input_length", type=int, default=128_000)
    parser.add_argument("--log_path_prefix", type=str, default="./outputs/")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
    )
    parser.add_argument("--debug_first_n", type=int, default=10)
    parser.add_argument(
        "--trace_line_numbers",
        type=str,
        default=DEFAULT_TRACE_LINE_NUMBERS,
        help="Comma-separated 1-based line numbers to dump full reasoning into a sidecar jsonl file. Disabled by default.",
    )
    parser.add_argument(
        "--only_line_numbers",
        type=str,
        default=DEFAULT_ONLY_LINE_NUMBERS,
        help="Comma-separated 1-based line numbers to run exclusively.",
    )
    return parser.parse_args()


def _parse_line_number_set(raw: str) -> set[int]:
    result = set()
    if not raw:
        return result
    raw = raw.replace("，", ",")
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value > 0:
            result.add(value)
    return result


def _normalize_log_path_prefix(log_path_prefix: str, task_id: int) -> str:
    normalized = os.path.normpath(log_path_prefix)
    task_dirname = f"task_{task_id}"
    base_name = os.path.basename(normalized)
    if base_name == task_dirname:
        return normalized
    if base_name.startswith("run_all_"):
        return os.path.join(normalized, task_dirname)
    return normalized


def evaluate(
    task_id: int,
    qwen_tokenizer: AutoTokenizer,
    max_input_length: int,
    log_path_prefix: str,
    debug_first_n: int,
    trace_line_numbers: set[int],
    only_line_numbers: set[int],
):
    assert task_id == 6, f"main_task6.py only supports task_id=6, but got {task_id}."
    log_path_prefix = _normalize_log_path_prefix(log_path_prefix, task_id)

    task_file = TASK_FILES[task_id]
    with open(task_file, "r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    icl_examples = task_dict.get("examples", [])
    test_samples = task_dict["test_samples"]

    os.makedirs(log_path_prefix, exist_ok=True)
    version = 1
    output_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}.jsonl")
    while os.path.exists(output_file):
        version += 1
        output_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}.jsonl")

    trace_file = None
    if trace_line_numbers:
        trace_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}-trace.jsonl")

    print(f"Starting Task {task_id}: {task_name}")
    print(f"Output will be saved to: {output_file}")
    if trace_file:
        print(f"Trace output will be saved to: {trace_file}")
    else:
        print("Trace output is disabled. Pass --trace_line_numbers to enable sidecar trace output.")
    if only_line_numbers:
        print(f"Only running line numbers: {sorted(only_line_numbers)}")

    for idx, test_sample in enumerate(tqdm(test_samples)):
        line_number = idx + 1
        if only_line_numbers and line_number not in only_line_numbers:
            continue

        test_record = {"test_sample_id": test_sample["id"]}
        text2annotate = test_sample["input"]
        should_trace = line_number in trace_line_numbers

        prompt = build_prompt(task_id, task_description, text2annotate)
        examples_str = select_examples(icl_examples, task_description, text2annotate)
        input_prompt = prompt.replace("[[EXAMPLES]]", examples_str)

        tokenized = qwen_tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False)
        if tokenized["input_ids"].shape[1] > max_input_length:
            test_record["prediction"] = None
            raw_output = "INPUT_TOO_LONG" if should_trace else None
        else:
            if should_trace or (debug_first_n and idx < debug_first_n) or line_number in only_line_numbers:
                prediction, raw_output = annotate(
                    input_prompt,
                    task_id=task_id,
                    debug=True,
                    text2annotate=text2annotate,
                )
                test_record["prediction"] = prediction
                if (debug_first_n and idx < debug_first_n) or line_number in only_line_numbers:
                    print(f"\n[DEBUG {line_number}] Input: {text2annotate}")
                    print(f"[RAW]: {raw_output}")
                    print(f"[PRED]: {prediction}\n" + "-" * 50)
            else:
                raw_output = None
                test_record["prediction"] = annotate(
                    input_prompt,
                    task_id=task_id,
                    debug=False,
                    text2annotate=text2annotate,
                )

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + "\n")

        if should_trace and trace_file:
            trace_record = {
                "line_number": line_number,
                "test_sample_id": test_sample["id"],
                "input": text2annotate,
                "prediction": test_record["prediction"],
                "raw_output": raw_output,
            }
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    evaluate(
        args.task_id,
        qwen_tokenizer,
        args.max_input_length,
        args.log_path_prefix,
        args.debug_first_n,
        _parse_line_number_set(args.trace_line_numbers),
        _parse_line_number_set(args.only_line_numbers),
    )
