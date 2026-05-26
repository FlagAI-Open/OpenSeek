import argparse
import json
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from method_task5 import annotate_nvidia as annotate
    from method_task5 import build_prompt, select_examples
    from method_task5 import extract_task5_trace
except ImportError as exc:
    raise ImportError("Run this script from flagos/src or with flagos/src on PYTHONPATH.") from exc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "Qwen3-4B")

DEFAULT_TRACE_LINE_NUMBERS = ""
DEFAULT_TRACE_SAMPLE_IDS = ""
DEFAULT_ONLY_LINE_NUMBERS = ""
DEFAULT_ONLY_SAMPLE_IDS = ""

TASK_FILES = {
    1: os.path.join(PROJECT_ROOT, "data", "openseek-1_closest_integers.json"),
    2: os.path.join(PROJECT_ROOT, "data", "openseek-2_count_nouns_verbs.json"),
    3: os.path.join(PROJECT_ROOT, "data", "openseek-3_collatz_conjecture.json"),
    4: os.path.join(PROJECT_ROOT, "data", "openseek-4_conala_concat_strings.json"),
    5: os.path.join(PROJECT_ROOT, "data", "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"),
    6: os.path.join(PROJECT_ROOT, "data", "openseek-6_mnli_same_genre_classification.json"),
    7: os.path.join(PROJECT_ROOT, "data", "openseek-7_jeopardy_answer_generation_all.json"),
    8: os.path.join(PROJECT_ROOT, "data", "openseek-8_kernel_generation.json"),
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
    parser.add_argument("--debug_first_n", type=int, default=0)
    parser.add_argument(
        "--trace_line_numbers",
        type=str,
        default=DEFAULT_TRACE_LINE_NUMBERS,
        help="Comma-separated 1-based line numbers to dump full reasoning into a sidecar jsonl file. Disabled by default.",
    )
    parser.add_argument(
        "--trace_sample_ids",
        type=str,
        default=DEFAULT_TRACE_SAMPLE_IDS,
        help="Comma-separated sample ids to dump full reasoning into a sidecar jsonl file. Disabled by default.",
    )
    parser.add_argument(
        "--only_line_numbers",
        type=str,
        default=DEFAULT_ONLY_LINE_NUMBERS,
        help="Comma-separated 1-based line numbers to run exclusively.",
    )
    parser.add_argument(
        "--only_sample_ids",
        type=str,
        default=DEFAULT_ONLY_SAMPLE_IDS,
        help="Comma-separated sample ids to run exclusively.",
    )
    return parser.parse_args()


def _parse_line_number_set(raw: str) -> set[int]:
    result = set()
    if not raw:
        return result
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


def _parse_id_set(raw: str) -> set[str]:
    result = set()
    if not raw:
        return result
    for item in raw.split(","):
        item = item.strip()
        if item:
            result.add(item)
    return result


def evaluate(
    task_id: int,
    qwen_tokenizer,
    max_input_length: int,
    log_path_prefix: str,
    debug_first_n: int,
    trace_line_numbers: set[int],
    trace_sample_ids: set[str],
    only_line_numbers: set[int],
    only_sample_ids: set[str],
):
    task_file = TASK_FILES[task_id]
    with open(task_file, "r", encoding="utf-8") as f:
        task_dict = json.load(f)

    task_name = task_dict["task_name"]
    task_description = task_dict["Definition"][0]
    icl_examples = task_dict.get("examples", [])[:100]
    test_samples = task_dict["test_samples"]

    os.makedirs(log_path_prefix, exist_ok=True)
    version = 1
    output_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}.jsonl")
    while os.path.exists(output_file):
        version += 1
        output_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}.jsonl")

    trace_file = None
    if trace_line_numbers or trace_sample_ids:
        trace_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}-trace.jsonl")

    print(f"Starting Task {task_id}: {task_name}")
    print(f"Output will be saved to: {output_file}")
    if trace_file:
        print(f"Trace output will be saved to: {trace_file}")
    else:
        print("Trace output is disabled. Pass --trace_line_numbers or --trace_sample_ids to enable sidecar trace output.")
    if only_line_numbers:
        print(f"Only running line numbers: {sorted(only_line_numbers)}")
    if only_sample_ids:
        print(f"Only running sample ids: {sorted(only_sample_ids)}")

    cached_examples_str = None

    for idx, test_sample in enumerate(tqdm(test_samples, desc=f"Evaluation on Task {task_id}: {task_name}")):
        line_number = idx + 1
        sample_id = test_sample["id"]

        if only_line_numbers and line_number not in only_line_numbers:
            continue
        if only_sample_ids and sample_id not in only_sample_ids:
            continue

        text2annotate = test_sample["input"]
        should_trace = (line_number in trace_line_numbers) or (sample_id in trace_sample_ids)
        test_record = {"test_sample_id": sample_id}

        prompt = build_prompt(task_id, task_description, text2annotate)
        if task_id == 5:
            cached_examples_str = select_examples(icl_examples, task_description, text2annotate)
        elif cached_examples_str is None:
            cached_examples_str = select_examples(icl_examples, task_description, text2annotate)
        input_prompt = prompt.replace("[[EXAMPLES]]", cached_examples_str)

        if qwen_tokenizer is not None:
            tokenized = qwen_tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False)
            input_length = tokenized["input_ids"].shape[1]
        else:
            input_length = max(1, len(input_prompt) // 4)

        if input_length > max_input_length:
            prediction = None
            raw_output = "INPUT_TOO_LONG" if should_trace else None
        else:
            need_debug = (
                should_trace
                or (debug_first_n and idx < debug_first_n)
                or line_number in only_line_numbers
                or sample_id in only_sample_ids
            )
            if need_debug:
                prediction, raw_output = annotate(
                    input_prompt,
                    task_id=task_id,
                    debug=True,
                    text2annotate=text2annotate,
                )
                debug_record = {
                    "line_number": line_number,
                    "test_sample_id": sample_id,
                    "input": text2annotate,
                    "prediction": prediction,
                    "raw_output": raw_output,
                }
                if task_id == 5 and raw_output:
                    debug_record["structured_trace"] = extract_task5_trace(raw_output)
            else:
                raw_output = None
                prediction = annotate(
                    input_prompt,
                    task_id=task_id,
                    debug=False,
                    text2annotate=text2annotate,
                )

        test_record["prediction"] = prediction

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + "\n")

        if should_trace and trace_file:
            trace_record = {
                "line_number": line_number,
                "test_sample_id": sample_id,
                "input": text2annotate,
                "prediction": prediction,
                "raw_output": raw_output,
            }
            if task_id == 5 and raw_output:
                trace_record["structured_trace"] = extract_task5_trace(raw_output)
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parser_args()
    qwen_tokenizer = None
    if AutoTokenizer is not None:
        qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    evaluate(
        args.task_id,
        qwen_tokenizer,
        args.max_input_length,
        args.log_path_prefix,
        args.debug_first_n,
        _parse_line_number_set(args.trace_line_numbers),
        _parse_id_set(args.trace_sample_ids),
        _parse_line_number_set(args.only_line_numbers),
        _parse_id_set(args.only_sample_ids),
    )
