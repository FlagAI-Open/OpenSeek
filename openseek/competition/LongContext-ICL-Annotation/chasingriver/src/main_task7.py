import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from method_task7 import annotate_nvidia as annotate
from method_task7 import classify_strategy_class_nvidia
from method_task7 import format_example
from method_task7 import build_prompt, select_examples
from method_task7 import extract_candidates_from_debug_output
from method_task7 import describe_candidate_transformations

try:
    from method_task7 import solve_deterministic
except ImportError:
    solve_deterministic = None

DEFAULT_TRACE_LINE_NUMBERS = ""
DEFAULT_ONLY_LINE_NUMBERS = DEFAULT_TRACE_LINE_NUMBERS

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
        default="Qwen3-4B",
    )
    parser.add_argument("--debug_first_n", type=int, default=0)
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


def _build_prompt_with_examples(
    task_id: int,
    task_description: str,
    text2annotate: str,
    retrieval_info: dict,
) -> str:
    prompt = build_prompt(
        task_id,
        task_description,
        text2annotate,
        answer_type_hint=retrieval_info["detected_answer_type"],
        route_summary=retrieval_info["route_summary"],
        strategy_class=retrieval_info["strategy_class"],
    )
    return prompt.replace("[[EXAMPLES]]", retrieval_info["examples_str"])


def _shrink_examples_to_fit(
    qwen_tokenizer: AutoTokenizer,
    max_input_length: int,
    task_id: int,
    task_description: str,
    text2annotate: str,
    retrieval_info: dict,
) -> tuple[str, int]:
    examples = list(retrieval_info.get("selected_examples", []))
    while True:
        retrieval_info["examples_str"] = "\n".join(format_example(example) for example in examples)
        retrieval_info["selected_example_ids"] = [example.get("id") for example in examples]
        retrieval_info["selected_example_count"] = len(examples)
        input_prompt = _build_prompt_with_examples(task_id, task_description, text2annotate, retrieval_info)
        tokenized = qwen_tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False)
        if tokenized["input_ids"].shape[1] <= max_input_length or not examples:
            return input_prompt, len(examples)
        examples = examples[:-1]


def evaluate(
    task_id: int,
    qwen_tokenizer: AutoTokenizer,
    max_input_length: int,
    log_path_prefix: str,
    debug_first_n: int,
    trace_line_numbers: set[int],
    only_line_numbers: set[int],
):
    assert task_id == 7, f"main_task7.py only supports task_id=7, but got {task_id}."
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
    candidates_file = os.path.join(log_path_prefix, f"openseek-{task_id}-v{version}-candidates.jsonl")

    print(f"Starting Task {task_id}: {task_name}")
    print(f"Output will be saved to: {output_file}")
    if trace_file:
        print(f"Trace output will be saved to: {trace_file}")
    else:
        print("Trace output is disabled. Pass --trace_line_numbers to enable sidecar trace output.")
    print(f"Candidate output will be saved to: {candidates_file}")
    if only_line_numbers:
        print(f"Only running line numbers: {sorted(only_line_numbers)}")

    for idx, test_sample in enumerate(tqdm(test_samples, desc=f"Evaluation on Task {task_id}: {task_name}")):
        line_number = idx + 1
        if only_line_numbers and line_number not in only_line_numbers:
            continue

        test_record = {"test_sample_id": test_sample["id"]}
        text2annotate = test_sample["input"]
        candidate_predictions: list[str] = []
        candidate_transformations: list[dict] = []
        should_trace = line_number in trace_line_numbers
        should_debug = should_trace or (debug_first_n and idx < debug_first_n) or line_number in only_line_numbers

        if should_debug:
            strategy_class, router_raw_output = classify_strategy_class_nvidia(
                text2annotate,
                all_examples=icl_examples,
                debug=True,
            )
        else:
            strategy_class = classify_strategy_class_nvidia(
                text2annotate,
                all_examples=icl_examples,
                debug=False,
            )
            router_raw_output = None

        retrieval_info = select_examples(
            icl_examples,
            task_description,
            text2annotate,
            tokenizer=qwen_tokenizer,
            strategy_class=strategy_class,
        )
        input_prompt, _ = _shrink_examples_to_fit(
            qwen_tokenizer,
            max_input_length,
            task_id,
            task_description,
            text2annotate,
            retrieval_info,
        )

        deterministic_prediction = (
            solve_deterministic(text2annotate, icl_examples)
            if solve_deterministic is not None
            else None
        )
        if deterministic_prediction is not None:
            test_record["prediction"] = deterministic_prediction
            candidate_predictions = [deterministic_prediction]
            candidate_transformations = [
                {
                    "raw_candidate": deterministic_prediction,
                    "normalized_candidate": deterministic_prediction,
                    "final_candidate": deterministic_prediction,
                }
            ]
            raw_output = f"DETERMINISTIC_SOLVER\n{deterministic_prediction}" if should_trace else None
            if (debug_first_n and idx < debug_first_n) or line_number in only_line_numbers:
                print(f"\n[DEBUG {line_number}] Input: {text2annotate}")
                print(f"[ROUTE]: {strategy_class}")
                print(f"[DETERMINISTIC]: {deterministic_prediction}\n" + "-" * 50)
        else:
            tokenized = qwen_tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False)
            if tokenized["input_ids"].shape[1] > max_input_length:
                test_record["prediction"] = None
                raw_output = "INPUT_TOO_LONG" if should_trace else None
            elif should_debug:
                prediction, raw_output, candidate_predictions, candidate_transformations = annotate(
                    input_prompt,
                    task_id=task_id,
                    debug=True,
                    text2annotate=text2annotate,
                    return_candidates=True,
                    return_candidate_transformations=True,
                )
                test_record["prediction"] = prediction
                if (debug_first_n and idx < debug_first_n) or line_number in only_line_numbers:
                    print(f"\n[DEBUG {line_number}] Input: {text2annotate}")
                    print(f"[ROUTE]: {strategy_class}")
                    print(f"[ROUTER RAW]: {router_raw_output}")
                    print(f"[RAW]: {raw_output}")
                    print(f"[PRED]: {prediction}\n" + "-" * 50)
            else:
                raw_output = None
                prediction, candidate_predictions, candidate_transformations = annotate(
                    input_prompt,
                    task_id=task_id,
                    debug=False,
                    text2annotate=text2annotate,
                    return_candidates=True,
                    return_candidate_transformations=True,
                )
                test_record["prediction"] = prediction

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + "\n")

        if raw_output is not None and not candidate_predictions:
            candidate_predictions = extract_candidates_from_debug_output(raw_output)
        if candidate_predictions and not candidate_transformations:
            candidate_transformations = describe_candidate_transformations(text2annotate, candidate_predictions)
        with open(candidates_file, "a", encoding="utf-8") as f:
            if candidate_predictions:
                for candidate_prediction in candidate_predictions:
                    candidate_record = {
                        "test_sample_id": test_sample["id"],
                        "prediction": candidate_prediction,
                    }
                    f.write(json.dumps(candidate_record, ensure_ascii=False) + "\n")
            else:
                candidate_record = {
                    "test_sample_id": test_sample["id"],
                    "prediction": None,
                }
                f.write(json.dumps(candidate_record, ensure_ascii=False) + "\n")

        if should_trace and trace_file:
            trace_record = {
                "line_number": line_number,
                "test_sample_id": test_sample["id"],
                "input": text2annotate,
                "prediction": test_record["prediction"],
                "strategy_class": strategy_class,
                "router_raw_output": router_raw_output,
                "retrieval_strategy_class": retrieval_info["strategy_class"],
                "detected_answer_type": retrieval_info["detected_answer_type"],
                "reasoning_route": retrieval_info["reasoning_route"],
                "selected_example_count": retrieval_info["selected_example_count"],
                "selected_example_ids": retrieval_info["selected_example_ids"],
                "candidate_transformations": candidate_transformations,
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
