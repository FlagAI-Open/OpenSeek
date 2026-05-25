import argparse
import json
from pathlib import Path
from functools import partial

# 全局设置 print 默认 flush=True，解决日志缓存问题
print = partial(print, flush=True)

from method import (
    get_strategy,
    normalize_examples,
    select_examples,
)
from llm_client import is_completion_server_available
from strategy_verified import VerifiedProgramStrategy
from submission_utils import (
    ensure_directory,
    get_output_file,
    save_jsonl,
    zip_submission_files,
)

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / 'submission_results'
DEFAULT_ZIP_PATH = DEFAULT_OUTPUT_DIR / 'result.zip'
DEFAULT_DATA_DIR = ROOT_DIR / 'data'

TASK_FILES = {
    1: DEFAULT_DATA_DIR / 'openseek-1_closest_integers.json',
    2: DEFAULT_DATA_DIR / 'openseek-2_count_nouns_verbs.json',
    3: DEFAULT_DATA_DIR / 'openseek-3_collatz_conjecture.json',
    4: DEFAULT_DATA_DIR / 'openseek-4_conala_concat_strings.json',
    5: DEFAULT_DATA_DIR / 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: DEFAULT_DATA_DIR / 'openseek-6_mnli_same_genre_classification.json',
    7: DEFAULT_DATA_DIR / 'openseek-7_jeopardy_answer_generation_all.json',
    8: DEFAULT_DATA_DIR / 'openseek-8_kernel_generation.json',
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, nargs='*',
                        help='Task IDs to evaluate. If omitted, runs all tasks 1-8.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of samples for each selected task. Useful for quick testing.')
    parser.add_argument('--output_dir', type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help='Directory used to store submission jsonl files.')
    parser.add_argument('--zip_path', type=str,
                        default=str(DEFAULT_ZIP_PATH),
                        help='Zip file path for direct competition submission.')
    args = parser.parse_args()
    return args


def resolve_task_ids(task_ids: list[int] | None) -> list[int]:
    if not task_ids:
        return list(TASK_FILES.keys())

    invalid_task_ids = [task_id for task_id in task_ids if task_id not in TASK_FILES]
    if invalid_task_ids:
        raise ValueError(f"task_id should be in [1, 8], but got {invalid_task_ids}.")

    return list(dict.fromkeys(task_ids))


def evaluate_task(task_id: int,
                  output_dir: Path = DEFAULT_OUTPUT_DIR,
                  limit: int | None = None,
                  ):
    assert task_id in TASK_FILES, f"task_id should be in [1, 8], but got {task_id}."

    task_file = TASK_FILES[task_id]
    with open(task_file, 'r', encoding='utf-8') as f:
        task_dict = json.load(f)

    task_description = task_dict['Definition'][0]
    test_samples = task_dict['test_samples']
    if limit is not None:
        test_samples = test_samples[:limit]

    output_file = get_output_file(task_id, output_dir)
    ensure_directory(output_dir)

    records: list[dict] = []
    total_samples = len(test_samples)

    if not is_completion_server_available():
        print(f'Model service is unavailable. Please check your config.')
        for test_sample in test_samples:
            records.append({'test_sample_id': test_sample['id'], 'prediction': None})
        save_jsonl(records, output_file)
        return output_file, 0, 0

    raw_examples = task_dict['examples']
    all_normalized_examples = normalize_examples(raw_examples)
    prompt_examples = select_examples(raw_examples, example_count=3)

    strategy = get_strategy(task_id)
    print(f"Starting Task {task_id} using {strategy.__class__.__name__}...")

    if hasattr(strategy, '_select_relevant_examples'):
        current_context_examples = all_normalized_examples
    else:
        current_context_examples = prompt_examples

    solution_code = None
    if isinstance(strategy, VerifiedProgramStrategy):
        solution_code = strategy.prepare_solution(
            task_description=task_description,
            prompt_examples=prompt_examples,
        )
        if not solution_code:
            print(f"Failed to generate code for Task {task_id}, will retry execution for each sample if needed.")

    for idx, test_sample in enumerate(test_samples, 1):
        test_sample_id = test_sample['id']
        input_text = test_sample['input']

        try:
            if isinstance(strategy, VerifiedProgramStrategy) and solution_code:
                prediction = strategy.execute(solution_code, input_text)
            else:
                prediction = strategy.predict(
                    task_id=task_id,
                    task_description=task_description,
                    prompt_examples=current_context_examples,
                    input_text=input_text
                )
        except Exception as exc:
            print(f"Execution failed for sample {test_sample_id}: {exc}")
            prediction = None

        records.append({'test_sample_id': test_sample_id, 'prediction': prediction})
        print(f"[{idx}/{total_samples}] ID: {test_sample_id} | Prediction generated.")
        save_jsonl(records, output_file)

    return output_file, len(records), total_samples


if __name__ == '__main__':
    args = parser_args()
    output_dir = Path(args.output_dir)
    zip_path = Path(args.zip_path)
    task_ids = resolve_task_ids(args.task_id)

    task_results: list[tuple[int, Path, int, int]] = []
    for task_id in task_ids:
        output_file, processed, total = evaluate_task(task_id, output_dir, args.limit)
        task_results.append((task_id, output_file, processed, total))

    zip_submission_files(output_dir, zip_path, DEFAULT_DATA_DIR)

    for task_id, output_file, processed, total in task_results:
        print(f'Task {task_id} completed. Processed {processed}/{total} samples.')
        print(f'output_file: {output_file}')
    print(f'zip_file: {zip_path}')