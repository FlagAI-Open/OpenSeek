import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# from method import build_prompt, select_examples, annotate

from method import (
    TASK_SAMPLE_ATTEMPTS,
    annotate_task7_rerank,
    build_chat_examples,
    build_prompt,
    build_task5_retrieval_context,
    build_task6_retrieval_context,
    build_task7_retrieval_context,
    get_example_pool_limit,
    get_profile_name,
    get_profile_retrieval_tasks,
    get_task7_rerank_retrieval_mode,
    get_task7_rerank_secondary_candidates,
    get_task7_rerank_secondary_profile,
    get_task7_rerank_secondary_retrieval_mode,
    reorder_task5_examples_by_lexical_retrieval,
    reorder_task6_examples_by_genre_retrieval,
    reorder_task7_examples_by_lexical_retrieval,
    select_examples,
    should_use_task7_author_direct_fact_projection,
    should_use_task7_rerank,
    should_use_chat_retrieval,
    solve_task_locally,
    _build_task7_author_answer_catalog,
)
from method import CHAT_THINKING_TASKS, annotate_chat_thinking
from task8_retrieval import build_task8_lexical_retrieval_context, reorder_task8_examples_by_lexical_retrieval

from method import annotate_nvidia as annotate # For Nvidia GPU
# from method import annotate_ascend as annotate # For Huawei Ascend

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

TASK_FILES = {
    1: DATA_DIR / "openseek-1_closest_integers.json",
    2: DATA_DIR / "openseek-2_count_nouns_verbs.json",
    3: DATA_DIR / "openseek-3_collatz_conjecture.json",
    4: DATA_DIR / "openseek-4_conala_concat_strings.json",
    5: DATA_DIR / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    6: DATA_DIR / "openseek-6_mnli_same_genre_classification.json",
    7: DATA_DIR / "openseek-7_jeopardy_answer_generation_all.json",
    8: DATA_DIR / "openseek-8_kernel_generation.json",
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID to evaluate, should be in [1, 8].')
    parser.add_argument('--max_input_length', type=int, default=10_000,
                        help='Maximum input length for the model.')
    parser.add_argument('--log_path_prefix', type=str, 
                        default=str(OUTPUTS_DIR),
                        help='Prefix path to save the evaluation logs.')
    parser.add_argument('--tokenizer_path', type=str,
                        default=None,
                        help='Optional local tokenizer/model path for token counting.')
    parser.add_argument('--examples_limit', type=int, default=None,
                        help='Maximum number of ICL examples to consider before selection. Defaults to the active profile.')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='Optional cap on the number of test samples for smoke testing.')
    parser.add_argument('--profile', type=str, default=None,
                        help='Execution profile name. Defaults to OPENSEEK_PROFILE or baseline.')
    args = parser.parse_args()
    return args

def evaluate(task_id:int, 
             qwen_tokenizer:Any|None,
             max_input_length:int=128_000,
             log_path_prefix:str='./outputs/',
             examples_limit:int=100,
             sample_limit:int|None=None,
             profile_name:str|None=None,
        )->str:
    assert task_id in [i for i in range(1, 9)],\
        f"task_id should be in [1, 8], but got {task_id}."
    profile_name = get_profile_name(profile_name)
    
    task_file = TASK_FILES[task_id]
    with open(task_file, 'r', encoding='utf-8') as f:
        task_dict = json.load(f)
    
    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    resolved_examples_limit = get_example_pool_limit(
        task_id=task_id,
        examples_limit=examples_limit,
        profile_name=profile_name,
    )
    icl_examples = task_dict['examples'][:resolved_examples_limit]
    test_samples = task_dict['test_samples']
    if sample_limit is not None:
        test_samples = test_samples[:sample_limit]
    
    version = 1
    output_path = Path(log_path_prefix)
    output_file = output_path / f'openseek-{task_id}-v{version}.jsonl'
    os.makedirs(output_path, exist_ok=True)
    while os.path.exists(output_file):
        version += 1
        output_file = output_path / f'openseek-{task_id}-v{version}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
    
    examples_str = None
    task8_retrieval_context = build_task8_lexical_retrieval_context(icl_examples) if task_id == 8 else None
    chat_retrieval_context = (
        build_task5_retrieval_context(icl_examples) if task_id == 5 and should_use_chat_retrieval(task_id) else None
    )
    profile_retrieval_tasks = get_profile_retrieval_tasks(profile_name)
    task6_retrieval_context = (
        build_task6_retrieval_context(icl_examples) if task_id == 6 and task_id in profile_retrieval_tasks else None
    )
    task7_retrieval_context = build_task7_retrieval_context(icl_examples) if task_id == 7 and task_id in profile_retrieval_tasks else None
    task7_rerank_retrieval_mode = get_task7_rerank_retrieval_mode() if should_use_task7_rerank(task_id) else "none"
    use_task7_profile_retrieval = task_id == 7 and task_id in profile_retrieval_tasks
    use_task7_primary_retrieval = task_id == 7 and (
        task7_rerank_retrieval_mode == "lexical" or use_task7_profile_retrieval
    )
    task7_secondary_profile_name = get_task7_rerank_secondary_profile() if should_use_task7_rerank(task_id) else None
    task7_secondary_retrieval_mode = get_task7_rerank_secondary_retrieval_mode() if should_use_task7_rerank(task_id) else "none"
    task7_secondary_candidate_count = get_task7_rerank_secondary_candidates() if should_use_task7_rerank(task_id) else 0
    task7_author_catalog = None
    if task_id == 7 and should_use_task7_rerank(task_id) and should_use_task7_author_direct_fact_projection():
        task7_author_catalog = _build_task7_author_answer_catalog(icl_examples)
    task7_secondary_examples = None
    task7_secondary_retrieval_context = None
    if task_id == 7 and task7_secondary_profile_name and task7_secondary_candidate_count > 0:
        secondary_limit = get_example_pool_limit(
            task_id=task_id,
            examples_limit=examples_limit,
            profile_name=task7_secondary_profile_name,
        )
        task7_secondary_examples = task_dict["examples"][:secondary_limit]
        if task7_secondary_retrieval_mode == "lexical":
            task7_secondary_retrieval_context = build_task7_retrieval_context(task7_secondary_examples)
    for test_sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        test_record = dict()
        
        test_sample_id = test_sample['id']
        test_record['test_sample_id'] = test_sample_id


        text2annotate = test_sample['input']
        local_prediction = solve_task_locally(task_id, text2annotate)
        if local_prediction is not None:
            test_record['prediction'] = local_prediction
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(test_record)+'\n')
            continue

        prompt = build_prompt(task_description, text2annotate, task_id=task_id, profile_name=profile_name)
        current_examples_str = examples_str
        secondary_input_prompt = None
        if task_id == 8:
            candidate_examples = icl_examples
            candidate_examples = reorder_task8_examples_by_lexical_retrieval(
                icl_examples,
                text2annotate,
                task8_retrieval_context,
            )
            current_examples_str = select_examples(
                candidate_examples,
                task_description,
                text2annotate,
                task_id=task_id,
                profile_name=profile_name,
            )
        elif task_id == 6 and task_id in profile_retrieval_tasks:
            candidate_examples = reorder_task6_examples_by_genre_retrieval(
                icl_examples,
                text2annotate,
                task6_retrieval_context or build_task6_retrieval_context(icl_examples),
            )
            current_examples_str = select_examples(
                candidate_examples,
                task_description,
                text2annotate,
                task_id=task_id,
                profile_name=profile_name,
            )
        elif (
            task_id == 7
            and use_task7_primary_retrieval
        ):
            candidate_examples = reorder_task7_examples_by_lexical_retrieval(
                icl_examples,
                text2annotate,
                task7_retrieval_context or build_task7_retrieval_context(icl_examples),
            )
            current_examples_str = select_examples(
                candidate_examples,
                task_description,
                text2annotate,
                task_id=task_id,
                profile_name=profile_name,
            )
        elif examples_str is None:
            current_examples_str = select_examples(
                icl_examples,
                task_description,
                text2annotate,
                task_id=task_id,
                profile_name=profile_name,
            )
            examples_str = current_examples_str
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", current_examples_str+'\n\n')
        if task_id == 7 and task7_secondary_examples and task7_secondary_candidate_count > 0:
            secondary_prompt = build_prompt(
                task_description,
                text2annotate,
                task_id=task_id,
                profile_name=task7_secondary_profile_name,
            )
            secondary_candidate_examples = task7_secondary_examples
            if task7_secondary_retrieval_mode == "lexical":
                secondary_candidate_examples = reorder_task7_examples_by_lexical_retrieval(
                    task7_secondary_examples,
                    text2annotate,
                    task7_secondary_retrieval_context or build_task7_retrieval_context(task7_secondary_examples),
                )
            secondary_examples_str = select_examples(
                secondary_candidate_examples,
                task_description,
                text2annotate,
                task_id=task_id,
                profile_name=task7_secondary_profile_name,
            )
            secondary_input_prompt = secondary_prompt.replace("[[EXAMPLES]]\n\n", secondary_examples_str + '\n\n')

        prediction = None

        # Use chat+thinking mode for supported tasks
        if task_id in CHAT_THINKING_TASKS:
            if task_id == 5 and should_use_chat_retrieval(task_id):
                reordered_examples = reorder_task5_examples_by_lexical_retrieval(
                    icl_examples,
                    text2annotate,
                    chat_retrieval_context,
                )
                current_chat_examples = build_chat_examples(
                    reordered_examples,
                    task_id=task_id,
                    examples_limit=examples_limit,
                    profile_name=profile_name,
                )
            else:
                current_chat_examples = None
            if current_chat_examples is None and (
                not hasattr(evaluate, '_chat_examples_str') or evaluate._chat_task_id != task_id
            ):
                evaluate._chat_examples_str = build_chat_examples(
                    icl_examples,
                    task_id=task_id,
                    examples_limit=examples_limit,
                    profile_name=profile_name,
                )
                evaluate._chat_task_id = task_id
                evaluate._chat_profile_name = profile_name
            elif current_chat_examples is None and getattr(evaluate, '_chat_profile_name', None) != profile_name:
                evaluate._chat_examples_str = build_chat_examples(
                    icl_examples,
                    task_id=task_id,
                    examples_limit=examples_limit,
                    profile_name=profile_name,
                )
                evaluate._chat_task_id = task_id
                evaluate._chat_profile_name = profile_name
            prediction = annotate_chat_thinking(
                current_chat_examples or evaluate._chat_examples_str,
                text2annotate,
                task_id,
                task_description,
                profile_name=profile_name,
            )
            test_record['prediction'] = prediction
        elif should_use_task7_rerank(task_id):
            prediction = annotate_task7_rerank(
                input_prompt,
                text2annotate,
                secondary_input_prompt=secondary_input_prompt,
                author_catalog=task7_author_catalog,
            )
            test_record['prediction'] = prediction
        elif qwen_tokenizer is not None:
            tokenized_input = qwen_tokenizer(input_prompt, return_tensors="pt")
            if tokenized_input['input_ids'].shape[1] > max_input_length:
                test_record['prediction'] = None
            else:
                sample_attempts = TASK_SAMPLE_ATTEMPTS.get(task_id, 1)
                for _ in range(sample_attempts):
                    prediction = annotate(input_prompt, task_id=task_id)
                    if prediction is not None:
                        break
                test_record['prediction'] = prediction
        else:
            sample_attempts = TASK_SAMPLE_ATTEMPTS.get(task_id, 1)
            for _ in range(sample_attempts):
                prediction = annotate(input_prompt, task_id=task_id)
                if prediction is not None:
                    break
            test_record['prediction'] = prediction
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_record)+'\n')
    return str(output_file)

if __name__ == '__main__':
    args = parser_args()
    qwen_tokenizer = None
    if args.tokenizer_path:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required when --tokenizer_path is provided")
        qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    evaluate(
        args.task_id,
        qwen_tokenizer,
        args.max_input_length,
        args.log_path_prefix,
        args.examples_limit,
        args.sample_limit,
        args.profile,
    )
