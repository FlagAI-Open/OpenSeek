import json, os, argparse
from tqdm import tqdm
import prompt

TASK_FILES = {
    1: '../data/openseek-1_closest_integers.json',
    2: '../data/openseek-2_count_nouns_verbs.json',
    3: '../data/openseek-3_collatz_conjecture.json',
    4: '../data/openseek-4_conala_concat_strings.json',
    5: '../data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: '../data/openseek-6_mnli_same_genre_classification.json',
    7: '../data/openseek-7_jeopardy_answer_generation_all.json',
    8: '../data/openseek-8_kernel_generation.json',
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, nargs='+', required=True,
                        help='Task ID(s) to evaluate, e.g. --task_id 2 or --task_id 2 3 4.')
    parser.add_argument('--log_path_prefix', type=str, 
                        default='../outputs/result/',
                        help='Prefix path to save the evaluation logs.')
    parser.add_argument('--backend', type=str, choices=['nvidia', 'ascend'], default='nvidia',
                        help='Which backend to use for the model execution.')
    args = parser.parse_args()
    return args

def evaluate(task_id: int, log_path_prefix: str, backend: str):
    assert task_id in [i for i in range(1, 9)], f'task_id should be in [1, 8], but got {task_id}.'
    
    task_file = TASK_FILES[task_id]
    with open(task_file, 'r', encoding='utf-8') as f:
        task_dict = json.load(f)
    
    task_name = task_dict.get('task_name', f'Task {task_id}')
    test_samples = task_dict['test_samples']
    
    version = 1
    # Adding a trailing slash if not present to mimic previous functionality
    if not log_path_prefix.endswith('/'):
        log_path_prefix += '/'

    output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)
    
    while os.path.exists(output_file):
        version += 1
        output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
        
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
        
    for sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        prediction = prompt.execute_annotation(task_id, task_file, sample['input'], backend)
        
        if task_id == 3 and isinstance(prediction, str):
            prediction = f"[{prediction}]"

        test_record = {
            "test_sample_id": sample["id"],
            "prediction": prediction
        }
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    args = parser_args()
    for t_id in args.task_id:
        evaluate(t_id, args.log_path_prefix, args.backend)
