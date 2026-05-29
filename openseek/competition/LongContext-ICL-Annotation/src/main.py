import json, os, argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer

# from method import build_prompt, select_examples, annotate

from method import build_prompt, select_examples, build_prompt_cot

# from method import annotate_nvidia as annotate # For Nvidia GPU
from method import annotate_ascend as annotate # For Huawei Ascend

TASK_FILES = {
    1: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-1_closest_integers.json',
    2: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-2_count_nouns_verbs.json',
    3: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-3_collatz_conjecture.json',
    4: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-4_conala_concat_strings.json',
    5: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-6_mnli_same_genre_classification.json',
    7: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-7_jeopardy_answer_generation_all.json',
    8: '/root/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data/openseek-8_kernel_generation.json',
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID to evaluate, should be in [1, 7].')
    parser.add_argument('--max_input_length', type=int, default=10_000,
                        help='Maximum input length for the model.')
    parser.add_argument('--log_path_prefix', type=str, 
                        default='../outputs/',
                        help='Prefix path to save the evaluation logs.')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/root/Qwen3-4B')
    args = parser.parse_args()
    return args

def evaluate(task_id:int, 
             qwen_tokenizer:AutoTokenizer,
             max_input_length:int=128_000,
             log_path_prefix:str='./outputs/'
        )->float:
    assert task_id in [i for i in range(1, 9)],\
        f"task_id should be in [1, 8], but got {task_id}."
    
    task_file = TASK_FILES[task_id]
    with open(task_file, 'r') as f:
        task_dict = json.load(f)
    
    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples'][:50]
    test_samples = task_dict['test_samples']
    
    version = 1
    output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)
    while os.path.exists(output_file):
        version += 1
        output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    with open(output_file, 'w') as f:
        pass
    
    examples_str = None
    for test_sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        test_record = dict()
        
        test_sample_id = test_sample['id']
        test_record['test_sample_id'] = test_sample_id
        
        
        text2annotate = test_sample['input']
        
        # Use CoT prompt for Task 3 and 4, standard prompt for others (Account 3 strategy)
        # Task 3: Collatz conjecture (math reasoning) - CoT helps
        # Task 4: String concatenation - CoT significantly helped in Account 2 (+29.2%)
        # Task 8: Kernel generation - CoT was harmful in Account 2 (6.0% -> 0.6%)
        if task_id in [3, 4]:
            prompt = build_prompt_cot(task_description, text2annotate, task_id)
        else:
            prompt = build_prompt(task_description, text2annotate)
        
        if examples_str is None:
            examples_str = select_examples(icl_examples, task_description, text2annotate)
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str+'\n\n')
        
        # tokenized_input = qwen_tokenizer(input_prompt, return_tensors="pt")
        # if tokenized_input['input_ids'].shape[1] > max_input_length:
        #     test_record['prediction'] = None
        # else:
        #     prediction = annotate(input_prompt, task_id)
        #     test_record['prediction'] = prediction
        prediction = annotate(input_prompt, task_id)
        test_record['prediction'] = prediction
        with open(output_file, 'a') as f:
            f.write(json.dumps(test_record)+'\n')

if __name__ == '__main__':
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    evaluate(args.task_id, qwen_tokenizer, args.max_input_length, args.log_path_prefix)