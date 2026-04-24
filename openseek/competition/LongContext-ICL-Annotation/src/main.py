import json, os, argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer

# from method import build_prompt, select_examples, annotate

from method import build_prompt, select_examples, get_multitask_task_config

# from method import annotate_nvidia as annotate # For Nvidia GPU
from method import annotate_ascend as annotate # For Huawei Ascend
from method import annotate_batch

DATA_DIR = '/root/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data'
OUTPUT_DIR = '/root/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs'

TASK_FILES = {
    1: f'{DATA_DIR}/openseek-1_closest_integers.json',
    2: f'{DATA_DIR}/openseek-2_count_nouns_verbs.json',
    3: f'{DATA_DIR}/openseek-3_collatz_conjecture.json',
    4: f'{DATA_DIR}/openseek-4_conala_concat_strings.json',
    5: f'{DATA_DIR}/openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: f'{DATA_DIR}/openseek-6_mnli_same_genre_classification.json',
    7: f'{DATA_DIR}/openseek-7_jeopardy_answer_generation_all.json',
    8: f'{DATA_DIR}/openseek-8_kernel_generation.json',
}


                                                 
def build_multitask_example_pool(current_task_id: int, current_examples: list[dict]) -> list[dict]:
    example_pool = []
    for example in current_examples:
        enriched_example = dict(example)
        enriched_example['source_task_id'] = current_task_id
        example_pool.append(enriched_example)

    for source_task_id in range(1, 8):
        if source_task_id == current_task_id:
            continue
        try:
            with open(TASK_FILES[source_task_id], 'r') as shared_file:
                shared_task_dict = json.load(shared_file)
            for example in shared_task_dict.get('examples', [])[:8]:
                enriched_example = dict(example)
                enriched_example['source_task_id'] = source_task_id
                example_pool.append(enriched_example)
        except Exception as exc:
            print(f"警告：共享任务池加载失败 task={source_task_id}: {exc}")
    return example_pool

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID to evaluate, should be in [1, 7].')
    parser.add_argument('--max_input_length', type=int, default=10_000,
                        help='Maximum input length for the model.')
    parser.add_argument('--log_path_prefix', type=str, 
                        default='/root/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/',
                        help='Prefix path to save the evaluation logs.')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/root/flagos/Qwen3-4B')
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
    
    batch_size = 4
    prompts_batch = []
    sample_ids_batch = []
    use_multitask_optimization = (1 <= task_id <= 7)
    task_config = get_multitask_task_config(task_id) if use_multitask_optimization else {'cot': False, 'self_consistency': False}
    use_dynamic_adaptive = (1 <= task_id <= 7)
    use_cot_examples = task_config['cot'] if use_multitask_optimization else (1 <= task_id <= 7)
    use_self_consistency = task_config['self_consistency'] if use_multitask_optimization else (1 <= task_id <= 7)
                                                        
    candidate_examples = build_multitask_example_pool(task_id, icl_examples) if use_multitask_optimization else icl_examples
    
    # Task 8 is code generation, needs more tokens and different post-processing
    max_tokens = 1024 if task_id == 8 else 256
    use_count_answer = False if task_id == 8 else True
    
    for test_sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        test_sample_id = test_sample['id']
        text2annotate = test_sample['input']
        prompt = build_prompt(task_description, text2annotate)
                                                                      
        is_code_generation = (task_id == 8)
        examples_str = select_examples(
            candidate_examples,
            task_description,
            text2annotate,
            is_code_generation=is_code_generation,
            task_id=task_id,
            use_dynamic_adaptive=use_dynamic_adaptive,
            use_cot=use_cot_examples and not is_code_generation,
            use_multitask_optimization=use_multitask_optimization and not is_code_generation,
        )
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str+'\n\n')
        
        prompts_batch.append(input_prompt)
        sample_ids_batch.append(test_sample_id)
        
        # Process batch when full
        if len(prompts_batch) >= batch_size:
            results = annotate_batch(
                prompts_batch,
                num_workers=4,
                max_tokens=max_tokens,
                use_count_answer=use_count_answer,
                task_id=task_id,
                use_self_consistency=use_self_consistency,
            )
            for sid, (pred, _) in zip(sample_ids_batch, results):
                test_record = {'test_sample_id': sid, 'prediction': pred}
                with open(output_file, 'a') as f:
                    f.write(json.dumps(test_record)+'\n')
            prompts_batch = []
            sample_ids_batch = []
    
    # Process remaining samples
    if prompts_batch:
        results = annotate_batch(
            prompts_batch,
            num_workers=4,
            max_tokens=max_tokens,
            use_count_answer=use_count_answer,
            task_id=task_id,
            use_self_consistency=use_self_consistency,
        )
        for sid, (pred, _) in zip(sample_ids_batch, results):
            test_record = {'test_sample_id': sid, 'prediction': pred}
            with open(output_file, 'a') as f:
                f.write(json.dumps(test_record)+'\n')

if __name__ == '__main__':
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    evaluate(args.task_id, qwen_tokenizer, args.max_input_length, args.log_path_prefix)