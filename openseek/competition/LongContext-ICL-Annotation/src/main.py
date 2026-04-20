import json, os, argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer

# from method import build_prompt, select_examples, annotate

from method import build_prompt, select_examples

# from method import annotate_nvidia as annotate # For Nvidia GPU
from method import annotate_ascend as annotate # For Huawei Ascend
from method import annotate_batch
from method import annotate_with_self_consistency                         

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
    
    examples_str = None
    batch_size = 8
    prompts_batch = []
    sample_ids_batch = []
    
    # Task 8 is code generation, needs more tokens and different post-processing
    max_tokens = 1024 if task_id == 8 else 256
    use_count_answer = False if task_id == 8 else True
    
                                         
                                             
    use_self_consistency = (task_id == 6 or (1 <= task_id <= 4))
    
                                           
    use_social_media_enhancement = (task_id == 5)
    
    for test_sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        test_sample_id = test_sample['id']
        text2annotate = test_sample['input']
                                               
                                         
                                             
        use_multitask_sharing = (1 <= task_id <= 6)
        prompt = build_prompt(task_description, text2annotate, task_id=task_id, use_social_media_enhancement=use_social_media_enhancement, use_multitask_sharing=use_multitask_sharing)
        if examples_str is None:
            # Task 8 is code generation task
            is_code_generation = (task_id == 8)
                                         
                                                       
            use_cot = (1 <= task_id <= 4)                
                                             
            balance_sentiment = (task_id == 5)
                                                       
            use_importance_weighting = (1 <= task_id <= 4)
                                                                         
            use_contrastive_learning = (1 <= task_id <= 4 or task_id == 6)
                                                    
            use_reinforcement_learning = (1 <= task_id <= 7)
                                                                 
            use_multitask_sharing = False if use_reinforcement_learning else use_multitask_sharing
            examples_str = select_examples(icl_examples, task_description, text2annotate, 
                                          is_code_generation=is_code_generation,
                                          use_task_aware=True, task_id=task_id,
                                          use_quality_filter=True, quality_threshold=0.5,
                                          use_diversity=False, use_similarity=False,
                                          use_cot=use_cot,
                                          balance_sentiment=balance_sentiment,
                                          use_importance_weighting=use_importance_weighting,
                                          use_contrastive_learning=use_contrastive_learning,
                                          use_multitask_sharing=use_multitask_sharing,
                                          use_reinforcement_learning=use_reinforcement_learning)
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str+'\n\n')
        
                                        
        if use_self_consistency:
                              
            final_answer, confidence, all_predictions = annotate_with_self_consistency(
                input_prompt, 
                num_samples=5,          
                temperature_range=[0.7, 0.85, 1.0, 1.1, 1.2],
                max_tokens=max_tokens,
                task_id=task_id,
                confidence_threshold=0.4
            )
            test_record = {
                'test_sample_id': test_sample_id, 
                'prediction': final_answer,
                'confidence': confidence,
                'all_predictions': all_predictions
            }
            with open(output_file, 'a') as f:
                f.write(json.dumps(test_record)+'\n')
        else:
                        
            prompts_batch.append(input_prompt)
            sample_ids_batch.append(test_sample_id)
            
            # Process batch when full
            if len(prompts_batch) >= batch_size:
                results = annotate_batch(prompts_batch, num_workers=4, max_tokens=max_tokens, use_count_answer=use_count_answer, task_id=task_id)
                for sid, (pred, _) in zip(sample_ids_batch, results):
                    test_record = {'test_sample_id': sid, 'prediction': pred}
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(test_record)+'\n')
                prompts_batch = []
                sample_ids_batch = []
    
                                          
    if not use_self_consistency and prompts_batch:
        results = annotate_batch(prompts_batch, num_workers=4, max_tokens=max_tokens, use_count_answer=use_count_answer, task_id=task_id)
        for sid, (pred, _) in zip(sample_ids_batch, results):
            test_record = {'test_sample_id': sid, 'prediction': pred}
            with open(output_file, 'a') as f:
                f.write(json.dumps(test_record)+'\n')

if __name__ == '__main__':
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    evaluate(args.task_id, qwen_tokenizer, args.max_input_length, args.log_path_prefix)