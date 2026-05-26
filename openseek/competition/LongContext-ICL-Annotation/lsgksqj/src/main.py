import json, os, argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer

# 从 method.py 导入
from method import build_prompt, select_examples
from method import annotate_nvidia as annotate # 确保 method.py 里的 annotate_nvidia 返回 (prediction, whole_result)

TASK_FILES = {
    1: './data/openseek-1_closest_integers.json',
    2: './data/openseek-2_count_nouns_verbs.json',
    3: './data/openseek-3_collatz_conjecture.json',
    4: './data/openseek-4_conala_concat_strings.json',
    5: './data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: './data/openseek-6_mnli_same_genre_classification.json',
    7: './data/openseek-7_jeopardy_answer_generation_all.json',
    8: './data/openseek-8_kernel_generation.json',
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID to evaluate, should be in [1, 8].')
    parser.add_argument('--max_input_length', type=int, default=10_000,
                        help='Maximum input length for the model.')
    parser.add_argument('--log_path_prefix', type=str, 
                        default='../outputs/',
                        help='Prefix path to save the evaluation logs.')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/root/autodl-tmp/Qwen-Final')
    args = parser.parse_args()
    return args

def evaluate(task_id:int, 
             qwen_tokenizer:AutoTokenizer,
             max_input_length:int=128_000,
             log_path_prefix:str='../outputs/'
        )->None:
    assert task_id in [i for i in range(1, 9)],\
        f"task_id should be in [1, 8], but got {task_id}."
    
    task_file = TASK_FILES[task_id]
    with open(task_file, 'r', encoding='utf-8') as f:
        task_dict = json.load(f)
    
    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples'][:100]
    test_samples = task_dict['test_samples']
    
    # --- 1. 处理文件名逻辑 (保持 jsonl 纯净，增加 txt 存原始回复) ---
    version = 1
    output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    response_file = f'{log_path_prefix}openseek-{task_id}-v{version}_responses.txt'
    
    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)
    
    while os.path.exists(output_file):
        version += 1
        output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
        response_file = f'{log_path_prefix}openseek-{task_id}-v{version}_responses.txt'
    
    # 初始化文件
    with open(output_file, 'w', encoding='utf-8') as f, open(response_file, 'w', encoding='utf-8') as f2:
        pass
    
    examples_str = None
    for test_sample in tqdm(test_samples, desc=f'Evaluation on Task {task_id}: {task_name}'):
        test_record = dict()
        test_sample_id = test_sample['id']
        test_record['test_sample_id'] = test_sample_id
        
        text2annotate = test_sample['input']
        prompt = build_prompt(task_description, text2annotate)
        
        if examples_str is None:
            examples_str = select_examples(icl_examples, task_description, text2annotate, qwen_tokenizer)
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str+'\n\n')
        
        # --- 2. 获取双重返回值 ---
        # 此时调用的是 method.py 里的 annotate_nvidia，它会返回 (prediction, whole_result)
        prediction, raw_content = annotate(input_prompt)
        
        # --- 3. 写入原始逻辑文件 (jsonl) ---
        test_record['prediction'] = prediction
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_record) + '\n')
            
        # --- 4. 写入原始回复文件 (txt) ---
        with open(response_file, 'a', encoding='utf-8') as f_resp:
            f_resp.write(f"ID: {test_sample_id}\n")
            f_resp.write(f"PREDICTION: {prediction}\n")
            f_resp.write(f"RAW_RESPONSE:\n{raw_content}\n")
            f_resp.write("="*80 + "\n\n")

if __name__ == '__main__':
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    evaluate(args.task_id, qwen_tokenizer, args.max_input_length, args.log_path_prefix)