import json, os, argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from method import build_prompt, select_examples, annotate_vllm as annotate

TASK_FILES = {
    1: "../data/openseek-1_closest_integers.json",
    2: "../data/openseek-2_count_nouns_verbs.json",
    3: "../data/openseek-3_collatz_conjecture.json",
    4: "../data/openseek-4_conala_concat_strings.json",
    5: "../data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    6: "../data/openseek-6_mnli_same_genre_classification.json",
    7: "../data/openseek-7_jeopardy_answer_generation_all.json",
    8: "../data/openseek-8_kernel_generation.json",
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--max_input_length', type=int, default=10_000)
    parser.add_argument('--log_path_prefix', type=str, default='../outputs/')
    parser.add_argument('--tokenizer_path', type=str, default='/root/OpenSeek/openseek/competition/Qwen3-4B')
    parser.add_argument('--workers', type=int, default=8, help='并发线程数，建议4-8')
    return parser.parse_args()

def process_sample(test_sample, task_description, icl_examples, output_file):
    try:
        test_sample_id = test_sample['id']
        text2annotate = test_sample['input']
        
        prompt = build_prompt(task_description, text2annotate)
        examples_str = select_examples(icl_examples, task_description, text2annotate)
        input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + '\n\n')
        
        prediction = annotate(input_prompt)
        
        # --- 关键修改：只有 prediction 不为 None 时才写入文件 ---
        if prediction is not None:
            result = {'test_sample_id': test_sample_id, 'prediction': prediction}
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            return True
        else:
            # 这种情况通常是模型没吐标签，我们可以记录一个日志但不占坑
            print(f"⚠️ 样本 {test_sample_id} 未提取到有效标签，跳过写入，待后续重试。")
            return False
            
    except Exception as e:
        # 如果是 Connection refused 等网络错误，不写文件，断点续传下次会自动重跑
        print(f"❌ 样本 {test_sample_id} 请求发生异常: {e}")
        return False

def evaluate_concurrent():
    args = parser_args()
    task_file = TASK_FILES[args.task_id]
    
    with open(task_file, 'r') as f:
        task_dict = json.load(f)
    
    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples'][:100]
    test_samples = task_dict['test_samples']
    
    # --- 断点续传逻辑 ---
    output_file = f'{args.log_path_prefix}openseek-{args.task_id}-vLLM_Final.jsonl'
    os.makedirs(args.log_path_prefix, exist_ok=True)
    
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['test_sample_id'])
                except: continue
        print(f"📌 检测到断点：已跳过 {len(processed_ids)} 个已完成样本。")

    # 过滤掉已处理的样本
    remaining_samples = [s for s in test_samples if s['id'] not in processed_ids]

    # --- 并发执行 ---
    print(f"🚀 启动并发标注 | 任务: {task_name} | 并发数: {args.workers}")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_sample, sample, task_description, icl_examples, output_file) 
            for sample in remaining_samples
        ]
        
        # 使用 tqdm 监控进度
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Annotating"):
            pass

if __name__ == '__main__':
    evaluate_concurrent()
