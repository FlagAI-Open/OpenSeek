"""
生产代码：动态 token 控制 ICL 上下文在 30K-32K 之间
每个任务的逻辑封装在 tasks/ 目录中独立类中。
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import json, os, argparse, threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

from method import annotate_ascend
from tasks import TASK_REGISTRY


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True, help='Task ID to evaluate')
    parser.add_argument('--max_input_length', type=int, default=32_000)
    parser.add_argument('--log_path_prefix', type=str, default='../outputs/')
    parser.add_argument('--tokenizer_path', type=str, default='/root/models/qwen/Qwen3-4B')
    parser.add_argument('--workers', type=int, default=8, help='并发请求数')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit test samples (None = all)')
    return parser.parse_args()


def call_model(input_prompt, cfg):
    """Call the model API."""
    return annotate_ascend(
        input_prompt,
        temperature=cfg['temperature'],
        system_prompt=cfg.get('system_prompt', "You are a helpful assistant."),
        max_tokens=cfg.get('max_tokens', 4096),
        stop_tokens=cfg.get('stop_tokens', None),
    )


def evaluate(task_id: int, qwen_tokenizer, max_input_length: int = 128_000, log_path_prefix: str = './outputs/', workers: int = 8, max_samples: int = None):
    # Load data and prepare task
    task_class = TASK_REGISTRY[task_id]
    task = task_class(qwen_tokenizer)
    task.cfg = dict(task.DEFAULT_CFG)
    task.load_data()
    task.prepare()

    if max_samples:
        task.test_samples = task.test_samples[:max_samples]
        print(f"🔍 Running {max_samples} sample(s) only")

    # Versioned output file
    version = 1
    output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)
    while os.path.exists(output_file):
        version += 1
        output_file = f'{log_path_prefix}openseek-{task_id}-v{version}.jsonl'
    with open(output_file, 'w') as f:
        pass

    file_lock = threading.Lock()

    cfg = task.cfg
    extra_info = []
    if cfg.get('balanced'): extra_info.append('balanced')
    extra_str = ', ' + ', '.join(extra_info) if extra_info else ''
    print(f"⚙️  Task {task_id} config: temp={cfg['temperature']}, top_k={cfg.get('top_k', 'N/A')}, votes={cfg['num_votes']}{extra_str}")
    print(f"📊 ICL tokens: {task.min_icl_tokens:,}-{task.max_icl_tokens:,}, train={len(task.icl_examples)}, padding={len(task.padding_pool)}, test={len(task.test_samples)}")

    def process_sample(test_sample):
        final_prediction, prompt, raw_outputs, candidates, _ = task.run_inference(
            test_sample, call_model
        )

        # Write submission output
        test_record = {
            'test_sample_id': test_sample['id'],
            'prediction': final_prediction,
        }

        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(test_record, ensure_ascii=False) + '\n')

    # Thread pool execution
    print(f"🚀 启动多线程动态检索轰炸，当前并发数：{workers}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_sample, sample): sample['id'] for sample in task.test_samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Evaluation Task {task_id}'):
            try:
                future.result()
            except Exception as e:
                print(f"\n❌ Sample {futures[future]} failed: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    evaluate(args.task_id, qwen_tokenizer, args.max_input_length, args.log_path_prefix, args.workers, args.max_samples)
