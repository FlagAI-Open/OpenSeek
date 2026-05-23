import json, os, argparse, threading
from tqdm import tqdm
from transformers import AutoTokenizer
import asyncio 

write_lock = threading.Lock()

# Import functions from main.py
from main import TASK_FILES, annotate_sample

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=0, type=int,
                        help='Task ID to evaluate (0 = evaluate all tasks 1-8)')
    parser.add_argument('--max_examples', type=int, default=0,
                        help='Max examples per task to evaluate (0 = use all examples)')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/FlagRelease/Qwen3-4B-FlagOS-Ascend/',
                        help='Path to tokenizer')
    parser.add_argument('--output_summary', type=str, default='accuracy_summary.json',
                        help='Path to save accuracy summary JSON')
    parser.add_argument('--threads', type=int, default=64,
                        help='Number of concurrent threads (max 128)')
    parser.add_argument('--error_log_path', type=str, default='evaluation_error.jsonl',
                        help='Path to save evaluation error logs (JSONL format)')
    args = parser.parse_args()
    return args

def process_example(example, idx, all_examples, task_description, task_id, args_mock, correct_ref, total_ref, errors_ref, pbar, error_log_path):
    try:
        ground_truth = example['output'][0].strip()
        input_text = example['input']

        # Exclude current example from ICL pool to avoid data leakage
        icl_pool = all_examples[:idx] + all_examples[idx+1:]
        # 复用main.py的annotate_sample统一逻辑，支持按tag选择样本
        prediction = annotate_sample(task_description, input_text, task_id,
                                    icl_examples=icl_pool, args=args_mock, tag=example.get('tag'))

        prediction = prediction.strip()

        with write_lock:
            if prediction == ground_truth:
                correct_ref[0] += 1
            else:
                pass
                #result = asyncio.run(bot.run(f'''
                #result is wrong, please analysis reasons
                #'''))
                #print(result.content)

            total_ref[0] += 1

    except Exception as e:
        error_msg = f"Example {example['id']}: {str(e)}"
        # 收集错误详情
        error_detail = {
            "example_id": example.get('id', 'unknown'),
            "task_id": task_id,
            "input": input_text,
            "ground_truth": ground_truth,
            "prediction": locals().get('prediction', None),
            "error": str(e)
        }
        with write_lock:
            errors_ref.append(error_msg)
            tqdm.write(f"Error: {error_msg}")
            # 写入错误日志文件
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_detail, ensure_ascii=False) + '\n')
    finally:
        with write_lock:
            pbar.update(1)

def evaluate_task(task_id: int, qwen_tokenizer: AutoTokenizer, tokenizer_path: str, max_examples: int = 0, threads: int = 64, error_log_path: str = 'evaluation_error.jsonl'):
    assert task_id in range(1, 9), f"task_id should be 1-8, got {task_id}"

    task_file = TASK_FILES[task_id]
    with open(task_file, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    all_examples = task_dict['examples']

    # Limit number of examples if specified
    if max_examples > 0 and len(all_examples) > max_examples:
        eval_examples = all_examples[:max_examples]
    else:
        eval_examples = all_examples

    # 使用列表来保存可变对象，便于多线程修改
    correct = [0]
    total = [0]
    errors = []

    # Create args mock for select_examples
    class ArgsMock:
        def __init__(self, tokenizer, tokenizer_path, max_input_length=10000):
            self.tokenizer = tokenizer
            self.tokenizer_path = tokenizer_path
            self.max_input_length = max_input_length
    args_mock = ArgsMock(qwen_tokenizer, tokenizer_path)

    print(f"\n{'='*60}")
    print(f"Task {task_id}: {task_name}")
    print(f"Evaluating {len(eval_examples)} examples with {threads} threads...")
    print('='*60)

    # 限制最大线程数为128，和main.py保持一致
    thread_num = min(threads, 128)
    if thread_num <= 0:
        thread_num = len(eval_examples)

    pbar = tqdm(total=len(eval_examples), desc=f"Task {task_id}")

    # 按批次处理样本
    for batch_start in range(0, len(eval_examples), thread_num):
        batch_end = min(batch_start + thread_num, len(eval_examples))
        batch_examples = eval_examples[batch_start:batch_end]
        threads_list = []

        for idx_in_batch, example in enumerate(batch_examples):
            global_idx = batch_start + idx_in_batch
            kwargs = {
                "example": example,
                "idx": global_idx,
                "all_examples": all_examples,
                "task_description": task_description,
                "task_id": task_id,
                "args_mock": args_mock,
                "correct_ref": correct,
                "total_ref": total,
                "errors_ref": errors,
                "pbar": pbar,
                "error_log_path": error_log_path
            }
            t = threading.Thread(target=process_example, kwargs=kwargs)
            t.daemon = True
            t.start()
            threads_list.append(t)

        # 等待当前批次所有线程完成
        for t in threads_list:
            t.join()

    pbar.close()

    accuracy = correct[0] / total[0] if total[0] > 0 else 0.0

    print(f"\nResults for Task {task_id}:")
    print(f"  Total: {total[0]} | Correct: {correct[0]} | Errors: {len(errors)}")
    print(f"  Accuracy: {accuracy:.2%}")
    if errors:
        print(f"  Errors: {len(errors)} errors occurred during evaluation")

    return {
        'task_id': task_id,
        'task_name': task_name,
        'total': total[0],
        'correct': correct[0],
        'errors': len(errors),
        'accuracy': accuracy,
        'error_list': errors
    }

def main():
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    tasks = list(range(1,9)) if args.task_id == 0 else [args.task_id]
    results = []

    for task_id in tasks:
        task_result = evaluate_task(task_id, qwen_tokenizer, args.tokenizer_path, args.max_examples, args.threads, args.error_log_path)
        results.append(task_result)

    # Generate final summary
    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY SUMMARY")
    print('='*60)
    print(f"{'Task':<6} {'Task Name':<40} {'Accuracy':<10} {'Correct/Total':<15}")
    print('-'*60)

    total_correct_all = 0
    total_total_all = 0

    for res in results:
        print(f"{res['task_id']:<6} {res['task_name']:<40} {res['accuracy']:<10.2%} {res['correct']}/{res['total']:<15}")
        total_correct_all += res['correct']
        total_total_all += res['total']

    overall_accuracy = total_correct_all / total_total_all if total_total_all > 0 else 0.0
    print('-'*60)
    print(f"{'Overall':<6} {'Average':<40} {overall_accuracy:<10.2%} {total_correct_all}/{total_total_all:<15}")
    print('='*60)

    # Save summary to file
    summary = {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct_all,
        'total_total': total_total_all,
        'tasks': results
    }

    with open(args.output_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved to: {args.output_summary}")

if __name__ == '__main__':
    main()
