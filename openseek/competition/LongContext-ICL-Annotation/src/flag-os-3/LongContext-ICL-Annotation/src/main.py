import json, os, argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

# from method import build_prompt, select_examples, annotate

from method import build_prompt, select_examples, select_examples_by_genre, select_examples_by_tag, select_examples_from_cache, annotate_nanobot, annotate_nanobot_task8, select_n_examples
from rules import TASK6_ANNOTATION_RULES

# 导入统一路径配置
from config import (
    TASK_FILES, PROCESSED_TASK_FILES, TASK5_CACHE_PATH, OUTPUT_DIR
)

# from method import annotate_nvidia as annotate  # For Nvidia GPU
import threading
write_lock = threading.Lock()

from method import annotate_ascend as annotate # For Huawei Ascend

def majority_vote(predictions: list) -> str:
    """
    多数投票函数：返回出现次数最多的预测结果
    - 如果有多个结果出现次数相同，返回第一个出现的
    - 自动过滤None和空字符串
    """
    from collections import Counter
    # 过滤无效结果
    valid_preds = [p for p in predictions if p and p.strip()]
    if not valid_preds:
        return ""
    # 统计频次
    count = Counter(valid_preds)
    # 找到最高频次
    max_freq = max(count.values())
    # 找到所有达到最高频次的结果
    candidates = [pred for pred, freq in count.items() if freq == max_freq]
    # 返回第一个出现的最高频次结果
    return candidates[0]

def annotate_sample(task_description: str, text2annotate: str, task_id: int,
                   icl_examples: list = None, args = None, tag: str = None) -> str:
    """
    通用标注函数：根据任务ID自动选择合适的标注方式
    封装所有标注逻辑，方便重复调用实现投票
    每次调用独立生成/加载示例，保证投票多样性
    """
    from rules import RULES
    use_tools = True if task_id in [1,2,3,4] else False
    if task_id == 8:
        # 任务8使用nanobot带迭代验证机制
        examples_str = "" #select_examples(icl_examples, task_description, text2annotate, args)
        rules = ""
        _, prediction = annotate_nanobot_task8(task_description, text2annotate, task_id, examples_str, rules, use_tools)
        return prediction
    else:
        # 任务1-6使用nanobot标注
        if task_id == 5:
            # 任务5自动加载缓存示例，自动按最大上下文长度过滤
            examples_str = select_examples_from_cache(TASK5_CACHE_PATH, task_description, text2annotate, args, max_samples=50)
            rules = RULES[task_id]

            # 预处理：去掉@用户名，净化推文
            import re
            def clean_mentions(text):
                # 去掉@后面跟着的用户名（直到空格/标点/结束）
                text = re.sub(r'@[\w_]+(\s+|$)', '', text)
                # 去掉多个连续空格
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            # 处理待标注文本
            text_clean = clean_mentions(text2annotate)
            # 处理示例中的@
            examples_str_clean = clean_mentions(examples_str)

            _, prediction = annotate_nanobot(task_description, text_clean, task_id, examples_str_clean, rules, use_tools)
        elif task_id == 6:
            examples_str = select_examples_by_genre(icl_examples, task_description, text2annotate, args, n_per_genre=50)
            rules = TASK6_ANNOTATION_RULES
            _, prediction = annotate_nanobot(task_description, text2annotate, task_id, examples_str, rules, use_tools)
        elif task_id in [7]:
            max_samples = 50
            if tag:
                # 按tag选择相同类别的few-shot样本
                examples_str = select_examples_by_tag(icl_examples, task_description, text2annotate, args, target_tag=tag, max_samples=max_samples)
            else:
                # 没有tag的话使用select_n_examples选择最多50个示例，自动按长度过滤
                examples_str = select_n_examples(icl_examples, task_description, text2annotate, args, n_samples=max_samples)
            rules = RULES[task_id]
            _, prediction = annotate_nanobot(task_description, text2annotate, task_id, examples_str, rules, use_tools)
        else:
            # 其他任务直接调用nanobot [1 2 3 4]
            rules = RULES[task_id]
            _, prediction = annotate_nanobot(task_description, text2annotate, task_id,rules = rules, use_tools = use_tools)
        return prediction

# 使用config.py中统一配置的TASK_FILES和PROCESSED_TASK_FILES
# PROCESSED_TASK_FILES包含预处理后的带标签数据
# TASK_FILES包含原始数据路径


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=1, type=int, required=True,
                        help='Task ID to evaluate, should be in [1, 7].')
    parser.add_argument('--max_input_length', type=int, default=10_000,
                        help='Maximum input length for the model.')
    parser.add_argument('--output_dir', type=str,
                        default=OUTPUT_DIR,
                        help='Prefix path to save the evaluation logs.')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/share/project/wuhaiming/spaces/data_agent/OpenSeek-main/openseek/competition/LongContext-ICL-Annotation/src/Qwen3-4B')
    parser.add_argument('--threads', type=int,
                        default=64)
    args = parser.parse_args()
    return args

def process_sample(test_sample, task_description, icl_examples, args, output_file, task_id):
    test_record = dict()
    test_sample_id = test_sample['id']
    test_record['test_sample_id'] = test_sample_id

    text2annotate = test_sample['input']

    # 3次投票标注：每次调用annotate_sample，内部会独立采样示例
    predictions = []
    votetimes = 1 if task_id == 8 else 9
    for i in range(votetimes):
        pred = annotate_sample(task_description, text2annotate, task_id,
                              icl_examples=icl_examples, args=args, tag=test_sample.get('tag'))
        predictions.append(pred)
        print(f"🔍 Vote {i+1}: {pred}")

    # 多数投票得到最终结果
    final_prediction = majority_vote(predictions)
    print(f"✅ Final prediction: {final_prediction} (votes: {predictions})")

    test_record['prediction'] = final_prediction

    with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_record, ensure_ascii=False) + '\n')


def evaluate(task_id: int,
             qwen_tokenizer: AutoTokenizer,
             max_input_length: int = 128_000,
             output_dir: str = OUTPUT_DIR
             ,args = None) -> float:
    assert task_id in [i for i in range(1, 9)], \
        f"task_id should be in [1, 8], but got {task_id}."

    # 任务2、7优先使用预处理后的带标签数据
    if task_id in PROCESSED_TASK_FILES:
        task_file = PROCESSED_TASK_FILES[task_id]
        print(f"📂 加载预处理后的数据(带标签): {task_file}")
    else:
        task_file = TASK_FILES[task_id]
        print(f"📂 加载原始数据: {task_file}")

    with open(task_file, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples']
    test_samples = task_dict['test_samples']

    print(f'{task_name}: {task_description}')
    print(f'num:{len(test_samples)}')

    examples_str = None
    # 确定线程数，最多128（和原有逻辑保持一致）
    max_workers = args.threads if args.threads > 0 else len(test_samples)
    max_workers = min(max_workers, 128)

    # ====================== 任务8断点续标逻辑 ======================
    if task_id == 8:
        # 任务8固定用v1版本，保留已有结果
        output_file = f'{output_dir}/openseek-8-v1.jsonl'
        output_path = os.path.dirname(output_file)
        os.makedirs(output_path, exist_ok=True)

        # 读取已有标注结果
        existing_results = {}
        if os.path.exists(output_file):
            print(f"📂 读取已有标注结果：{output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        existing_results[record['test_sample_id']] = record
                    except Exception as e:
                        print(f"⚠️  读取无效行：{str(e)}")

        # 过滤有效标注结果：只保留有有效预测结果的记录，废弃空结果
        valid_existing = {}
        for sample_id, record in existing_results.items():
            if record.get('prediction') and str(record['prediction']).strip():
                valid_existing[sample_id] = record

        # 清空原有文件，重新写入所有有效结果（重建索引，清除无效行，避免文件损坏
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in valid_existing.values():
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"✅ 重建标注文件，已写入 {len(valid_existing)} 条有效结果，原无效记录已清理")

        # 更新existing_results为有效结果集
        existing_results = valid_existing
        print(test_samples[0])

        # 不再过滤样本，所有样本都执行检测流程
        to_process = [i for i in test_samples if i['id'] not in [j for j in  existing_results]]
        print(f"✅ 总样本数：{len(test_samples)}，已标注：{len(existing_results)}")

    else:
        # 其他任务保持原有自动版本递增逻辑
        version = 1
        output_file = f'{output_dir}/openseek-{task_id}-v{version}.jsonl'
        output_path = os.path.dirname(output_file)
        os.makedirs(output_path, exist_ok=True)
        while os.path.exists(output_file):
            version += 1
            output_file = f'{output_dir}/openseek-{task_id}-v{version}.jsonl'
        with open(output_file, 'w') as f:
            pass
        to_process = test_samples
    # =============================================================

    print(f"🚀 启动线程池，最大{max_workers}个工作线程，处理{len(to_process)}个样本")
    # 使用线程池连续处理样本
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for test_sample in to_process:
            future = executor.submit(
                process_sample,
                test_sample=test_sample,
                task_description=task_description,
                icl_examples=icl_examples,
                args=args,
                output_file=output_file,
                task_id=task_id
            )
            futures.append(future)

        # 等待所有任务完成，显示实时进度
        for future in tqdm(futures, desc=f'Evaluation on Task {task_id}: {task_name}'):
            try:
                future.result()
            except Exception as e:
                print(f"❌ 处理样本出错: {str(e)}")

    # 任务8处理完成后合并去重结果，重新写入
    if task_id == 8:
        print(f"💾 合并所有标注结果并去重...")
        all_results = {}
        # 重新读取全部结果
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        all_results[record['test_sample_id']] = record
                    except:
                        pass
        # 全部重新写入，去重
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in all_results.values():
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"✅ 结果已保存到：{output_file}，共{len(all_results)}条有效记录")



if __name__ == '__main__':
    args = parser_args()
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    evaluate(args.task_id, qwen_tokenizer, args.max_input_length, args.output_dir, args)
