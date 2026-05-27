import os
import re
import time
import json
import random
import argparse
random.seed(123)

from tqdm import tqdm, trange
from transformers import AutoTokenizer

from method import select_examples, select_operations, refer_get_operation_for_icl, refer_get_result_for_test
from const import CONST_TASK_FILES, CONST_TASK_OUTPUT_MAX_TOKENS_SIZES, CONST_LIMIT_PROMPT_MAX_TOKENS, CONST_LIMIT_OUTPUT_MAX_TOKENS, CONST_TASK_TEST_RETURN_DATA_TYPES


# 文件目录: {your-project-path}/OpenSeek/openseek/competition/LongContext-ICL-Annotation
LongContext_ICL_Annotation_DIR = os.path.dirname(os.path.dirname(__file__))

data_folder = 'data'
operation_folder = 'operations_tmp'
result_folder = 'outputs'

get_task_filepath = lambda task_id: os.path.join(LongContext_ICL_Annotation_DIR, data_folder, CONST_TASK_FILES[task_id])
get_operation_filepath = lambda task_id: os.path.join(LongContext_ICL_Annotation_DIR, operation_folder, f'operation-{task_id}.json')
get_result_filepath = lambda task_id, version: os.path.join(LongContext_ICL_Annotation_DIR, result_folder, f'openseek-{task_id}-v{version}.jsonl')

get_icl_progress_filepath = lambda task_id: os.path.join(LongContext_ICL_Annotation_DIR, f'icl_progress-{task_id}.log')
get_test_progress_filepath = lambda task_id: os.path.join(LongContext_ICL_Annotation_DIR, f'test_progress-{task_id}.log')


# 记录ICL提取执行进度
def print_icl_progress(task_id, *args):
    print(args)

# 记录TEST执行进度
def print_test_progress(task_id, *args):
    print(args)


# 利用ICL提取特定问题的关键操作Operation，可以重复执行多轮，每一轮使用不同的ICL样本组合批次
def processing(
    task_id:int,
    qwen_tokenizer:AutoTokenizer,
    start=0,
    stop=-1
):
    assert 1 <= task_id <= 8, f"task_id should be in [1,2,3,4,5,6,7,8], but got {task_id}."
    assert stop < 0 or stop >= start, f"must be `stop < 0 or stop >= start`, stop error: {stop}"

    # 保存Operation
    operation_file = get_operation_filepath(task_id)
    print_icl_progress(task_id, 'operation_file', operation_file)

    task_operations = set()
    operation_dir = os.path.dirname(operation_file)
    os.makedirs(operation_dir, mode=0o755, exist_ok=True)

    task_file = get_task_filepath(task_id)
    print_icl_progress(task_id, 'task_file', task_file)
    with open(task_file, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    icl_examples = task_dict['examples']
    print_icl_progress(task_id, 'task_name', task_name)
    print_icl_progress(task_id, 'icl_examples-size', len(icl_examples))

    # 遍历所有示例样本，自动计算合适长度的批次
    start = start if start >= 0 else 0
    stop = stop if stop >= start else len(icl_examples) - 1
    while start <= stop:

        icl_batch_data, start_next = select_examples(icl_examples, qwen_tokenizer, start, stop, CONST_LIMIT_OUTPUT_MAX_TOKENS)

        print("\n")
        print_icl_progress(task_id, 'icl_batch_data-strlen', len(icl_batch_data), "start", start, "batch_size", start_next - start, "task", task_id, task_name)
        start = start_next

        operations, state = refer_get_operation_for_icl(task_description, icl_batch_data, CONST_LIMIT_OUTPUT_MAX_TOKENS)

        print("\n--------------------------------------------------------------------------------------------------------------------")
        if operations == None:
            print_icl_progress(task_id, f"The batch get operations is None, state: {state}", )
        else:
            print_icl_progress(task_id, f"The batch get operations is Success, state: {state}", operations)
            for v in operations:
                task_operations.add(v)

    # 写入持久化文件
    operations_list = [v for v in task_operations]
    print_icl_progress(task_id, 'operations_list-size', len(operations_list))

    if len(operations_list) > 0:
        with open(operation_file, 'w') as f:
            json.dump(operations_list, f)

    print_icl_progress(task_id, "*"*50)


# 利用特定问题的Operation获取测试数据的推理结果
def evaluate(
    task_id:int,
    qwen_tokenizer:AutoTokenizer,
    start=0,
    stop=-1
):
    assert 1 <= task_id <= 8, f"task_id should be in [1,2,3,4,5,6,7,8], but got {task_id}."
    assert stop < 0 or stop >= start, f"must be `stop < 0 or stop >= start`, stop error: {stop}"

    # 加载Operation
    operation_file = get_operation_filepath(task_id)
    print_test_progress(task_id, 'operation_file', operation_file)
    with open(operation_file, 'r') as f:
        task_operations = json.load(f)

    # 无论有多少关键操作，都只提交推理一次测试，所以也只遍历一次关键操作，简单获取足量的关键操作即可
    random.shuffle(task_operations)
    core_operations, _ = select_operations(task_operations, qwen_tokenizer, 0, len(task_operations) - 1, CONST_LIMIT_PROMPT_MAX_TOKENS)
    print_test_progress(task_id, 'core_operations-strlen', len(core_operations))

    # 加载任务文件
    task_file = get_task_filepath(task_id)
    with open(task_file, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    test_samples = task_dict['test_samples']
    print_test_progress(task_id, 'test_samples-size', len(test_samples), "task", task_id, task_name)

    # 测试结果保存位置
    version = 1
    result_file = get_result_filepath(task_id, version)
    result_dir = os.path.dirname(result_file)
    os.makedirs(result_dir, mode=0o755, exist_ok=True)
    while os.path.exists(result_file):
        version += 1
        result_file = get_result_filepath(task_id, version)
    with open(result_file, 'w') as f:
        pass

    # 提交测试
    start = start if start >= 0 else 0
    stop = stop if stop >= start else len(test_samples) - 1
    for i in range(start, stop + 1):
        print(f'Evaluation item={i} on Task {task_id}: {task_name}')

        test_sample_id = test_samples[i]['id']
        test_input = test_samples[i]['input']
        prediction, state = refer_get_result_for_test(task_description, test_input, core_operations, CONST_TASK_TEST_RETURN_DATA_TYPES[task_id], CONST_TASK_OUTPUT_MAX_TOKENS_SIZES[task_id])

        print("\n--------------------------------------------------------------------------------------------------------------------")
        print_test_progress(task_id, f"Answer state={state} item={i}:", prediction)

        # 保存测试结果
        test_record = {'test_sample_id': test_sample_id, "state": state, 'reps': 0, 'prediction': prediction}
        with open(result_file, 'a') as f:
            f.write(json.dumps(test_record)+'\n')

    print_test_progress(task_id, "*"*50)


# 命令参数解析
def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID, one of [1,2,3,4,5,6,7,8]')
    parser.add_argument('--task_step', type=int, required=True,
                        help='Task Step, one of [1,2], 1 for processing step, 2 for evaluate step')

    # 可定位某个ICL批次或某个TEST测试
    parser.add_argument('--start', type=int, default=0, help='Last Stop Task Row Index[0,len-1].')
    parser.add_argument('--stop', type=int, default=-1, help='Next Stop Task Row Index[0,len-1].')

    # 分词器参数位置路径
    default_tokenizer_path = os.path.join(os.path.expanduser('~'), '.cache/modelscope/hub/models/Qwen/Qwen3-4B')
    parser.add_argument('--tokenizer_path', type=str, default=default_tokenizer_path, help='Model path for serve.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()

    tokenizer_path = os.environ.get('SERVE_TOKENIZER_PATH', args.tokenizer_path)
    qwen_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 首先提取Operation
    if args.task_step == 1:
        processing(args.task_id, qwen_tokenizer, args.start, args.stop)

    # 然后提交测试
    if args.task_step == 2:
        evaluate(args.task_id, qwen_tokenizer, args.start, args.stop)
