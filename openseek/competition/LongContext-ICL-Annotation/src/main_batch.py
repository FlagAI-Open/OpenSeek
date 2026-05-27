import os
import re
import json
import time
import random
import shutil
import argparse
random.seed(123)

from tqdm import tqdm, trange
from transformers import AutoTokenizer

# 必须强制 DEBUG_PRINT_STREAM=0 防止在后台运行时实时打印模型的输出内容，会拖慢速度，而且实时打印是胡乱无序的
os.environ['DEBUG_PRINT_STREAM'] = '0'

from method import select_examples, select_operations, parse_result, batch_refer_get_operation_for_icl, batch_refer_get_result_for_test
from const import CONST_TASK_FILES, CONST_TASK_OUTPUT_MAX_TOKENS_SIZES, CONST_LIMIT_PROMPT_MAX_TOKENS, CONST_LIMIT_OUTPUT_MAX_TOKENS, CONST_TASK_CHECK_STRLEN_SIZES, CONST_TASK_TEST_RETURN_DATA_TYPES


# 提取轮次（若NPU卡较少，为了加速实验可选择默认N=1，若NPU卡较多 可以尝试多跑几遍）
try:
    N = int(os.environ.get('ICL_REPEAT_N', 1))
except ValueError:
    N = 0

# 异步并发数（与服务器支持的批次并行相近即可, 协同服务端张量和数据并行计算, 可加速实验进度）
try:
    req_concurrency_size = int(os.environ.get('REQ_CONCURRENCY_NUM', 2))
except ValueError:
    req_concurrency_size = 0

if N <=0:
    raise Exception('ICL_REPEAT_N must be a integer and bigger than 0')
if req_concurrency_size <= 0:
    raise Exception('REQ_CONCURRENCY_NUM must be a integer and bigger than 0')


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
def log_icl_progress(task_id, *args):
    with open(get_icl_progress_filepath(task_id), 'a') as f:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(dt + ' :: ' + json.dumps(args)+'\n')


# 记录TEST执行进度
def log_test_progress(task_id, *args):
    with open(get_test_progress_filepath(task_id), 'a') as f:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(dt + ' :: ' + json.dumps(args)+'\n')


# 利用ICL提取特定问题的关键操作Operation，可以重复执行多轮，每一轮使用不同的ICL样本组合批次
def processing(task_id:int, qwen_tokenizer:AutoTokenizer, start = 0, stop = -1):
    assert 1 <= task_id <= 8, f"task_id should be in [1, 8], but got {task_id}."
    assert stop < 0 or stop >= start, f"must be `stop < 0 or stop >= start`, stop error: {stop}"

    # 保存Operation
    operation_file = get_operation_filepath(task_id)
    log_icl_progress(task_id, 'operation_file', operation_file)

    task_operations = set()

    operation_dir = os.path.dirname(operation_file)
    os.makedirs(operation_dir, mode=0o755, exist_ok=True)

    if os.path.exists(operation_file):
        if N == 1:
            log_icl_progress(task_id, 'not need repeat processing for N=1')
            return
        # 与上一轮的去重合并
        with open(operation_file, 'r') as f:
            old_operations_list = json.load(f)
            task_operations.update(old_operations_list)

    task_file = get_task_filepath(task_id)
    log_icl_progress(task_id, 'task_file', task_file)
    with open(task_file, 'r') as f:
        task_dict = json.load(f)

     task_name = task_dict['task_name']
     task_description = task_dict['Definition'][0]
     icl_examples = task_dict['examples']
     log_icl_progress(task_id, 'task_name', task_name)
     log_icl_progress(task_id, 'icl_examples-size', len(icl_examples))

    for i in range(N):
        log_icl_progress(task_id, f"第 {i + 1} 轮次ICL遍历")

        # 在重复采样时，打乱ICL的顺序，每轮即可得到不同样本组合的批次，让样本表现为多样性
        if i > 0:
            random.shuffle(icl_examples)

        # 遍历所有示例样本，自动计算合适长度的批次
        icl_batch_data_list = []
        start = start if start >= 0 else 0
        stop = stop if stop >= start else len(icl_examples) - 1
        while start <= stop:
            icl_batch_data, start_next = select_examples(icl_examples, qwen_tokenizer, start, stop, CONST_LIMIT_PROMPT_MAX_TOKENS)
            log_icl_progress(task_id, 'icl_batch_data-strlen:', len(icl_batch_data), f"N_i={i}", "start:", start, "icl_batch_size:", start_next - start, "task:", task_id, task_name)
            start = start_next

            icl_batch_data_list.append(icl_batch_data)
            if len(icl_batch_data_list) == req_concurrency_size or start >= stop:
                operations_list = batch_refer_get_operation_for_icl(task_description, icl_batch_data_list, CONST_LIMIT_OUTPUT_MAX_TOKENS)
                icl_batch_data_list = []

                for operations, state in operations_list:
                    if operations == None:
                        log_icl_progress(task_id, f"The N_i={i} req_concurrency_size: {req_concurrency_size} get operations is None, state: {state}")
                    else:
                        log_icl_progress(task_id, f"The N_i={i} req_concurrency_size: {req_concurrency_size} get operations is Success, state: {state}", operations)
                        for v in operations:
                            task_operations.add(v)

        # 写入持久化文件
        operations_list = [v for v in task_operations]
        log_icl_progress(task_id, 'operations_list-size', len(operations_list))

        if len(operations_list) > 0:
            with open(operation_file, 'w') as f:
                json.dump(operations_list, f)

        log_icl_progress(task_id, "*"*50)


# 利用特定问题的Operation获取测试数据的推理结果
def evaluate(task_id:int, qwen_tokenizer:AutoTokenizer, start = 0, stop = -1):
    assert 1 <= task_id <= 8, f"task_id should be in [1, 8], but got {task_id}."
    assert stop < 0 or stop >= start, f"must be `stop < 0 or stop >= start`, stop error: {stop}"

    # 加载Operation
    operation_file = get_operation_filepath(task_id)
    log_test_progress(task_id, 'operation_file', operation_file)
    with open(operation_file, 'r') as f:
        task_operations = json.load(f)

    # 无论有多少关键操作，都只提交推理一次测试，所以也只遍历一次关键操作，简单获取足量的关键操作即可
    random.shuffle(task_operations)
    core_operations, _ = select_operations(task_operations, qwen_tokenizer, 0, len(task_operations) - 1, CONST_LIMIT_PROMPT_MAX_TOKENS)
    log_test_progress(task_id, 'core_operations-strlen', len(core_operations))

    # 加载任务文件
    task_file = get_task_filepath(task_id)
    with open(task_file, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    test_samples = task_dict['test_samples']
    log_test_progress(task_id, 'test_samples-size', len(test_samples), "task", task_id, task_name)

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
    for i in range(start, stop + 1, req_concurrency_size):
        log_test_progress(task_id, f'Evaluation items({i}:{i+req_concurrency_size}) on Task {task_id}: {task_name}')

        test_sample_id_list = []
        test_input_list = []
        for test_sample in test_samples[i: min(i + req_concurrency_size, stop + 1)]:
            test_sample_id_list.append(test_sample['id'])
            test_input_list.append(test_sample['input'])

        prediction_list = batch_refer_get_result_for_test(task_description, test_input_list, core_operations, CONST_TASK_TEST_RETURN_DATA_TYPES[task_id], CONST_TASK_OUTPUT_MAX_TOKENS_SIZES[task_id])

        j = 0
        for test_sample_id, (prediction, state) in zip(test_sample_id_list, prediction_list):
            log_test_progress(task_id, f"Answer state={state} item={i+j}:", prediction)
            j += 1

            # 保存测试结果
            test_record = {'test_sample_id': test_sample_id, 'state': state, 'reps': 0, 'prediction': prediction}
            with open(result_file, 'a') as f:
                f.write(json.dumps(test_record)+'\n')

    log_test_progress(task_id, "*"*50)


# 自动检查异常数据：可能是服务端为正常返回，也可能是KV缓存，也可能是模型不稳定
# 像这样的异常数据，可用通过重新请求服务，最终得到正常输出
def check_evaluate_retry(task_id:int, qwen_tokenizer:AutoTokenizer, reps = 0):
    assert 1 <= task_id <= 8, f"task_id should be in [1, 8], but got {task_id}."

    # 测试结果保存位置
    version = 1
    result_file = get_result_filepath(task_id, version)
    result_dir = os.path.dirname(result_file)
    if not os.path.exists(result_dir):
        # 没有结果文件，无需检查
        print(f'task_id={task_id}暂无对应结果文件，无需检查')
        return

    min_len, max_len = CONST_TASK_CHECK_STRLEN_SIZES[task_id]['min'], CONST_TASK_CHECK_STRLEN_SIZES[task_id]['max']
    checks = {}
    outputs = []
    with open(result_file, 'r') as file:
        # 逐行读取并解析JSON
        i = 0
        for line in file:
            # 解析每行的JSON数据
            data = json.loads(line)
            outputs.append(data)
            # 检查是否满足特定条件
            data_len = len(data['prediction']) if data['prediction'] != None else 0
            if data['state'] == False or data_len < min_len or data_len > max_len:
                checks[data['test_sample_id']] = i
            # next
            i += 1

    if len(checks.values()) == 0:
        # 没有异常数据，无需检查
        print(f'task_id={task_id}没有异常数据，无需检查')
        return

    # 备份
    backup_result_file = result_file + '_backup'
    shutil.copy(result_file, backup_result_file)

    # 加载Operation
    operation_file = get_operation_filepath(task_id)
    log_test_progress(task_id, 'operation_file', operation_file)
    with open(operation_file, 'r') as f:
        task_operations = json.load(f)

    # 无论有多少关键操作，都只提交推理一次测试，所以也只遍历一次关键操作，简单获取足量的关键操作即可
    random.shuffle(task_operations)
    core_operations, _ = select_operations(task_operations, qwen_tokenizer, 0, len(task_operations) - 1, CONST_LIMIT_PROMPT_MAX_TOKENS)
    log_test_progress(task_id, 'core_operations-strlen', len(core_operations))

    # 加载任务文件
    task_file = get_task_filepath(task_id)
    with open(task_file, 'r') as f:
        task_dict = json.load(f)

    task_name = task_dict['task_name']
    task_description = task_dict['Definition'][0]
    test_samples = task_dict['test_samples']
    log_test_progress(task_id, 'test_samples-size', len(test_samples), "task", task_id, task_name)

    check_ids = checks.keys()
    indexs = [ i for i, item in enumerate(test_samples) if item['id'] in check_ids ]
    for k in range(0, len(indexs), req_concurrency_size):
        sample_idxs = indexs[k : k + req_concurrency_size]

        test_sample_id_list = []
        test_input_list = []
        for i in sample_idxs:
            test_sample_id_list.append(test_samples[i]['id'])
            test_input_list.append(test_samples[i]['input'])

        log_test_progress(task_id, f"Retry items={sample_idxs}")
        prediction_list = batch_refer_get_result_for_test(task_description, test_input_list, core_operations, CONST_TASK_TEST_RETURN_DATA_TYPES[task_id], CONST_TASK_OUTPUT_MAX_TOKENS_SIZES[task_id])

        j = 0
        for test_sample_id, (prediction, state) in zip(test_sample_id_list, prediction_list):
            raw = prediction
            if state == False:
                prediction, _ = parse_result(prediction, CONST_TASK_TEST_RETURN_DATA_TYPES[task_id])

            log_test_progress(task_id, f"Correct state={state} item={sample_idxs[j]}:", prediction)
            j += 1

            # 保存测试结果
            outputs[checks[test_sample_id]] = {'test_sample_id': test_sample_id, 'state': state, 'reps': reps, 'prediction': prediction}
            if raw != prediction:
                outputs[checks[test_sample_id]]['raw'] = raw

    new_result_file = get_result_filepath(task_id, version + 1)
    with open(new_result_file, 'w') as f:
        for data in outputs:
            f.write(json.dumps(data) + '\n')

    # 覆盖旧文件
    os.replace(new_result_file, result_file)
    log_test_progress(task_id, "*"*50)


# 命令参数解析
def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--task_id',
        type=str,
        required=True,
        help='Task ID, examples: task_id=1 or task_id=[1,2] or task_id=0, all task if task_id=0 or task_id is empty, one task if task_id>=1 and task_id<=8, multi task if task_id is list'
    )
    parser.add_argument(
        '--task_step',
        type=str,
        required=True,
        help='Task Step, examples: task_step=1 or task_step=[1,2] or task_step=0, all if task_step=0 or task_step is empty, processing if task_step=1, evaluate if task_step=2, check_evaluate_retry if task_step=3'
    )

    # 可定位某个ICL批次或某个TEST测试
    parser.add_argument('--start', type=int, default=0, help='Last Stop Task Row Index[0,len-1].')
    parser.add_argument('--stop', type=int, default=-1, help='Next Stop Task Row Index[0,len-1].')

    # 分词器参数位置路径
    default_tokenizer_path = os.path.join(os.path.expanduser('~'), '.cache/modelscope/hub/models/Qwen/Qwen3-4B')
    parser.add_argument('--tokenizer_path', type=str, default=default_tokenizer_path, help='Model path for serve.')

    return parser.parse_args()


def parse_args_task_id_or_step(arg_value: str, default_list: list):

    arg_value = arg_value.strip()
    if arg_value == '' or arg_value == '0':
        arg_out = default_list
    elif arg_value.isdigit():
        arg_out = [ int(arg_value) ]
    else:
        if arg_value.find('[') == -1:
            arg_value = '[' + arg_value
        if arg_value.find(']') == -1:
            arg_value = arg_value + ']'

        arg_out = json.loads(arg_value)

    return arg_out


if __name__ == '__main__':
    args = parser_args()

    task_step_list = parse_args_task_id_or_step(args.task_step, [1, 2, 3])
    for task_step in task_step_list:
        if not task_step in [1,2,3]:
            raise Exception(f'Task Step error: {task_step_list}')

    task_id_list = parse_args_task_id_or_step(args.task_id, [1, 2, 3, 4, 5, 6, 7, 8])
    for task_id in task_id_list:
        if not task_id in [1,2,3,4,5,6,7,8]:
            raise Exception(f'Task ID error: {task_id_list}')

    # tokenizer
    tokenizer_path = os.environ.get('SERVE_TOKENIZER_PATH', args.tokenizer_path)
    qwen_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 串行处理任务
    for task_id in task_id_list:

        # 首先提取Operation
        if 1 in task_step_list:
            processing(task_id, qwen_tokenizer, args.start, args.stop)
            args.start = 0
            args.stop = -1

        # 然后提交测试
        if 2 in task_step_list:
            evaluate(task_id, qwen_tokenizer, args.start, args.stop)
            args.start = 0
            args.stop = -1

        # 最后检查Result异常数据：可能是GPU负荷太高、GPU高温降频等无法正常完成，也可能是KV缓存干扰，也可能是模型输出不稳定，可能是网络问题等等
        if 3 in task_step_list:
            # 检查3遍
            for i in range(3):
                reps = i + 1
                check_evaluate_retry(task_id, qwen_tokenizer, reps)
                time.sleep(30 * reps)
