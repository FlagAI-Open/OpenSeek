"""paths.py — COM 提交目录的路径清单

复现时只需按本机环境改下面几个绝对路径赋值即可，其余路径都从这几个根路径
拼出来，不会再有"漏改一处"的问题。

使用方式::

    from common.paths import (
        DATA_DIR, MODEL_DIR, FINAL_OUTPUT_DIR,
        VLLM_BASE_URL, VLLM_MODEL_ID,
        task_data_file, task_cot_dir, task_log_dir,
        final_output_file,
    )
"""


# ---------------------------------------------------------------------------
# 复现时改这几行 ↓↓↓
# ---------------------------------------------------------------------------
COM_ROOT = '/home/blue/dev4T/flagOS/code/OpenSeek/openseek/competition/LongContext-ICL-Annotation/COM'

DATA_DIR         = COM_ROOT + '/data'                  # 原始数据：openseek-{1..8}_*.json
MODEL_DIR        = COM_ROOT + '/models/Qwen/Qwen3-4B'  # Qwen3-4B 权重
FINAL_OUTPUT_DIR = COM_ROOT + '/outputs'               # 最终交付 jsonl 目录
SRC_DIR          = COM_ROOT + '/src'                   # 源码目录
ENV_DIR          = COM_ROOT + '/env'                   # 环境/服务/补丁配置目录

VLLM_BASE_URL    = 'http://localhost:2026/v1'          # vLLM OpenAI 兼容端点
VLLM_MODEL_ID    = '../models/Qwen/Qwen3-4B'           # vLLM 启动时的 model 标识（依赖 cwd）
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 以下都是基于上面根路径的简单拼接，一般不用动
# ---------------------------------------------------------------------------

def task_dir(task_id: int) -> str:
    """``COM/src/task<N>``"""
    return f'{SRC_DIR}/task{task_id}'


def task_cot_dir(task_id: int) -> str:
    """``COM/src/task<N>/cot_data``"""
    return f'{task_dir(task_id)}/cot_data'


def task_log_dir(task_id: int) -> str:
    """``COM/src/task<N>/logs``"""
    return f'{task_dir(task_id)}/logs'


def final_output_file(task_id: int, version: int = 1) -> str:
    """``COM/outputs/openseek-<N>-v<version>.jsonl``（最终交付名；main.py 自动按已存在版本递增，首次跑即 v1）"""
    return f'{FINAL_OUTPUT_DIR}/openseek-{task_id}-v{version}.jsonl'


# Task id → 原始数据文件名映射（与 COM/data/ 实际文件对齐）
_TASK_DATA_FILENAMES = {
    1: 'openseek-1_closest_integers.json',
    2: 'openseek-2_count_nouns_verbs.json',
    3: 'openseek-3_collatz_conjecture.json',
    4: 'openseek-4_conala_concat_strings.json',
    5: 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: 'openseek-6_mnli_same_genre_classification.json',
    7: 'openseek-7_jeopardy_answer_generation_all.json',
}

# Task 8 离线 normalize 时使用的原始数据
TASK8_RAW_DATA_FILENAME = 'openseek-8_kernel_generation.json'


def task_data_file(task_id: int) -> str:
    """``COM/data/openseek-<N>_*.json``"""
    return f'{DATA_DIR}/{_TASK_DATA_FILENAMES[task_id]}'


__all__ = [
    'COM_ROOT', 'SRC_DIR', 'ENV_DIR',
    'DATA_DIR', 'MODEL_DIR', 'FINAL_OUTPUT_DIR',
    'VLLM_BASE_URL', 'VLLM_MODEL_ID',
    'task_dir', 'task_cot_dir', 'task_log_dir',
    'task_data_file', 'final_output_file',
]


if __name__ == '__main__':
    # 自检：python -m common.paths（前提：sys.path 含 COM/src）
    print(f'COM_ROOT          = {COM_ROOT}')
    print(f'DATA_DIR          = {DATA_DIR}')
    print(f'MODEL_DIR         = {MODEL_DIR}')
    print(f'FINAL_OUTPUT_DIR  = {FINAL_OUTPUT_DIR}')
    print(f'VLLM_BASE_URL     = {VLLM_BASE_URL}')
    print(f'VLLM_MODEL_ID     = {VLLM_MODEL_ID}')
    for tid in range(1, 9):
        if tid in _TASK_DATA_FILENAMES:
            print(f'  task{tid}: data={task_data_file(tid)}')
        else:
            print(f'  task{tid}: data=<task-local, see src/task{tid}/>')
        print(f'         final={final_output_file(tid)}')
