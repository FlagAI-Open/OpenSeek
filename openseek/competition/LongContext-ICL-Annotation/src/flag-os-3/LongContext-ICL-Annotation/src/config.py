import os

# 基础路径配置 - 所有数据统一放在项目根目录的data下
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 数据目录
DATA_DIR = os.path.join(BASE_PATH, 'data')
DATA_PROCESSED_DIR = os.path.join(BASE_PATH, 'data_processed')
OUTPUT_DIR = os.path.join(BASE_PATH, 'outputs')
CONFIG_DIR = os.path.join(BASE_PATH, 'config')

# nanobot配置
NANOBOT_HOME = CONFIG_DIR

# 各任务数据文件路径
TASK_FILES = {
    1: os.path.join(DATA_DIR, 'openseek-1_closest_integers.json'),
    2: os.path.join(DATA_DIR, 'openseek-2_count_nouns_verbs.json'),
    3: os.path.join(DATA_DIR, 'openseek-3_collatz_conjecture.json'),
    4: os.path.join(DATA_DIR, 'openseek-4_conala_concat_strings.json'),
    5: os.path.join(DATA_DIR, 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json'),
    6: os.path.join(DATA_DIR, 'openseek-6_mnli_same_genre_classification.json'),
    7: os.path.join(DATA_DIR, 'openseek-7_jeopardy_answer_generation_all.json'),
    8: os.path.join(DATA_DIR, 'openseek-8_kernel_generation.json'),
}

# 预处理后的数据路径
PROCESSED_TASK_FILES = {
    2: os.path.join(DATA_PROCESSED_DIR, 'openseek-2_count_nouns_verbs_tagged.json'),
    7: os.path.join(DATA_PROCESSED_DIR, 'openseek-7_jeopardy_answer_generation_all_tagged.json'),
}

# 缓存路径
TASK5_CACHE_PATH = os.path.join(DATA_PROCESSED_DIR, 'task5_sadness_cache')

# 输出路径配置
TASK8_OUTPUT_V1 = os.path.join(OUTPUT_DIR, 'openseek-8-v1.jsonl')
TASK8_OUTPUT_V2 = os.path.join(OUTPUT_DIR, 'openseek-8-v2.jsonl')

# 自动创建目录
for dir_path in [DATA_DIR, DATA_PROCESSED_DIR, OUTPUT_DIR, CONFIG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
