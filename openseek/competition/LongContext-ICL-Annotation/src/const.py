# 限制输出长度
CONST_LIMIT_PROMPT_MAX_TOKENS = 8192

# 限制输出长度
CONST_LIMIT_OUTPUT_MAX_TOKENS = 10240

# 调试实时打印流式输出
CONST_DEBUG_SHOW_STREAM = 1

# 调试打印正文结果（排除think后的answer)
CONST_DEBUG_SHOW_ANSWER = 2

# 任务数据文件名称
CONST_TASK_FILES = {
    1: 'openseek-1_closest_integers.json',
    2: 'openseek-2_count_nouns_verbs.json',
    3: 'openseek-3_collatz_conjecture.json',
    4: 'openseek-4_conala_concat_strings.json',
    5: 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
    6: 'openseek-6_mnli_same_genre_classification.json',
    7: 'openseek-7_jeopardy_answer_generation_all.json',
    8: 'openseek-8_kernel_generation.json',
}

# 模型输出tokens长度限制
CONST_TASK_OUTPUT_MAX_TOKENS_SIZES = {
    1: 8192 + 8,
    2: 8192 + 8,
    3: 8192 + 128,
    4: 8192 + 128,
    5: 8192 + 16,
    6: 8192 + 16,
    7: 8192 + 128,
    8: 8192 + 2048,
}

# 输出答案的字符串长度范围（注：既不包含think长度，也不是tokens长度）
CONST_TASK_CHECK_STRLEN_SIZES = {
    1: { 'min': 1, 'max': 8 },
    2: { 'min': 1, 'max': 8 },
    3: { 'min': 1, 'max': 256 },
    4: { 'min': 1, 'max': 256 },
    5: { 'min': 1, 'max': 32 },
    6: { 'min': 1, 'max': 4 },
    7: { 'min': 1, 'max': 256 },
    8: { 'min': 384, 'max': 10240 },
}

# TEST预测返回类型
CONST_TASK_TEST_RETURN_DATA_TYPES = {
    1: 'number',    # 例如：12, 3, 5, -1, -0.36 等数字
    2: 'number',
    3: 'array',     # 例如: [1,2,3], ['a','b', 'c'] 等数组
    4: 'text',      # 任意英文字符串
    5: 'text',
    6: 'text',
    7: 'phrase',    # 英文短语
    8: 'code',      # 编程代码
}
