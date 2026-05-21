"""验证V8代码的正确性"""
import json
import sys
sys.path.insert(0, r'C:\D\compet\dcic\FlagOS开放计算全球挑战赛\OpenSeek\openseek\competition\LongContext-ICL-Annotation\TOPGO\src')

from method import (
    deterministic_compute, build_prompt, select_examples, 
    _select_examples_balanced, TASK_CONFIG, count_answer
)

DATA_DIR = r'C:\D\compet\dcic\FlagOS开放计算全球挑战赛\OpenSeek\openseek\competition\LongContext-ICL-Annotation\data'

print("=" * 60)
print("V8 代码验证")
print("=" * 60)

# 1. 验证确定性计算
print("\n--- 1. 确定性计算验证 ---")
t1_result = deterministic_compute(1, '[59, 26, -96, -30]')
print(f"Task1: [59, 26, -96, -30] -> {t1_result} (expected: 33) {'✓' if t1_result == '33' else '✗'}")

t1_result2 = deterministic_compute(1, '[-3, 20, 0, -27, -38, 98]')
print(f"Task1: [-3, 20, 0, -27, -38, 98] -> {t1_result2} (expected: 3) {'✓' if t1_result2 == '3' else '✗'}")

t3_result = deterministic_compute(3, '[72, 29, 49]')
print(f"Task3: [72, 29, 49] -> {t3_result} (expected: [36, 88, 148]) {'✓' if t3_result == '[36, 88, 148]' else '✗'}")

t4_result = deterministic_compute(4, "['p', 'that.', 'o']")
print(f"Task4: ['p', 'that.', 'o'] -> {t4_result} (expected: pthat.o) {'✓' if t4_result == 'pthat.o' else '✗'}")

# 2. 验证Prompt构建
print("\n--- 2. Prompt构建验证 ---")
for tid in range(1, 9):
    p = build_prompt("Test task description", "Test input", task_id=tid)
    print(f"Task{tid} prompt length: {len(p)} chars {'✓' if len(p) > 100 else '✗'}")

# 3. 验证平衡示例选择 (Task 5)
print("\n--- 3. 平衡示例选择验证 ---")
with open(f'{DATA_DIR}/openseek-5_semeval_2018_task1_tweet_sadness_detection.json', 'r', encoding='utf-8') as f:
    d5 = json.load(f)

result5 = _select_examples_balanced(d5['examples'][:200], 28000, None, 'Sad', 'Not sad')
sad_count = result5.count('Sad')
notsad_count = result5.count('Not sad')
total_chars5 = len(result5)
ratio5 = min(sad_count, notsad_count) / max(sad_count, notsad_count) if max(sad_count, notsad_count) > 0 else 0
print(f"Task5: Sad={sad_count}, Not sad={notsad_count}, Ratio={ratio5:.2f}, Chars={total_chars5}")
print(f"  Balance: {'✓' if ratio5 > 0.7 else '✗'} (ratio > 0.7)")

# 4. 验证平衡示例选择 (Task 6)
with open(f'{DATA_DIR}/openseek-6_mnli_same_genre_classification.json', 'r', encoding='utf-8') as f:
    d6 = json.load(f)

result6 = _select_examples_balanced(d6['examples'][:200], 28000, None, 'Y', 'N')
y_count = result6.count('Y')
n_count = result6.count('N')
total_chars6 = len(result6)
ratio6 = min(y_count, n_count) / max(y_count, n_count) if max(y_count, n_count) > 0 else 0
print(f"Task6: Y={y_count}, N={n_count}, Ratio={ratio6:.2f}, Chars={total_chars6}")
print(f"  Balance: {'✓' if ratio6 > 0.7 else '✗'} (ratio > 0.7)")

# 5. 验证任务配置
print("\n--- 4. 任务配置验证 ---")
for tid in range(1, 9):
    config = TASK_CONFIG[tid]
    print(f"Task{tid}: temp={config['temperature']}, max_tokens={config['max_tokens']}, target_length={config['target_length']}")

# 6. 验证count_answer
print("\n--- 5. count_answer验证 ---")
test_text = "Some reasoning here <label>Sad</label> more text"
result = count_answer(test_text)
print(f"count_answer test: '{test_text}' -> {result} {'✓' if result == 'Sad' else '✗'}")

test_text2 = "<label>33</label>"
result2 = count_answer(test_text2)
print(f"count_answer test: '{test_text2}' -> {result2} {'✓' if result2 == '33' else '✗'}")

# 7. 批量验证确定性计算在真实数据上的表现
print("\n--- 6. 批量验证确定性计算 ---")
for tid in [1, 3, 4]:
    filenames = {
        1: 'openseek-1_closest_integers.json',
        3: 'openseek-3_collatz_conjecture.json',
        4: 'openseek-4_conala_concat_strings.json',
    }
    with open(f'{DATA_DIR}/{filenames[tid]}', 'r', encoding='utf-8') as f:
        d = json.load(f)
    
    success = 0
    total = min(20, len(d['test_samples']))
    for i in range(total):
        sample = d['test_samples'][i]
        result = deterministic_compute(tid, sample['input'])
        if result is not None:
            success += 1
    
    print(f"Task{tid}: {success}/{total} samples computed successfully {'✓' if success == total else '✗'}")

print("\n" + "=" * 60)
print("V8 验证完成!")
print("=" * 60)
