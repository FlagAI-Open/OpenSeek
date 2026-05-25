#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, 'src')

from method import build_prompt, select_examples, annotate_ascend
from transformers import AutoTokenizer

# 测试第一个任务
task_file = './data/openseek-1_closest_integers.json'
with open(task_file, 'r') as f:
    task_dict = json.load(f)

task_description = task_dict['Definition'][0]
icl_examples = task_dict['examples'][:5]  # 只用5个示例
test_samples = task_dict['test_samples'][:1]  # 第一个测试样本

print(f"Task description: {task_description[:100]}...")
print(f"Number of examples: {len(icl_examples)}")
print(f"Test sample input: {test_samples[0]['input'][:100]}...")

# 构建prompt
text2annotate = test_samples[0]['input']
prompt = build_prompt(task_description, text2annotate)

# 选择示例（简化版）
class Args:
    tokenizer_path = '/FlagRelease/Qwen3-4B-FlagOS-Ascend/'

args = Args()
# 使用select_examples函数
examples_str = select_examples(icl_examples, task_description, text2annotate, args)
input_prompt = prompt.replace("[[EXAMPLES]]\n\n", examples_str + '\n\n')

print("\n" + "="*80)
print("Generated prompt (first 500 chars):")
print(input_prompt[:500])
print("..." if len(input_prompt) > 500 else "")
print("="*80 + "\n")

# 调用标注函数
print("Calling annotate_ascend...")
try:
    prediction = annotate_ascend(input_prompt)
    print(f"Prediction: {prediction}")
    print(f"Prediction type: {type(prediction)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()