#!/usr/bin/env python
"""
给任务2样本打分类标签，按照verbs/nouns类型 + 是否包含ing结尾单词
标签格式："{type},{ing}"，例如 "verbs,ing"、"nouns,noing"
"""
import json
import re
from collections import Counter
from config import TASK_FILES, PROCESSED_TASK_FILES

def main():
    # 从统一配置读取路径
    input_path = TASK_FILES[2]
    output_path = PROCESSED_TASK_FILES[2]

    print(f"加载原始数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 正则匹配规则
    type_pattern = re.compile(r"Count the number of (verbs|nouns)")
    sentence_pattern = re.compile(r"Sentence: '(.+?)'\. Count")
    # 匹配以ing结尾的单词（忽略大小写，匹配单词边界）
    ing_pattern = re.compile(r"\b\w+ing\b", re.IGNORECASE)

    def tag_sample(sample):
        """给单个样本打标签"""
        input_text = sample['input']

        # 1. 提取任务类型（verbs/nouns）
        type_match = type_pattern.search(input_text)
        if not type_match:
            print(f"⚠️ 无法识别任务类型: {input_text[:100]}...")
            sample['tag'] = "unknown"
            return sample
        task_type = type_match.group(1)

        # 2. 提取句子内容
        sentence_match = sentence_pattern.search(input_text)
        if not sentence_match:
            print(f"⚠️ 无法提取句子: {input_text[:100]}...")
            sample['tag'] = f"{task_type},unknown"
            return sample
        sentence = sentence_match.group(1)

        # 3. 检查是否有ing结尾的单词
        has_ing = bool(ing_pattern.search(sentence))
        ing_tag = "ing" if has_ing else "noing"

        # 4. 组合最终标签：类型,ing标记
        sample['tag'] = f"{task_type},{ing_tag}"

        # 额外标记：verb+ing+结果为0的样本标记为useless
        output_val = sample['output'][0].strip() if sample.get('output') else ''
        sample['useless'] = (task_type == 'verbs' and has_ing and output_val == '0')

        return sample

    # 处理examples
    print(f"\n处理examples: {len(data.get('examples', []))}个样本")
    tagged_examples = []
    for sample in data.get('examples', []):
        tagged_examples.append(tag_sample(sample))
    data['examples'] = tagged_examples

    # 处理test_samples
    print(f"处理test_samples: {len(data.get('test_samples', []))}个样本")
    tagged_tests = []
    for sample in data.get('test_samples', []):
        tagged_tests.append(tag_sample(sample))
    data['test_samples'] = tagged_tests

    # 统计标签分布
    print(f"\n📊 标签分布统计:")
    all_samples = tagged_examples + tagged_tests
    all_tags = [s['tag'] for s in all_samples]
    tag_counts = Counter(all_tags)
    for tag, count in sorted(tag_counts.items()):
        ratio = count / len(all_tags) * 100
        print(f"  {tag:<15} {count:4d}  {ratio:.1f}%")

    # 统计useless样本数量
    useless_count = sum(1 for s in all_samples if s.get('useless', False))
    print(f"\n🗑️  useless样本(verb+ing+结果为0)总数: {useless_count} 个")

    # 保存带标签的数据
    print(f"\n保存带标签数据到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ 处理完成！")

    # 输出几个示例验证
    print(f"\n🔍 示例验证:")
    for i, sample in enumerate(tagged_examples[:3]):
        print(f"\n示例{i+1}:")
        print(f"  Input: {sample['input'][:150]}...")
        print(f"  Tag: {sample['tag']}")
        print(f"  Output: {sample['output']}")

    # 筛选并打印verb+ing+结果为0的样本，同时提取ing词语
    print(f"\n\n📋 筛选 verb类型 + 带ing + 统计结果为0 的样本(提取ing词语):")
    print(f"{'='*100}")
    matched_samples = []
    all_samples = tagged_examples + tagged_tests
    # 正则匹配ing结尾的单词
    ing_pattern = re.compile(r"\b\w+ing\b", re.IGNORECASE)
    sentence_pattern = re.compile(r"Sentence: '(.+?)'\. Count")

    for sample in all_samples:
        tag = sample.get('tag', '')
        output = sample.get('output', [''])[0].strip()
        if tag == 'verbs,ing' and output == '0':
            # 提取句子内容
            sentence_match = sentence_pattern.search(sample['input'])
            sentence = sentence_match.group(1) if sentence_match else sample['input']
            # 提取所有ing词语
            ing_words = ing_pattern.findall(sentence)
            sample['ing_words'] = ing_words
            matched_samples.append(sample)

    print(f"共找到 {len(matched_samples)} 个符合条件的样本:\n")
    for i, sample in enumerate(matched_samples[:20], 1):  # 最多打印前20个
        print(f"【{i}/{len(matched_samples)}】")
        print(f"句子: {sample['input'].split('Count the number')[0]}")
        print(f"提取到的ing词语: {', '.join(sample['ing_words'])}")
        print(f"输出: {sample['output'][0]}")
        print(f"{'-'*80}")

    if len(matched_samples) > 20:
        print(f"...... 还有 {len(matched_samples) - 20} 个样本未显示\n")
    print(f"✅ verb+ing+结果为0 样本总数: {len(matched_samples)} 个")

    # 统计所有ing词语的出现频率
    print(f"\n📊 这些样本中ing词语出现频率统计(前15):")
    all_ing_words = []
    for sample in matched_samples:
        all_ing_words.extend([w.lower() for w in sample['ing_words']])
    word_counts = Counter(all_ing_words)
    for word, count in word_counts.most_common(15):
        print(f"  {word:<15} 出现 {count} 次")

    # 筛选并打印verb+ing+结果不为0的样本
    print(f"\n\n📋 筛选 verb类型 + 带ing + 统计结果不为0 的样本:")
    print(f"{'='*100}")
    non_zero_samples = []
    for sample in all_samples:
        tag = sample.get('tag', '')
        output = sample.get('output', [''])[0].strip()
        if tag == 'verbs,ing' and output != '0':
            non_zero_samples.append(sample)

    print(f"共找到 {len(non_zero_samples)} 个符合条件的样本:\n")
    for i, sample in enumerate(non_zero_samples[:10], 1):  # 最多打印前10个
        print(f"【{i}/{len(non_zero_samples)}】")
        print(f"ID: {sample['id']}")
        print(f"输入: {sample['input']}")
        print(f"输出: {sample['output'][0]}")
        print(f"{'-'*80}")

    if len(non_zero_samples) > 10:
        print(f"...... 还有 {len(non_zero_samples) - 10} 个样本未显示\n")
    print(f"✅ verb+ing+结果不为0 样本总数: {len(non_zero_samples)} 个")

if __name__ == "__main__":
    main()
