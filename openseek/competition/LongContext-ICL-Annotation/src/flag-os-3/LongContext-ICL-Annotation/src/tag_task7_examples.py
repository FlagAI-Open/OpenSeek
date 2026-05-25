#!/usr/bin/env python
"""
给Task7的jeopardy样本打分类标签，用于后续按标签过滤few-shot
"""
import json
import random
import os
from tqdm import tqdm
from nanobot import Nanobot
import asyncio
from config import TASK_FILES, PROCESSED_TASK_FILES

# 标签体系
TAGS = [
    "auto_sports",       # 汽车赛事
    "food_dessert",      # 食物/甜点/果蔬
    "military",          # 军事相关
    "geography",         # 地理/城市/国家/岛屿/景区
    "celebrity_people",  # 名人/演员/诗人/公众人物
    "literature",        # 文学/小说/莎士比亚
    "bible_religion",    # 圣经/宗教
    "science",           # 化学/物理/生物/理科常识
    "history",           # 历史/帝王/考古/近代史
    "proverb",           # 谚语改错
    "language_trans",    # 外语释义/拉丁语翻译
    "animal",            # 动物/宠物/昆虫
    "number_trivia",     # 数字类常识
    "title_quote",       # 歌名/书名/名言（带引号答案）
    "anatomy_medical",   # 人体解剖/医学
    "daily_life",        # 生活常识/服饰/美妆/饮食
    "politics",          # 总统/政坛人物
    "other"              # 其他
]

TAG_DESCRIPTIONS = {
    "auto_sports": "汽车赛车、赛事相关内容",
    "food_dessert": "食物、甜点、水果、蔬菜、饮品等饮食相关",
    "military": "军事、武器、军队、战争相关内容",
    "geography": "地理、城市、国家、岛屿、景点、位置相关",
    "celebrity_people": "名人、演员、歌手、诗人、政治家、公众人物",
    "literature": "文学、小说、书籍、作者、诗歌、莎士比亚等",
    "bible_religion": "圣经、宗教、神话、信仰相关内容",
    "science": "科学、化学、物理、生物、数学、理科常识",
    "history": "历史事件、帝王、考古、近代史、古代史相关",
    "proverb": "谚语、俗语、俗语改错相关",
    "language_trans": "外语翻译、拉丁语、词汇释义、语言相关",
    "animal": "动物、宠物、昆虫、生物物种相关",
    "number_trivia": "数字相关的常识、数学趣味题",
    "title_quote": "歌曲名、书名、电影名、名言警句，带引号答案",
    "anatomy_medical": "人体解剖、医学、疾病、医药相关",
    "daily_life": "日常生活常识、服饰、美妆、饮食、生活用品",
    "politics": "总统、政府、政坛人物、政治事件相关",
    "other": "不属于以上分类的内容"
}

# 标签system prompt
TAG_PROMPT_TEMPLATE = """
# Task: Classify Jeopardy Trivia Question

Given a trivia question with its category and clue, classify it into one of the following tags:

## Available Tags and Descriptions:
{tag_list}

## Input:
{input_text}

## Requirements:
1. Choose the most appropriate tag from the list above
2. If none fit, choose "other"
3. Output ONLY the tag name wrapped in <label> tags, no explanations
"""

def build_tag_prompt(input_text: str) -> str:
    """构建分类prompt"""
    tag_list = "\n".join([f"- {tag}: {desc}" for tag, desc in TAG_DESCRIPTIONS.items()])
    return TAG_PROMPT_TEMPLATE.format(tag_list=tag_list, input_text=input_text)

async def classify_sample(sample: dict) -> str:
    """给单个样本分类"""
    input_text = sample['input']
    prompt = build_tag_prompt(input_text)

    try:
        from config import NANOBOT_HOME
        bot = Nanobot.from_config(f'{NANOBOT_HOME}/config.json',use_tools=False)
        result = await bot.run(prompt)

        # 提取标签
        content = result.content
        if '<label>' in content and '</label>' in content:
            tag = content.split('<label>')[1].split('</label>')[0].strip()
            # 验证标签是否有效
            if tag in TAGS:
                return tag
        return "other"
    except Exception as e:
        print(f"分类失败: {str(e)}")
        return "other"

def extract_answer(text: str) -> str:
    """提取<label>标签中的内容"""
    if '<label>' in text and '</label>' in text:
        return text.split('<label>')[1].split('</label>')[0].strip()
    return ""

async def batch_classify(samples: list, max_workers: int = 10) -> list:
    """批量分类样本"""
    semaphore = asyncio.Semaphore(max_workers)
    results = []

    async def process_sample(sample):
        async with semaphore:
            tag = await classify_sample(sample)
            sample_with_tag = sample.copy()
            sample_with_tag['tag'] = tag
            return sample_with_tag

    tasks = [process_sample(sample) for sample in samples]
    for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="标注进度"):
        results.append(await result)

    return results

def save_tagged_data(tagged_samples: list, output_path: str):
    """保存标注后的数据"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tagged_samples, f, ensure_ascii=False, indent=2)

def load_data(data_path: str) -> dict:
    """加载原始数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=TASK_FILES[7], help='原始数据路径')
    parser.add_argument('--num_examples', type=int, default=0, help='随机抽取标注的examples数量，0表示标注全部')
    parser.add_argument('--output_path', default=PROCESSED_TASK_FILES[7], help='输出文件路径')
    parser.add_argument('--workers', type=int, default=10, help='并发数量')
    parser.add_argument('--only_tag_tests', action='store_true', help='只标注test_samples，不标注examples')
    args = parser.parse_args()

    # 加载数据
    print(f"加载数据: {args.data_path}")
    data = load_data(args.data_path)
    all_examples = data.get('examples', [])
    test_samples = data.get('test_samples', [])

    print(f"总examples数量: {len(all_examples)}")
    print(f"test_samples数量: {len(test_samples)}")

    # 处理examples
    tagged_examples = all_examples
    if not args.only_tag_tests:
        if args.num_examples > 0:
            # 随机抽取N个examples
            random.seed(42)
            selected_indices = random.sample(range(len(all_examples)), min(args.num_examples, len(all_examples)))
            selected_examples = [all_examples[i] for i in selected_indices]
            print(f"随机抽取了 {len(selected_examples)} 个examples用于标注")

            # 标注抽取的examples
            print("\n开始标注examples...")
            tagged_selected = asyncio.run(batch_classify(selected_examples, args.workers))

            # 合并回原来的列表
            tagged_examples = all_examples.copy()
            for i, idx in enumerate(selected_indices):
                tagged_examples[idx] = tagged_selected[i]
        else:
            # 标注全部examples
            print("\n开始标注全部examples...")
            tagged_examples = asyncio.run(batch_classify(all_examples, args.workers))

    # 标注test_samples
    print("\n开始标注test_samples...")
    tagged_tests = asyncio.run(batch_classify(test_samples, args.workers))

    # 构建输出数据，保持原始结构，增加tag字段
    output_data = data.copy()
    output_data['examples'] = tagged_examples
    output_data['test_samples'] = tagged_tests

    # 保存结果
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print(f"\n标注完成！")
    print(f"标注后的数据保存到: {args.output_path}")

    # 统计标签分布
    from collections import Counter
    if not args.only_tag_tests:
        print("\nExamples标签分布:")
        example_tags = [s.get('tag', 'untagged') for s in tagged_examples if 'tag' in s]
        tag_counts = Counter(example_tags)
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag:<20} {count:4d} ({count/len(example_tags)*100:5.1f}%)")
        untagged_count = len(tagged_examples) - len(example_tags)
        if untagged_count > 0:
            print(f"  untagged             {untagged_count:4d} ({untagged_count/len(tagged_examples)*100:5.1f}%)")

    print("\nTest Samples标签分布:")
    test_tags = [s['tag'] for s in tagged_tests]
    test_tag_counts = Counter(test_tags)
    for tag, count in sorted(test_tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:<20} {count:4d} ({count/len(test_tags)*100:5.1f}%)")

if __name__ == "__main__":
    main()
