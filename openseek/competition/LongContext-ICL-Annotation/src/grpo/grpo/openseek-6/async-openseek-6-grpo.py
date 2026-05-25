'''
基于 openseek-5-base 的架构重构 openseek-6
任务：MNLI Same Genre Classification
'''

import os
import json
import asyncio
import random
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区 =================
# MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"  # 替换为你实际使用的模型名
MODEL_NAME = "Qwen3-4B-ascend-flagos"
CONCURRENCY_LIMIT = 16

file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}.jsonl"
OUTPUT_FILE_V1 = f"./result/{MODEL_NAME}/{file_name_without_ext}-v1.jsonl"
ERROR_OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}_errors.jsonl"

# 数据路径
# DATA_FILE = "../../../data/openseek-6_mnli_same_genre_classification.json"
DATA_FILE = "../../../flag_scale/data/openseek-6_mnli_same_genre_classification.json"

DEFAULT_VALUE = "N"
VALID_LABELS = ["Y", "N"]

# 重试的温度阶梯
# TEMPERATURE_STEPS = [0.0, 0.7, 1.0, 1.2]
TEMPERATURE_STEPS = [1.0, 1.1, 1., 1.2]
MAX_RETRIES = len(TEMPERATURE_STEPS) - 1
# ==========================================


# ================= Prompt =================
# 任务：MNLI 同类型判断（Same Genre Classification）
system_prompt = '''
You are a data processing assistant specializing in text classification. 
In this task, you're given two sentences, sentence 1 and sentence 2, and the genre they belong to.
Your job is to determine if the two sentences belong to the same genre or not.
Indicate your answer with Y and N respectively.

Genres available include: face-to-face, government, letters, 9/11, slate, telephone, travel, verbatim, oup, fiction.

You MUST output a valid JSON object in the following format:
{
    "reasoning": "Brief explanation",
    "result": "Y or N"
}

Rules:
- `result` must be strictly "Y" or "N".
- DO NOT output any extra explanations outside the JSON.
- Keep your thinking process (<think>) under 500 words.

# 参考经验(GRPO获取)
{
    "1": "政府类需明确法规或政策细节才能归类",
    "2": "需验证句子是否明确源自公共政府网站或官方政策文件",
    "3": "体育赛事数据隐含旅游场景需补充具体地点信息",
    "4": "政府类必须明确提及机构名称和具体法规细节，否则不适用",
    "5": "中性场景需结合上下文判断是否包含虚构元素",
    "6": "旅游类需明确景点名称或行程信息才能归类",
    "7": "政府类需明确提及机构名称和具体法规细节",
    "8": "对话中包含情感表达和角色名称则可能属于fiction",
    "9": "未明确来源的医疗政策陈述不适用政府类别",
    "10": "体育赛事数据需额外验证是否与旅游场景直接关联",
    "11": "中性时间描述需结合上下文确认是否为小说情节",
    "12": "文学人物生平描述若无旅行元素不构成travel类",
    "13": "真实历史事件需额外验证是否为虚构场景描述",
    "14": "头衔/地点描述需结合上下文判断是否为虚构",
    "15": "slate需明确包含文化评论或杂志专有语境",
    "16": "政府行动若无法规细节需结合文化语境判断类别",
    "17": "需验证对话是否包含具体人物或虚构场景元素",
    "18": "抽象动作描述无情感或角色要素时默认非fiction",
    "19": "电话类需明确对话结构特征才能归类",
    "20": "需确认动作描述是否在虚构语境中出现（如小说场景）",
    "21": "若句子涉及教育/就业与旅游场景的潜在关联，需结合上下文判断是否属于travel",
    "22": "文学人物评论需确认是否为独立文化分析",
    "23": "政治话题需明确文化分析背景或媒体评论视角才能归为slate",
    "24": "角色名称与地点描述的组合需结合上下文判断是否为虚构场景",
    "25": "宗教/文化场景需结合虚构角色或地点描述才能归为fiction",
    "26": "政府类需明确法规或政策细节，否则仅提及项目名称不适用",
    "27": "中性陈述需结合上下文确认是否为虚构场景",
    "28": "医疗政策陈述需明确来源才能归为政府类别",
    "29": "未明确地点的体育统计数据需结合上下文判断是否属于travel",
    "30": "角色名称与抽象场景结合需确认文化语境是否虚构",
    "31": "宗教建筑描述若无旅游场景说明不构成travel类别",
    "32": "角色名称与地点描述需结合上下文验证是否属于虚构场景",
    "33": "对话中重复语气词和非正式表达可辅助判断电话类",
    "34": "现实中的游戏体验不构成fiction",
    "35": "博物馆展览若与旅游相关可视为文化旅游场景",
    "36": "政府类需明确法规或政策细节，否则仅提及项目名称不适用",
    "37": "未明确旅游景点的住宿建议不构成travel类",
    "38": "政府类包含政策讨论，即使未提及具体法规也可基于历史事件归类",
    "39": "真实人物的抽象场景描述可能属于fiction需进一步验证",
    "40": "角色名称若与真实地点或机构相关，需结合上下文确认是否为虚构",
    "41": "抽象概念描述需结合文化语境判断虚构性",
    "42": "文化习俗描述若无具体旅游地点或行程信息则不适用travel类",
    "43": "中性陈述需结合上下文确认是否为虚构场景",
    "44": "travel类必须明确景点/行程信息才能成立",
    "45": "历史事件描述需验证是否为虚构创作",
    "46": "真实历史人物/地点的描述需额外验证是否为虚构场景",
    "47": "抽象情感描述需结合具体场景或角色互动判断虚构性",
    "48": "政治话题需明确文化分析背景或媒体评论视角才能归为slate",
    "49": "抽象动作描述需确认是否在虚构语境中出现",
    "50": "抽象描述需结合文化背景或叙事元素确认虚构性",
    "51": "抽象场景描述无具体对话动词不适用电话类",
    "52": "讨论教育/就业与旅游场景的潜在关联仍需明确景点信息",
    "53": "文化术语需结合具体语境确认是否为虚构性文学描写",
    "54": "政治话题需结合文化分析背景或媒体评论视角才适用slate",
    "55": "若描述隐含旅游场景需结合上下文判断是否为虚构创作",
    "56": "抽象场景描述需明确包含虚构性叙事特征"
}

# Example 1
input: "Sentence 1: However, if it is obtained in order to engage the patient in treatment, the information is protected under the above federal regulations that require the express, written permission of the patient before it can be shared with others. Sentence 2: Ecology of Fear, according to the columnists, contained errors but Los City of Quartz didn't. Genre: slate."
output JSON:
{"reasoning": "Sentence 1 讨论的是医疗隐私和联邦法规，用词正式。Sentence 2 则是对两本书的评论。两句话主题和语境相差甚远，风格不同，不属于同一 genre（体裁）。因此判断为 N。", "result": "N"}

# Example 2
input: "Sentence 1: were you have you i take it you haven't spent any time in the military Sentence 2: Jon said there is nothing else we can do. Genre: telephone."
output JSON:
{"reasoning": "Sentence 1 包含口语化的重复和停顿（were you have you），具有典型的电话录音或非正式口语交流的特征。而 Sentence 2 是一个简单的间接引语转述，缺乏相同体裁的明显标志，两者不属于同一个典型的体裁。因此判断为 N。", "result": "N"}

# Example 3
input: "Sentence 1: I do not have the energy to remedy these deficiencies now. Sentence 2: I don't have the strength to fix these problems now. Genre: letters."
output JSON:
{"reasoning": "这两句话结构和含义几乎完全相同，都是以第一人称描述自己无力解决当前的缺陷/问题。它们的表达风格高度一致，都符合个人书信（letters）或其他主观表达文体的特征。因此判断为 Y。", "result": "Y"}

# Example 4
input: "Sentence 1: To the west of Naoussa is Kolymbithres, a growing resort whose beaches are surrounded by strange rock features that are folded and sculpted by the wind. Sentence 2: Paros is a Greek island in the Aegean Sea. Genre: travel."
output JSON:
{"reasoning": "Sentence 1 描述了某个度假胜地的地理位置和自然风光，Sentence 2 描述了希腊岛屿的客观地理信息。两者都使用了典型的旅游指南（travel）风格的客观描述性语言，属于同一体裁。因此判断为 Y。", "result": "Y"}

# Example 5
input: "Sentence 1: The government released new healthcare guidelines today. Sentence 2: She ran quickly through the dark forest, heart pounding. Genre: government."
output JSON:
{"reasoning": "Sentence 1 是关于政府发布医疗保健指南的新闻式或公文式陈述（government）。Sentence 2 是描述人物动作和内心状态的叙事小说（fiction）风格。两者类型明显不同。因此判断为 N。", "result": "N"}

# Example 6
input: "Sentence 1: The attacks of September 11 changed American foreign policy forever. Sentence 2: On that morning, thousands lost their lives in lower Manhattan. Genre: 9/11."
output JSON:
{"reasoning": "Sentence 1 讨论 9/11 袭击对美国外交政策的影响。Sentence 2 描述了曼哈顿下城在那个早晨发生的伤亡。两者都明确围绕 9/11 事件展开，且语境与主题高度一致，属于相同的分析记述体裁（9/11）。因此判断为 Y。", "result": "Y"}
'''

system_prompt = '''
You are a data processing assistant specializing in text classification. 
In this task, you're given two sentences, sentence 1 and sentence 2, and the genre they belong to.
Your job is to determine if the two sentences belong to the same genre or not.
Indicate your answer with Y and N respectively.

Genres available include: face-to-face, government, letters, 9/11, slate, telephone, travel, verbatim, oup, fiction.

You MUST output a valid JSON object in the following format:
{
    "reasoning": "Brief explanation",
    "result": "Y or N"
}

Rules:
- `result` must be strictly "Y" or "N".
- DO NOT output any extra explanations outside the JSON.
- Keep your thinking process (<think>) under 500 words.

### CRITICAL EVALUATION GUIDELINES ###
1. **Semantic Connection (Y):** In this dataset, Sentence 2 is often a direct paraphrase, summary, entailment, or contradiction of Sentence 1. If S2 shares the exact same core topic, specific entities, or meaning as S1, they belong to the same pair. Output **Y**. Do NOT output N just because S2 is written in simpler, standard English and lacks the stylistic flair (e.g., stutters, formatting) of S1.
2. **Topic Disconnect (N):** If S1 and S2 discuss completely unrelated topics, people, or events (e.g., S1 is about a politician, S2 is about losing a game; S1 is about 'Mrs. Vandemeyer', S2 is about 'American Express'), they are randomly paired. Output **N**, even if their broad writing styles seem vaguely similar.
3. **Stylistic Clash (N):** Pay close attention to extreme formatting markers. 'telephone' or 'verbatim' often lack capitalization/punctuation and use fillers ("uh", "um"). 'fiction' relies on dialogue tags and character names. If S1 is an uncapitalized speech transcript and S2 is a perfectly structured technical document with no topic overlap, output **N**.

# Example 1
input: "Sentence 1: However, if it is obtained in order to engage the patient in treatment, the information is protected under the above federal regulations that require the express, written permission of the patient before it can be shared with others. Sentence 2: Ecology of Fear, according to the columnists, contained errors but Los City of Quartz didn't. Genre: slate."
output JSON:
{"reasoning": "Sentence 1 讨论的是医疗隐私和联邦法规。Sentence 2 则是对两本书的评论。两句话话题完全断裂 (Topic Disconnect)，不属于同一组。因此判断为 N。", "result": "N"}

# Example 2
input: "Sentence 1: oh this is kind of tough i don't too often watch you know shows that are on on a regular basis Sentence 2: I hardly ever watch shows that are on regularly. Genre: telephone."
output JSON:
{"reasoning": "Sentence 1 是充满停顿和口语词的电话录音（telephone）。Sentence 2 Although 缺乏口语特征，但它是 Sentence 1 的完美语义改写（Paraphrase/Summary）。两者讨论完全相同的具体含义，存在直接的 Semantic Connection。因此判断为 Y。", "result": "Y"}

# Example 3
input: "Sentence 1: I've brought you some tea. Mrs. Vandemeyer did not reply. Sentence 2: American Express is rejected everywhere, so I get why people complain so much about it happening. Genre: telephone."
output JSON:
{"reasoning": "Sentence 1 包含人名和动作描写，具有小说特征；Sentence 2 讨论信用卡。两者话题完全风马牛不相及，且风格存在冲突。因此判断为 N。", "result": "N"}

# Example 4
input: "Sentence 1: At the national level, there are a total of 575 wrestlers classed in six divisions according to their win-loss ratio in the annual tournaments. Sentence 2: There are about 60 wrestlers at the national level. Genre: travel."
output JSON:
{"reasoning": "Sentence 1 讨论摔跤手数量，Sentence 2 虽然给出了不同的数字，但针对的是完全相同的具体话题（national level wrestlers），构成了语义上的矛盾（Contradiction）。因为它们有极强的内在逻辑联系，属于同一组数据集。因此判断为 Y。", "result": "Y"}
'''
# ==========================================


client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:9010/v1",
    # api_key="ms-c429b084-79ba-4a00-a749-aae8681e902d",
    # base_url="https://api-inference.modelscope.cn/v1",
)


def generate_messages(current_input):
    system_prompt_new = system_prompt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = current_input.encode("utf-8", "ignore").decode("utf-8", "ignore")

    return [
        {"role": "system", "content": system_prompt_new},
        {"role": "user", "content": f"input: {text}"}
    ]


# ================= JSON 解析器 =================
def extract_json_from_response(text: str) -> dict:
    # 去掉 <think>
    if "</think>" in text:
        text = text.split("</think>")[-1]

    text = text.strip()

    # 提取 markdown code block
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```python" in text:
        text = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())
# ============================================


async def process_single_sample(sample, semaphore):
    async with semaphore:
        messages = generate_messages(sample["input"])
        ground_truth = sample.get("output", DEFAULT_VALUE)

        for attempt, temp in enumerate(TEMPERATURE_STEPS):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temp,
                    max_tokens=8192,
                )

                raw_content = response.choices[0].message.content
                parsed = extract_json_from_response(raw_content)

                if "result" in parsed and parsed["result"] in VALID_LABELS:
                    prediction = parsed["result"]

                    # 验证正确性
                    if isinstance(ground_truth, list):
                        is_correct = (prediction == ground_truth[0])
                    elif isinstance(ground_truth, str):
                        is_correct = (prediction == ground_truth)
                    else:
                        is_correct = False

                    return {
                        "test_sample_id": sample["id"],
                        "input": sample["input"],
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                        "is_correct": is_correct,
                        "reasoning": parsed.get("reasoning", ""),
                        "retries_used": attempt
                    }
                else:
                    raise ValueError(f"JSON 缺失 result 字段或值 '{parsed.get('result')}' 非法")

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_RETRIES:
                    continue
                else:
                    tqdm.write(f"样本 {sample['id']} 解析失败: {e} (已重试 {MAX_RETRIES} 次)")

            except Exception as e:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    tqdm.write(f"样本 {sample['id']} 调用失败: {e}")

        # 全部失败 fallback
        return {
            "test_sample_id": sample["id"],
            "input": sample["input"],
            "prediction": DEFAULT_VALUE,
            "ground_truth": ground_truth,
            "is_correct": (DEFAULT_VALUE == ground_truth),
            "reasoning": "Failed due to parsing or API errors",
            "retries_used": MAX_RETRIES
        }


async def main():
    random.seed(42)

    # ========= 读取数据 =========
    if not os.path.exists(DATA_FILE):
        print(f"数据文件不存在: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    # 兼容两种常见的数据格式 key
    # test_samples = task_data.get("examples", [])   # 包含真实标签
    test_samples = task_data.get("test_samples", [])   # 不包含真实标签

    # 测试用截取逻辑，如需跑全集请注释此代码段
    if len(test_samples) > 500:
        test_samples = random.sample(test_samples, 499) # 仅供快速测试使用
    else:
        print(f"警告：只有 {len(test_samples)} 条数据，不足500条，将使用全部数据")

    if not test_samples:
        print("未找到测试数据")
        return

    print(f"任务：MNLI Same Genre Classification (openseek-6)")
    print(f"样本数: {len(test_samples)} | 并发: {CONCURRENCY_LIMIT}")
    print(f"模型: {MODEL_NAME}")
    print(f"重试温度策略: {TEMPERATURE_STEPS}\n")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    tasks = [
        process_single_sample(sample, semaphore)
        for sample in test_samples
    ]

    results = await tqdm.gather(*tasks, desc="推理进度")

    # ========= 兼容 V1 格式 (只保留 ID 和 Prediction) =========
    results_v1 = [
        {
            "test_sample_id": result["test_sample_id"],
            "prediction": result["prediction"]
        }
        for result in results
    ]

    # ========= 评估 =========
    correct = sum(r["is_correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 40)
    print("🎯 推理评估报告")
    print("=" * 40)
    print(f"总样本数: {total}")
    print(f"正确数:   {correct}")
    print(f"错误数:   {total - correct}")
    print(f"准确率:   {accuracy:.2%}")
    print("=" * 40 + "\n")

    # ========= 保存结果 =========
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUTPUT_FILE_V1, "w", encoding="utf-8") as f:
        for result in results_v1:
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✅ 详细结果已保存: {OUTPUT_FILE}")
    print(f"✅ V1格式结果已保存: {OUTPUT_FILE_V1}")

    # ========= Bad Case =========
    errors = [r for r in results if not r["is_correct"]]

    if errors:
        with open(ERROR_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False, indent=2) + "\n")

        print(f"❌ 错误样本已单独保存分析: {ERROR_OUTPUT_FILE} ({len(errors)}条)")
    else:
        print("🎉 全部预测正确，无错误样本")


if __name__ == "__main__":
    asyncio.run(main())