'''
提升 2%

'''

import os
import json
import asyncio
import random
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区 =================
# MODEL_NAME = "Qwen3-4B-ascend-flagos"
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
API_KEY = "ms-c429b084-79ba-4a00-a749-aae8681e902d"
BASE_URL = "https://api-inference.modelscope.cn/v1"
CONCURRENCY_LIMIT = 1

file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}.jsonl"
OUTPUT_FILE_V1 = f"./result/{MODEL_NAME}/{file_name_without_ext}-v1.jsonl"
ERROR_OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}_errors.jsonl"

# 数据路径
DATA_FILE = "../../../data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
# DATA_FILE = "../../../flag_scale/flag-os-3/LongContext-ICL-Annotation/data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json"

DEFAULT_VALUE = "Not sad"
VALID_LABELS = ["Sad", "Not sad"]

# 重试的温度阶梯
TEMPERATURE_STEPS = [0, 0, 0, 0]
MAX_RETRIES = len(TEMPERATURE_STEPS) - 1

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
# ==========================================


# ================= Prompt =================
system_prompt = '''
# Role
You are an expert sentiment analyst specializing in social media text (tweets). Your task is to accurately detect whether the author of a given tweet is expressing sadness based on the specific annotation guidelines of the SemEval-2018 Task 1 dataset.

# Task Definition
Analyze the provided tweet and judge whether the author is "Sad" or "Not sad". 
Output ONLY the label: "Sad" or "Not sad". Do not provide any explanation.

# 可参考的经验(GRPO)


# Labeling Guidelines & Heuristics
Please follow these specific rules derived from the dataset's unique characteristics. **Note: This dataset defines "Sadness" extremely broadly and is highly sensitive to specific keywords.**

1. **Strong Lexical Triggers (CRUCIAL):**
   The presence of certain words almost always triggers a "Sad" label in this dataset, **regardless of the actual context**. If you see these themes/words, lean heavily toward "Sad":
   - **Physical expressions of sadness:** *cry, tears, pout, pouting*. (Even if crying from boiling onions, or pouting as flirting, label it "Sad").
   - **Clinical/Mental terms:** *depression, anxiety, hurting, grim, gloomy, joyless*. (Even if used as an insult like "joyless c*nt", promoting a seminar, or describing weather, label it "Sad").
   - **Explicit explicit evaluation:** *disappointing, dreadful, awful, unhappy, disgraceful*.

2. **Broad Definition of Sadness (Beyond Grief):**
   - **Daily Misfortunes & Helplessness:** Losing keys, getting bad customer service (e.g., T-Mobile/Argos complaints), lack of sleep, or experiencing physical pain (e.g., headaches). These represent distress/frustration and MUST be labeled "Sad".
   - **Melancholy, Empathy & Cynicism:** Philosophical quotes about broken dreams, feeling sorry for someone else (#poor[Name]), or calling out bullies. 
   - **Intense Frustration/Exhaustion:** If anger is mixed with helplessness, exhaustion, or venting about being scammed/losing something, it is "Sad".

3. **Text Overrides Emojis:**
   - Emojis are helpful but secondary. The text is the primary source of truth.
   - Crying emojis (😭) used with positive words ("lucky") = Not sad.
   - Laughing emojis (😂/lol) used with sad words ("unhappy", "hurting") = Sad.

4. **What is STRICTLY "Not Sad"?**
   - **Pure Disgust (Food/Objects):** Complaining purely about the bad taste of food (e.g., "taste changed #yuk") without personal distress.
   - **Vague/Ambiguous Statements:** Phrases lacking clear negative emotional words (e.g., "When the lights shut off, my main concern...").
   - **Pure Political/General Outrage WITHOUT despair words:** If it's purely arguing a point without words like "disgraceful", "dreadful", or expressing personal emotional toll.

You MUST output a valid JSON object in the following format:
{
    "reasoning": "Brief explanation",
    "result": "Sad or Not sad"
}
'''

# ==========================================


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
                    if isinstance(ground_truth, str):
                        is_correct = (prediction == ground_truth)

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
                print(f"error: {e}")
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
    test_samples = task_data.get("examples", [])
    # test_samples = task_data.get("test_samples", [])

    # 按需截取，如果您需要跑全集可以注释掉这部分逻辑
    if len(test_samples) > 500:
        test_samples = random.sample(test_samples, 100)
    else:
        print(f"警告：只有 {len(test_samples)} 条数据，不足500条，将使用全部数据")

    if not test_samples:
        print("未找到测试数据")
        return

    print(f"任务：Tweet Sadness Detection (openseek-5)")
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
            f.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n")

    with open(OUTPUT_FILE_V1, "w", encoding="utf-8") as f:
        for result in results_v1:
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✅ 结果已保存: {OUTPUT_FILE}")
    print(f"✅ V1格式结果已保存: {OUTPUT_FILE_V1}")

    # ========= Bad Case =========
    errors = [r for r in results if not r["is_correct"]]

    if errors:
        with open(ERROR_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False, indent=2) + "\n")

        print(f"❌ 错误样本已保存: {ERROR_OUTPUT_FILE} ({len(errors)}条)")
    else:
        print("🎉 无错误样本")


if __name__ == "__main__":
    asyncio.run(main())