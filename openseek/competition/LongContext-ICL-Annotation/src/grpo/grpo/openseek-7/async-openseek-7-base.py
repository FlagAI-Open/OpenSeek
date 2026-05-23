import os
import json
import asyncio
import random
import re
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区 =================
MODEL_NAME = "Qwen3-4B-ascend-flagos"
CONCURRENCY_LIMIT = 16
FEW_SHOT_K = 3  # 动态检索的 Few-shot 样本数量

file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}.jsonl"
OUTPUT_FILE_V1 = f"./result/{MODEL_NAME}/{file_name_without_ext}-v1.jsonl"
ERROR_OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}_errors.jsonl"

DATA_FILE = "../../../flag_scale/flag-os-3/LongContext-ICL-Annotation/data/openseek-7_jeopardy_answer_generation_all.json"
# DATA_FILE = "../../../data/openseek-7_jeopardy_answer_generation_all.json"
DEFAULT_VALUE = "unknown"

# TEMPERATURE_STEPS = [0.0, 0.7, 1.0, 1.2]
TEMPERATURE_STEPS = [1, 1, 1, 1.2]
MAX_RETRIES = len(TEMPERATURE_STEPS) - 1
# ==========================================

# ================= Prompt =================
system_prompt = '''
You are a highly accurate trivia assistant specialized in Jeopardy-style questions.

You will be given:
- A category
- A clue

Your task:
Identify the single best answer that fits BOTH the clue AND the category.

---------------------
STRICT REQUIREMENTS
---------------------
1. The answer MUST:
   - Be factually correct
   - Match the category context precisely
   - Be the most specific valid answer (avoid overly broad answers)

2. Output format:
   - Return ONLY a valid JSON object
   - NO extra text before or after JSON

3. JSON schema:
{
    "reasoning": "brief explanation",
    "result": "answer in lowercase"
}

4. Answer formatting rules:
   - ALL LOWERCASE
   - NO articles ("a", "an", "the") unless required
   - NO punctuation unless part of official name
   - Prefer canonical names (e.g., "albert einstein", not "einstein")

5. Reasoning rules:
   - Keep it SHORT (1 sentence)
   - Focus on key clue → answer mapping
   - Do NOT include uncertainty or speculation

---------------------
FAILURE AVOIDANCE
---------------------
- Do NOT ignore the category
- Do NOT output explanations outside JSON
- Do NOT output multiple answers
- Do NOT include "what is" / "who is" in result
'''
# ==========================================

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:9010/v1",
)


# ================= 技巧 1 & 2: 动态 Few-Shot 检索与类别对齐 =================

def parse_input(text):
    """提取 Category 和 Clue 以进行精细化匹配"""
    cat_match = re.search(r'Category:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    clue_match = re.search(r'Clue:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)

    cat = cat_match.group(1).strip() if cat_match else ""
    clue = clue_match.group(1).strip() if clue_match else text
    return cat, clue


def tokenize(text):
    """简单的分词器，用于 N-gram/关键词 Overlap"""
    return set(re.findall(r'\b\w+\b', text.lower()))

# # 常见英文无意义连接词 / 停用词
# STOP_WORDS = {
#     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of',
#     'for', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
#     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'my',
#     'your', 'this', 'that', 'these', 'those'
# }
#
# def tokenize(text):
#     """纯手写分词器：提取单词 + 过滤无意义连接词"""
#     words = re.findall(r'\b\w+\b', text.lower())
#     filtered = [w for w in words if w not in STOP_WORDS]
#     return set(filtered)


def retrieve_dynamic_few_shots(target_sample, pool, k=FEW_SHOT_K):
    """
    SimICL 检索算法：从 pool (即 examples) 中检索最相似的数据
    """
    target_cat, target_clue = parse_input(target_sample["input"])
    target_tokens = tokenize(target_clue)

    scored_pool = []
    for ex in pool:
        # 排除自身（以防万一 id 重复），以及没有标准答案的样本
        if ex.get("id") == target_sample.get("id") or not ex.get("output"):
            continue

        ex_cat, ex_clue = parse_input(ex["input"])
        ex_tokens = tokenize(ex_clue)

        score = 0
        # 技巧 2: Category 对齐优先
        if target_cat and ex_cat:
            if target_cat.lower() == ex_cat.lower():
                score += 100  # 同类别极大加分
            else:
                cat_overlap = len(tokenize(target_cat) & tokenize(ex_cat))
                score += cat_overlap * 20  # 类别部分重叠加分

        # 技巧 1: 语义相似度 (n-gram / keyword overlap)
        word_overlap = len(target_tokens & ex_tokens)
        score += word_overlap

        scored_pool.append((score, ex))

    # 按分数降序排序，取前 K 个
    scored_pool.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_pool[:k]]


# ================= 技巧 3: 答案风格对齐 =================

def format_few_shot_assistant(example):
    """
    强制 Few-shot 的输出风格与系统 Prompt 完全对齐
    """
    gt = example.get("output", "")
    if isinstance(gt, list): gt = gt[0]
    gt = str(gt).strip().lower()

    cat, _ = parse_input(example["input"])

    reasoning = f"The answer matches the clue and fits the '{cat}' category." if cat else "The answer matches the given clue."

    aligned_output = {
        "reasoning": reasoning,
        "result": gt
    }
    return json.dumps(aligned_output, ensure_ascii=False)


def generate_messages(current_input, few_shots):
    system_p = system_prompt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    messages = [{"role": "system", "content": system_p}]

    for fs in few_shots:
        fs_input = fs["input"].encode("utf-8", "ignore").decode("utf-8", "ignore")
        fs_output = format_few_shot_assistant(fs)
        messages.append({"role": "user", "content": f"input:\n{fs_input}"})
        messages.append({"role": "assistant", "content": fs_output})

    text = current_input.encode("utf-8", "ignore").decode("utf-8", "ignore")
    messages.append({"role": "user", "content": f"input:\n{text}"})

    return messages


# ================= JSON 解析器 =================
def extract_json_from_response(text: str) -> dict:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


# ============================================

async def process_single_sample(sample, few_shots, semaphore):
    async with semaphore:
        messages = generate_messages(sample["input"], few_shots)
        # test_samples 可能没有 output 字段，如果没有则 ground_truth 为 None
        ground_truth = sample.get("output", None)
        if isinstance(ground_truth, list): ground_truth = ground_truth[0]

        for attempt, temp in enumerate(TEMPERATURE_STEPS):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temp,
                    max_tokens=2048,
                )

                raw_content = response.choices[0].message.content
                parsed = extract_json_from_response(raw_content)

                if "result" in parsed:
                    prediction = str(parsed["result"]).strip().lower()

                    is_correct = None
                    if ground_truth is not None:
                        gt_val = ground_truth.strip().lower() if isinstance(ground_truth, str) else str(
                            ground_truth).lower()
                        is_correct = (prediction == gt_val)

                    return {
                        "test_sample_id": sample.get("id", "unknown"),
                        "input": sample["input"],
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                        "is_correct": is_correct,
                        "reasoning": parsed.get("reasoning", ""),
                        "retries_used": attempt
                    }
                else:
                    raise ValueError(f"JSON 缺失 result 字段")

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_RETRIES:
                    continue
                else:
                    tqdm.write(f"样本 {sample.get('id')} 解析失败: {e}")
            except Exception as e:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                else:
                    tqdm.write(f"样本 {sample.get('id')} 调用失败: {e}")

        return {
            "test_sample_id": sample.get("id", "unknown"),
            "input": sample["input"],
            "prediction": DEFAULT_VALUE,
            "ground_truth": ground_truth,
            "is_correct": False if ground_truth else None,
            "reasoning": "Failed to parse or API error",
            "retries_used": MAX_RETRIES
        }


async def main():
    random.seed(42)

    if not os.path.exists(DATA_FILE):
        print(f"数据文件不存在: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    # 1. 明确划分：检索池 (examples) 和 推理目标 (test_samples)
    knowledge_pool = task_data.get("examples", [])
    test_samples = task_data.get("test_samples", [])
    # test_samples = task_data.get("examples", [])

    if not test_samples:
        print("未找到 test_samples 测试数据")
        return
    if not knowledge_pool:
        print("警告：未找到 examples 检索数据，将使用 Zero-Shot！")

    # 如果测试集太大，支持随机截取测试（按需修改数字）
    if len(test_samples) > 500:
        target_samples = random.sample(test_samples, 100)
    else:
        target_samples = test_samples

    print(f"任务：Jeopardy Answer Generation (Dynamic Few-Shot ICL)")
    print(f"检索池大小 (examples): {len(knowledge_pool)}")
    print(f"当前推理样本数 (test_samples): {len(target_samples)}")
    print(f"模型: {MODEL_NAME} | 并发: {CONCURRENCY_LIMIT}")

    # 2. 从 knowledge_pool(即 examples) 中提取 Few-shot 样本
    print("正在为每个样本计算动态 Few-shot (Category-aware & Semantic)...")
    tasks_with_few_shots = []
    for sample in tqdm(target_samples, desc="Pre-computing Few-Shots"):
        # 修改点：将 pool 参数指定为 knowledge_pool
        best_few_shots = retrieve_dynamic_few_shots(sample, knowledge_pool, k=FEW_SHOT_K)
        tasks_with_few_shots.append((sample, best_few_shots))

    print("开始并发推理...\n")
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    tasks = [process_single_sample(sample, fs, semaphore) for sample, fs in tasks_with_few_shots]
    results = await tqdm.gather(*tasks, desc="Inference Progress")

    results_v1 = [{"test_sample_id": r["test_sample_id"], "prediction": r["prediction"]} for r in results]

    # 3. 统计结果时，过滤掉没有 ground_truth 的 test_samples（针对盲测集）
    valid_evals = [r for r in results if r["is_correct"] is not None]
    if valid_evals:
        correct = sum(1 for r in valid_evals if r["is_correct"])
        total = len(valid_evals)
        accuracy = correct / total
        print(f"\n🎯 本地评估 Accuracy: {accuracy:.2%} ({correct}/{total})")
    else:
        print(f"\n🎯 提示：因为 test_samples 中不包含 output 字段，跳过本地准确率统计。结果已保存用于提交验证。")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUTPUT_FILE_V1, "w", encoding="utf-8") as f:
        for r_v1 in results_v1:
            f.write(json.dumps(r_v1, ensure_ascii=False) + "\n")

    # 记录 Bad Case (如果有真实答案对照的话)
    errors = [r for r in results if r.get("is_correct") is False]
    if errors:
        with open(ERROR_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False, indent=2) + "\n")

    print(f"✅ 详细结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())