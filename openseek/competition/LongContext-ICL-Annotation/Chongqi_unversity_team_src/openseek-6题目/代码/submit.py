import json
import os
import time
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def task_data_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("task_id"), data.get("examples", []), data.get("test_samples", [])

client = OpenAI(
    api_key="dummy",
    base_url="https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1"
)

def qwen_api(messages, model="/Qwen3-4B/Qwen/Qwen3-4B", retries=3):
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return res.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                print(f"\nAPI 调用失败: {e}")
                return ""
            time.sleep(2)

system_prompt = """You are an expert data annotator.
Return exactly one label: Y or N.
Do not provide any explanation."""

user_prompt_template = """Task:
You are given Sentence 1, Sentence 2, and a target Genre.
Decide whether to output Y or N.

Label meaning:
- Y: The pair is derived from the same context/genre.
- N: The pair is disconnected and from different styles.

CRITICAL SHORTCUTS (EVALUATE THIS FIRST):
Do NOT independently evaluate the genre of Sentence 2 if it is clearly derived from Sentence 1. If ANY of the following conditions are met, you MUST output Y immediately:
1. EXACT ENTITY / NOUN OVERLAP: Both sentences mention the exact same specific nouns, proper names, or subjects (e.g., "New Sacristy", "Bhimsen Temple", "merchants", "shopkeepers"). Even if Sentence 2 contradicts Sentence 1 by saying the entity doesn't exist, output Y.
2. SYNTACTIC OVERLAP / NEGATION: Sentence 2 heavily borrows the phrasing or sentence structure of Sentence 1, often just adding or removing a negative word (e.g., S1: "this Statement requires...", S2: "This statement doesn't require..."). Output Y.
3. SEMANTIC PARAPHRASE: Sentence 2 provides specific examples of concepts mentioned in Sentence 1 (e.g., S1 mentions "politics and business sector", S2 mentions "governor of state or manager at a company"). Output Y.

When to output N:
Output N ONLY IF there is NO specific entity overlap AND the sentences belong to clearly different domains. 
- Example: S1 is about general "fiscal policy", but S2 suddenly talks about a specific historical event "Nixon and Ho Chi Minh" (Only a loose 'politics' topic, but no shared entities and different narrative style) -> Output N.
- Example: S1 is casual phone talk, S2 is a formal textbook sentence. -> Output N.

Genre hints:
- face-to-face: casual in-person dialogue
- government: formal public-information or policy language
- letters: fundraising or donor-oriented letter style
- 9/11: specifically about the 9/11 attacks
- slate: cultural/social commentary or magazine-style opinion/exposition
- telephone: spoken, conversational, disfluent, turn-taking phone dialogue
- travel: guidebook-like travel information
- verbatim: short linguistics-related posts
- oup: nonfiction educational/expository prose
- fiction: narrative or literary prose

Examples:
Example 1
Sentence 1: Therefore, this Statement requires that information on these resources be reported to highlight their long-term-benefit nature.
Sentence 2: This statement doesn't require that any information on these resources be collected.
Genre: government
Output: Y
(Reason: Syntactic overlap and direct negation. They discuss the exact same statement.)

Example 2
Sentence 1: Beyond is the Bhimsen Temple, a pagoda dedicated to the patron god of merchants (dear to Newari shopkeepers).
Sentence 2: The Bhimsen Temple was never built because the president hated merchants and shopkeepers.
Genre: travel
Output: Y
(Reason: Exact entity overlap of "Bhimsen Temple", "merchants", "shopkeepers". Contradictions are allowed.)

Example 3
Sentence 1: and you know occupying a very prominent role with the politics and in the business sector
Sentence 2: Being the governor of state or being a manager at a company.
Genre: telephone
Output: Y
(Reason: Semantic paraphrase. Governor/manager are specific examples of politics/business.)

Example 4
Sentence 1: Such simulations can help policymakers assess the long-term consequences of fiscal policy and saving choices made today.
Sentence 2: Nixon's decision to force Ho Chi Minh to withdraw saved 18,000 American lives.
Genre: government
Output: N
(Reason: No specific entity overlap. S2 is historical narrative, completely disconnected from the formal fiscal policy in S1.)

Example 5
Sentence 1: now he he is a good uh actually i did i played flute for almost ten years and and uh so i i i i appreciate his too his his music he he he's from Ireland isn't he
Sentence 2: Thank you too, goodbye.
Genre: telephone
Output: N
(Reason: Disconnected. S2 is a generic sign-off unrelated to the specific entities/topic of S1.)

Now solve this instance:

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Genre: {genre}

Return only Y or N.
"""

def extract_prediction(raw_output: str) -> str:
    if not raw_output:
        return "N"

    text = raw_output.strip().upper()

    if text == "Y":
        return "Y"
    if text == "N":
        return "N"

    match = re.search(r"\b([YN])\b", text)
    if match:
        return match.group(1)

    return "N"

def parse_input_text(input_text: str):
    pattern = r"Sentence 1:\s*(.*?)\s*Sentence 2:\s*(.*?)\s*Genre:\s*(.*)"
    match = re.search(pattern, input_text.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"输入格式无法匹配：{input_text}")

    sentence_1 = match.group(1).strip()
    sentence_2 = match.group(2).strip()
    genre = match.group(3).strip()

    return sentence_1, sentence_2, genre

def process_single_sample(sample, task_id):
    test_sample_id = sample.get("id", "")
    input_text = sample.get("input", "").strip()

    sentence_1, sentence_2, genre = parse_input_text(input_text)

    user_prompt = user_prompt_template.format(
        sentence_1=sentence_1,
        sentence_2=sentence_2,
        genre=genre
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_output = qwen_api(messages)
    prediction = extract_prediction(raw_output)

    return {
        "test_sample_id": test_sample_id,
        "prediction": prediction
    }

if __name__ == "__main__":
    file_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\data\openseek-6_mnli_same_genre_classification.json"
    out_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\rgs_q6\experiment/openseek-6-v1.jsonl"

    task_id, examples, test_samples = task_data_loader(file_path)

    eval_data = test_samples
    total_cnt = len(eval_data)
    process_func = partial(process_single_sample, task_id=task_id)

    max_workers = 50
    results_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_func, eval_data), total=total_cnt, desc="Predicting"):
            results_list.append(result)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in results_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n预测完成！总数：{total_cnt}")
    print(f"结果已成功保存至：{out_path}")