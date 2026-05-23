import json
import os
import time
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Any, Tuple

# 你的 few shot 样例保存的 txt 路径（改成你自己的）
FEW_SHOT_PATH = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\rgs_q7\experiment\filtered_routes_output.txt"

with open(FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    few_shot_examples = f.read().strip()  # 读取全部样例

# 先生成两阶段路由
ROUTE_PROMPT = f"""
# 任务说明
你需要对 Jeopardy trivia 题目进行双层级路由分类：一级路由route、二级子路由subroute。
【强约束】本题库95%题目为客观知识题，**优先判定为 route: factual**，非必要绝不使用 wordplay / spelling / multi / completion。
严格遵循分类规则，禁止冗余输出，**只输出纯净JSON**，无任何解释、无多余文字。
输出固定格式：
{{"route": "xxx", "subroute": "xxx"}}

# 路由分类规则
## 1. route = factual（默认首选）
Treat this as a factual knowledge clue.
- factual_person：人物识别
- factual_place：地点、国家、城市、地理相关
- factual_title：书籍、电影、歌曲、专辑、文艺作品
- factual_number：数字、年份、数量、数值
- factual_definition：名词、概念、常识、专业术语解释【最高频】
- factual_role_identity：职业、身份、称号
- factual_bridge：多线索推理
- factual_list_or_set_selection：集合选择

## 2. route = wordplay
仅谐音、押韵、变位词、文字游戏才使用，本题极少出现

## 3. route = completion
仅补全谚语、名言、歌词才使用

## 4. route = spelling
仅首字母、尾字母、单词长度限制才使用

## 5. route = multi
仅多选项、多物品集合题才使用

# 参考样例（Few-Shot）
{few_shot_examples}

强制要求：
1. 非文字游戏/补全/拼写题，一律 route=factual
2. 输出必须是严格单行JSON，无换行、无注释、无额外文字
3. 禁止输出思考过程，只返回{{"route":"","subroute":""}}
"""

PROBLEM_SOLVE_PROMPT = """You are an expert Jeopardy-style trivia solver.
You will be given:
- category
- clue
- forced route
- forced subroute
- instruction

Rules:
1. You MUST solve the clue under the forced route/subroute perspective.
2. Return the shortest canonical Jeopardy-style answer.
3. Prefer canonical short forms over expanded descriptions.
4. If uncertain, still provide the single most likely answer under this route/subroute.
5. Do not explain outside JSON.

Return ONLY valid JSON:
{
"reasoning_brief": "...",
"candidates": ["...", "...", "..."],
"final_answer": "..."
}

---

Now, answer the following clue:

You will be given:
- category
- clue
- forced route
- forced subroute
- instruction

Rules:
1. You MUST solve the clue under the forced route/subroute perspective.
2. Return the shortest canonical Jeopardy-style answer.
3. Prefer canonical short forms over expanded descriptions.
4. If uncertain, still provide the single most likely answer under this route/subroute.
5. Do not explain outside JSON.

Return ONLY valid JSON:
{
"reasoning_brief": "...",
"candidates": ["...", "...", "..."],
"final_answer": "..."
}
"""

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

def build_route_instruction(route: str, subroute: str) -> str:
    base_rules = [
        f"Forced route is {route}.",
        f"Forced subroute is {subroute}.",
        "Solve the clue strictly from this forced route/subroute perspective.",
        "Return one concise canonical Jeopardy-style answer.",
        "Generate 3 to 5 candidates internally if helpful, then output the single best one.",
        "Do not output explanations outside JSON."
    ]

    if route == "factual":
        base_rules += ["Treat this as a factual knowledge clue."]
        if subroute == "factual_person": base_rules += ["Bias toward identifying a person."]
        elif subroute == "factual_place": base_rules += ["Bias toward identifying a place."]
        elif subroute == "factual_title": base_rules += ["Bias toward identifying a titled work."]
        elif subroute == "factual_number": base_rules += ["Bias toward identifying a number or quantity."]
        elif subroute == "factual_definition": base_rules += ["Bias toward identifying a term or definition label."]
        elif subroute == "factual_role_identity": base_rules += ["Bias toward relationship or role-based identification."]
        elif subroute == "factual_bridge": base_rules += ["Bias toward multi-hop factual reasoning across anchors."]
        elif subroute == "factual_list_or_set_selection": base_rules += ["Bias toward factual selection from a set."]
        else: base_rules += ["Use straightforward factual reasoning."]
    elif route == "wordplay":
        base_rules += ["Treat this as a wordplay clue."]
        if subroute == "rhyme": base_rules += ["Bias toward rhyme-based reasoning."]
        elif subroute == "homophone": base_rules += ["Bias toward homophone or sounds-like reasoning."]
        elif subroute == "anagram": base_rules += ["Bias toward anagram-based reasoning."]
        elif subroute == "before_after": base_rules += ["Bias toward before-and-after phrase composition."]
        else: base_rules += ["Use general wordplay reasoning."]
    elif route == "completion":
        base_rules += ["Treat this as a completion clue."]
        if subroute == "quote_completion": base_rules += ["Bias toward completing a quotation."]
        elif subroute == "title_completion": base_rules += ["Bias toward completing a title."]
        elif subroute == "proverb_completion": base_rules += ["Bias toward completing a proverb."]
        else: base_rules += ["Bias toward completing a phrase."]
    elif route == "spelling":
        base_rules += ["Treat this as a spelling or form clue."]
        if subroute == "starts_with": base_rules += ["Bias toward starts-with constraints."]
        elif subroute == "ends_with": base_rules += ["Bias toward ends-with constraints."]
        elif subroute == "word_length": base_rules += ["Bias toward word-length constraints."]
        else: base_rules += ["Bias toward letter-pattern constraints."]
    elif route == "multi":
        base_rules += ["Treat this as a multi-item, set-selection, or composite clue."]
        if subroute == "list_selection": base_rules += ["Bias toward selecting from a set."]
        elif subroute == "gimmick_category": base_rules += ["Bias toward a gimmick-category interpretation."]
        else: base_rules += ["Bias toward multi-item reasoning."]

    return " ".join(base_rules)

def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', raw_text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return {}

def parse_input_text(input_text: str) -> Tuple[str, str]:
    pattern = r"Category:\s*(.*?)\nClue:\s*(.*)"
    match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    lines = input_text.split('\n')
    cat = lines[0].replace("Category:", "").strip() if len(lines) > 0 else "unknown"
    clue = lines[1].replace("Clue:", "").strip() if len(lines) > 1 else input_text
    return cat, clue

def process_single_sample(sample, task_id):
    """处理单条数据的逻辑"""
    sample_id = sample.get("id", "") 
    input_text = sample["input"] 

    content = "input: " + input_text
    messages = [
        {"role": "system", "content": ROUTE_PROMPT},
        {"role": "user", "content": content}
    ]
    raw_output = qwen_api(messages)
    raw_str = str(raw_output).strip()
    route_output = {}

    # 1. 先尝试用 JSON 解析
    try:
        json_match = re.search(r'\{.*?\}', raw_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            route_output = json.loads(json_str)
    except:
        route_output = {}

    # 2. 如果 JSON 解析失败 → 用正则强制提取（兜底方案）
    if not route_output.get("route") or not route_output.get("subroute"):
        # 提取 route
        route_match = re.search(r'"route"\s*:\s*"([^"]+)"', raw_str)
        # 提取 subroute
        subroute_match = re.search(r'"subroute"\s*:\s*"([^"]+)"', raw_str)
        
        route_output = {
            "route": route_match.group(1).strip() if route_match else "",
            "subroute": subroute_match.group(1).strip() if subroute_match else ""
        }

    # --------------------- 最终拿到结果 ---------------------
    route = route_output.get("route", "")
    subroute = route_output.get("subroute", "")
    # 两阶段路由作为二阶段提示词
    category, clue = parse_input_text(input_text)
    # 构造提示词
    payload = {
        "category": category,
        "clue": clue,
        "forced_route": route,
        "forced_subroute": subroute,
        "instruction": build_route_instruction(route, subroute)
    }
    
    content2 = "input: " + input_text
    messages2 = [
        {"role": "system", "content": PROBLEM_SOLVE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
    raw_output = qwen_api(messages2)
    raw_str = str(raw_output).strip()
    data = extract_json_from_text(raw_str)
    final_answer_raw = data.get("final_answer", raw_str).lower()

    # 将需要保存的所有信息打包成一个字典返回
    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "input": input_text,
        "model_output": final_answer_raw,
    }

def process_single_example(sample, task_id):
    """处理单条数据的逻辑"""
    sample_id = sample.get("id", "") 
    input_text = sample["input"] 
    gt = sample["output"][0]

    content = "input: " + input_text
    messages = [
        {"role": "system", "content": ROUTE_PROMPT},
        {"role": "user", "content": content}
    ]
    raw_output = qwen_api(messages)
    raw_str = str(raw_output).strip()
    route_output = {}

    # 1. 先尝试用 JSON 解析
    try:
        json_match = re.search(r'\{.*?\}', raw_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            route_output = json.loads(json_str)
    except:
        route_output = {}

    # 2. 如果 JSON 解析失败 → 用正则强制提取（兜底方案）
    if not route_output.get("route") or not route_output.get("subroute"):
        # 提取 route
        route_match = re.search(r'"route"\s*:\s*"([^"]+)"', raw_str)
        # 提取 subroute
        subroute_match = re.search(r'"subroute"\s*:\s*"([^"]+)"', raw_str)
        
        route_output = {
            "route": route_match.group(1).strip() if route_match else "",
            "subroute": subroute_match.group(1).strip() if subroute_match else ""
        }

    # --------------------- 最终拿到结果 ---------------------
    route = route_output.get("route", "")
    subroute = route_output.get("subroute", "")
    # 两阶段路由作为二阶段提示词
    category, clue = parse_input_text(input_text)
    # 构造提示词
    payload = {
        "category": category,
        "clue": clue,
        "forced_route": route,
        "forced_subroute": subroute,
        "instruction": build_route_instruction(route, subroute)
    }

    content2 = "input: " + input_text
    messages2 = [
        {"role": "system", "content": PROBLEM_SOLVE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
    raw_output = qwen_api(messages2)
    raw_str = str(raw_output).strip()
    data = extract_json_from_text(raw_str)
    final_answer_raw = data.get("final_answer", raw_str).lower()

    # 将需要保存的所有信息打包成一个字典返回
    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "input": input_text,
        "model_output": final_answer_raw,
        "predict_right":gt==final_answer_raw
    }

def task5_data_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("task_id"), data.get("examples", []), data.get("test_samples", [])


if __name__ == "__main__":
    file_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\data\openseek-7_jeopardy_answer_generation_all.json"
    out_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\rgs_q7\experiment\openseek-7-v1.jsonl"
    
    task5_id, examples, test_samples = task5_data_loader(file_path)
    eval_data = test_samples
    total_cnt = len(eval_data)
    
    eval_flag = False
    if eval_flag:
        # 准备一个 partial 函数，固定住 system_prompt 和 task_id，方便 map 调用
        process_func = partial(process_single_sample, task_id=task5_id)
        
        max_workers = 200 
        results_list = []
        correct_cnt = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用 executor.map 替代 as_completed，以保证返回顺序与 eval_data 完全一致
            # executor.map 会自动并发，但按输入顺序 yield 结果
            for result in tqdm(executor.map(process_func, eval_data), total=total_cnt, desc="Evaluating"):
                results_list.append(result)

        # 提取 sample_id 和 model_output
        output_data = [
            {
                "test_sample_id": item["sample_id"],
                "prediction": item["model_output"]
            }
            for item in results_list
        ]

        # 确保输出路径的文件夹存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                # 将单个字典转换为 JSON 字符串，然后手动添加换行符
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + '\n')

        print(f"评测结果已成功保存为 JSONL 格式：{out_path}")
    else:
        process_func = partial(process_single_example, task_id=task5_id)
        data = examples[:]
        total_cnt = len(data)
        correct_cnt = 0
        max_workers = 200 
        results_list = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用 executor.map 替代 as_completed，以保证返回顺序与 eval_data 完全一致
            # executor.map 会自动并发，但按输入顺序 yield 结果
            for result in tqdm(executor.map(process_func, data), total=total_cnt, desc="Evaluating"):
                results_list.append(result)

        for result in results_list:
            if result["predict_right"] == True:
                correct_cnt+=1
        
        correct_rate = float(correct_cnt)/float(total_cnt)
        print(f"总共{total_cnt}条，正确{correct_cnt}条\n正确率：{correct_rate}")