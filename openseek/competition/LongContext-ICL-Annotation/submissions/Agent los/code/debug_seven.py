import json
import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
TASK_ID = 7
MODEL_NAME = "Qwen3-4B"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# 确认输入文件路径
ORIG_FILE = "../data/openseek-7_jeopardy_answer_generation_all.json"
OUT_FILE = "../outputs/openseek-7-vLLM_Final.jsonl"
WORKERS = 4  # 知识问答任务，并发 4 没问题
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_jeopardy(input_text, max_retries=3):
    """Jeopardy 问答专属请求函数"""
    
    prompt_text = (
        "You are a Jeopardy champion. Read the Category and the Clue, then provide the exact, concise answer.\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Output ONLY the answer entity.\n"
        "2. The answer MUST be in ALL LOWERCASE letters.\n"
        "3. Wrap your final answer in <label> tags at the very end.\n\n"
        "Example 1:\n"
        "Input: Category: KANSAS CITY, KANSAS HERE WE COME \nClue: The Kansas 400 at KCK's Kansas Speedway is part of this auto racing organization's Nextel Cup\n"
        "Output: <label>nascar</label>\n\n"
        "Example 2:\n"
        "Input: Category: COFFEE \nClue: Ladyfingers are a common ingredient of this coffee-flavored Italian dessert\n"
        "Output: <label>tiramisu</label>\n\n"
        f"Input: {input_text}\n"
        "Output:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 1024,  # 给足它回忆和思考的空间
        "temperature": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=40) 
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                
                # 切除 <think> 过程
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                # 第一层提取：寻找标签内的答案，并强制转为小写！
                label_match = re.search(r'<label>\s*(.*?)\s*</label>', clean_res, re.I | re.DOTALL)
                if label_match:
                    ans = label_match.group(1).strip()
                    return ans.lower()  # 🌟 终极护盾：Python 强制小写
                
                # 第二层提取：如果模型忘了标签，尝试抓取清理后文本的最后一行或最后几个词
                lines = [line.strip() for line in clean_res.split('\n') if line.strip()]
                if lines:
                    return lines[-1].lower() # 🌟 终极护盾：Python 强制小写
                
                print(f"\n[抓取失败] 输入: {input_text[:50]}... | 模型回复: {repr(raw_res[-100:])}")
            else:
                time.sleep(1 * (attempt + 1)) 
                
        except Exception as e:
            time.sleep(1.5)
            continue
            
    # 如果 API 彻底崩溃，返回默认字符串防报错
    return "unknown"

def process_sample(sample_id, input_text):
    res = request_jeopardy(input_text)
    return sample_id, res

def main():
    if not os.path.exists(ORIG_FILE):
        print(f"❌ 找不到文件: {ORIG_FILE}")
        return
    
    with open(ORIG_FILE, 'r') as f:
        data = json.load(f)
        samples = data['test_samples']

    processed_ids = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, 'r') as f:
            for line in f:
                try: processed_ids.add(json.loads(line)['test_sample_id'])
                except: pass
    
    remaining = [s for s in samples if s['id'] not in processed_ids]
    print(f"📊 任务7 (Jeopardy) 待处理: {len(remaining)} / {len(samples)}")

    if not remaining:
        print("✅ 任务7 已全部完成！")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_id = {
            executor.submit(process_sample, s['id'], s['input']): s['id'] 
            for s in remaining
        }
        
        with open(OUT_FILE, 'a') as f:
            for future in tqdm(as_completed(future_to_id), total=len(remaining), desc="Task 7"):
                sid, pred = future.result()
                if pred:
                    f.write(json.dumps({"test_sample_id": sid, "prediction": pred}) + "\n")
                    f.flush()

if __name__ == "__main__":
    main()