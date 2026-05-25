import json
import os
import requests
import time
import re
from tqdm import tqdm

# ================= 配置区 =================
MODEL_NAME = "Qwen3-4B"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
ORIG_FILE = "../data/openseek-7_jeopardy_answer_generation_all.json"
OUT_FILE = "../outputs/openseek-7-vLLM_Final.jsonl"
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_jeopardy_sniper(input_text, max_retries=3):
    """补漏专用的狙击请求，带有强力防思考枷锁"""
    
    prompt_text = (
        "You are a Jeopardy champion. Read the Category and the Clue, then provide the exact, concise answer.\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Output ONLY the answer entity.\n"
        "2. The answer MUST be in ALL LOWERCASE letters.\n"
        "3. Wrap your final answer in <label> tags at the very end.\n"
        "4. DO NOT OVERTHINK. If you are unsure, just guess the most likely short noun phrase. DO NOT write long explanations.\n\n"
        "Example 1:\n"
        "Input: Category: KANSAS CITY, KANSAS HERE WE COME \nClue: The Kansas 400 at KCK's Kansas Speedway is part of this auto racing organization's Nextel Cup\n"
        "Output: <label>nascar</label>\n\n"
        f"Input: {input_text}\n"
        "Output:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 1024,
        "temperature": 0.1  # 稍微给一点点温度，让它在僵局中能“猜”出一个答案
    }
    
    for attempt in range(max_retries):
        try:
            # 极限超时时间
            response = requests.post(API_URL, json=payload, timeout=120) 
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                label_match = re.search(r'<label>\s*(.*?)\s*</label>', clean_res, re.I | re.DOTALL)
                if label_match:
                    ans = label_match.group(1).strip()
                    return ans.lower() 
                
                lines = [line.strip() for line in clean_res.split('\n') if line.strip()]
                if lines:
                    return lines[-1].lower()
                
            else:
                time.sleep(2) 
                
        except Exception as e:
            time.sleep(2)
            continue
            
    return "unknown" # 如果单线程+120秒还失败，说明这题真的是死穴了

def main():
    if not os.path.exists(OUT_FILE) or not os.path.exists(ORIG_FILE):
        print("❌ 找不到原始文件或输出文件！")
        return

    # 1. 读取原始题库，建立 ID -> 文本 的映射
    with open(ORIG_FILE, 'r') as f:
        data = json.load(f)
        samples_dict = {s['id']: s['input'] for s in data['test_samples']}

    # 2. 扫荡输出文件，抓出所有的 unknown，并保留其他正常数据
    all_records = []
    unknown_ids = []
    
    with open(OUT_FILE, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                all_records.append(record)
                if record['prediction'] == 'unknown':
                    unknown_ids.append(record['test_sample_id'])
            except: pass

    print(f"🎯 扫描完毕！发现 {len(unknown_ids)} 个 'unknown' 目标，准备进行狙击重跑...")
    if not unknown_ids:
        print("✅ 完美！没有发现任何 unknown，不需要补漏。")
        return

    # 3. 开始单线程狙击
    fixed_count = 0
    for record in tqdm(all_records, desc="Sniper Retry"):
        if record['prediction'] == 'unknown':
            sid = record['test_sample_id']
            input_text = samples_dict.get(sid, "")
            
            # 发起狙击请求
            new_pred = request_jeopardy_sniper(input_text)
            
            # 更新记录
            record['prediction'] = new_pred
            if new_pred != 'unknown':
                fixed_count += 1

    # 4. 把修复后的数据重新覆盖写回原文件
    with open(OUT_FILE, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"\n🎉 补漏行动结束！成功修复了 {fixed_count} / {len(unknown_ids)} 个逃兵数据！")

if __name__ == "__main__":
    main()