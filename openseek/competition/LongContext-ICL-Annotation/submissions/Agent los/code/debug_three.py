import json
import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
TASK_ID = 3
MODEL_NAME = "Qwen3-4B"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
ORIG_FILE = "../data/openseek-3_collatz_conjecture.json"
OUT_FILE = "../outputs/openseek-3-vLLM_Final.jsonl"

# 针对最后几个易超时的数字，建议设为 1；如果还在跑大量数据，可以设为 2-4
WORKERS = 1  
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_single_number(num_str, max_retries=3):
    """Task 3 终极版：顺应推理天性 + 三层极限抓取"""
    prompt_text = (
        "We are doing a SINGLE STEP calculation. Apply this math logic to the Input number:\n"
        "- If it is even, divide it by 2.\n"
        "- If it is odd, multiply it by 3 and add 1.\n"
        "CRITICAL: ONLY PERFORM EXACTLY ONE STEP. STOP after one operation.\n"
        "You MUST output your final calculated number wrapped in <label> tags.\n\n"
        "Example 1:\n"
        "Input: 12\n"
        "Output: <label>6</label>\n\n"
        f"Input: {num_str}\n"
        "Output:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 2048,  # 给足空间让模型“碎碎念”
        "temperature": 0.0
        # ⚠️ 不使用 stop，防止在思考中途断开
    }
    
    for attempt in range(max_retries):
        try:
            # 针对刺客数字，超时给到 180 秒
            response = requests.post(API_URL, json=payload, timeout=180) 
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                
                # 物理切除思考过程
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                # 【第一层】：匹配 <label> 标签
                label_matches = re.findall(r'<label>\s*(\d+)', raw_res, re.I)
                if label_matches:
                    return label_matches[-1]
                
                # 【第二层】：匹配清洗后文本中的数字
                nums_clean = re.findall(r'\d+', clean_res)
                if nums_clean:
                    return nums_clean[-1]
                
                # 【第三层】：直接抓取原始回复中出现的最后一个数字（最后防线）
                nums_raw = re.findall(r'\d+', raw_res)
                if nums_raw:
                    return nums_raw[-1]
                
                print(f"\n[抓取失败] 输入: {num_str} | 模型回复片段: {repr(raw_res[-100:])}")
            else:
                time.sleep(2) 
                
        except Exception as e:
            print(f"\n[请求异常] 输入: {num_str} | 错误: {e}")
            time.sleep(2)
            continue
            
    return "0"

def process_sample(sample_id, input_text):
    res = request_single_number(input_text)
    return sample_id, res

def main():
    if not os.path.exists(ORIG_FILE):
        print(f"❌ 找不到输入文件: {ORIG_FILE}")
        return
    
    with open(ORIG_FILE, 'r') as f:
        data = json.load(f)
        samples = data['test_samples']

    # --- 断点续传逻辑 ---
    processed_ids = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, 'r') as f:
            for line in f:
                try: 
                    processed_ids.add(json.loads(line)['test_sample_id'])
                except: pass
    
    remaining = [s for s in samples if s['id'] not in processed_ids]
    print(f"📊 Task 3 进度: {len(samples)-len(remaining)}/{len(samples)} (剩余: {len(remaining)})")

    if not remaining:
        print("✅ Task 3 已全部完成！")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_id = {
            executor.submit(process_sample, s['id'], s['input']): s['id'] 
            for s in remaining
        }
        
        with open(OUT_FILE, 'a') as f:
            for future in tqdm(as_completed(future_to_id), total=len(remaining), desc="Task 3"):
                sid, pred = future.result()
                if pred:
                    f.write(json.dumps({"test_sample_id": sid, "prediction": pred}) + "\n")
                    f.flush()

if __name__ == "__main__":
    main()