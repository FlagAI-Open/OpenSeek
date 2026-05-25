import json
import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
TASK_ID = 2
MODEL_NAME = "Qwen3-4B"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# 请确保输入文件路径正确
ORIG_FILE = "../data/openseek-2_count_nouns_verbs.json"
OUT_FILE = "../outputs/openseek-2-vLLM_Final.jsonl"
WORKERS = 4 
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_count_pos(input_text, max_retries=3):
    """词性计数专属请求函数，强制提取单数字"""
    
    prompt_text = (
        "You are a strict linguistics expert. Read the sentence and count EXACTLY what is requested (either nouns OR verbs).\n"
        "CRITICAL RULES:\n"
        "1. Output ONLY a single integer representing the exact count requested.\n"
        "2. Do NOT write 'Nouns: X, Verbs: Y'. Do NOT write any extra words.\n"
        "3. You MUST wrap your final integer in <label> tags.\n\n"
        "Example 1:\n"
        "Input: Sentence: 'The ladder of a jet is lowered from the side for loading passengers'. Count the number of verbs in this sentence.\n"
        "Output: <label>2</label>\n\n"
        f"Input: {input_text}\n"
        "Output:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 512,  # 给它一点空间在 <think> 里数数
        "temperature": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=30) 
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                
                # 切除 <think> 过程
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                # 第一层：精确匹配 <label> 里的数字
                label_match = re.search(r'<label>\s*(\d+)\s*</label>', clean_res, re.I)
                if label_match:
                    return label_match.group(1)
                
                # 第二层：如果它没用标签，直接暴力提取清理后文本中的最后一个数字
                nums = re.findall(r'\d+', clean_res)
                if nums:
                    return nums[-1]
                
            else:
                time.sleep(1) 
                
        except Exception:
            time.sleep(1)
            continue
            
    # 终极兜底
    return "0"

def process_sample(sample_id, input_text):
    res = request_count_pos(input_text)
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
    print(f"📊 任务2 (Count Nouns/Verbs) 待处理: {len(remaining)} / {len(samples)}")

    if not remaining:
        print("✅ 任务2 已全部完成！")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_id = {
            executor.submit(process_sample, s['id'], s['input']): s['id'] 
            for s in remaining
        }
        
        with open(OUT_FILE, 'a') as f:
            for future in tqdm(as_completed(future_to_id), total=len(remaining), desc="Task 2"):
                sid, pred = future.result()
                if pred:
                    f.write(json.dumps({"test_sample_id": sid, "prediction": pred}) + "\n")
                    f.flush()

if __name__ == "__main__":
    main()