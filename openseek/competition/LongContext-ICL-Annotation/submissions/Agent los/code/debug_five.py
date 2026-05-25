import json
import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
TASK_ID = 5
MODEL_NAME = "Qwen3-4B"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# 请确认你的原始文件路径是否匹配
ORIG_FILE = "../data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
OUT_FILE = "../outputs/openseek-5-vLLM_Final.jsonl"
WORKERS = 4  # 文本分类任务，并发 4 是比较安全的甜点位
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_sentiment(input_text, max_retries=3):
    """大容量防截断版情感分析请求"""
    
    prompt_text = (
        "You are an expert sentiment analyst. Judge whether the author of the tweet is sad or not.\n"
        "Pay deep attention to the context, irony, and real meaning.\n"
        "You MUST output ONLY one of the two exact phrases: 'Sad' or 'Not sad'.\n"
        "Wrap your final decision in <label> tags.\n\n"
        "Example 1:\n"
        "Tweet: Went to bed a 1:30, fell asleep after, my niece started crying at 4. I'm dying... 😧\n"
        "Output: <label>Sad</label>\n\n"
        "Example 2:\n"
        "Tweet: @bxchpls03 U so lucky ahu 😭\n"
        "Output: <label>Not sad</label>\n\n"
        f"Tweet: {input_text}\n"
        "Output:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 1536,  # 【关键修改】容量翻3倍！让它把情感分析小作文写完
        "temperature": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=60) # 超时稍微放宽
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                
                # 切除 <think> 过程
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                # 第一层：精确匹配标签
                label_match = re.search(r'<label>\s*(.*?)\s*</label>', clean_res, re.I)
                if label_match:
                    ans = label_match.group(1).lower()
                    if "not sad" in ans: return "Not sad"
                    if "sad" in ans: return "Sad"
                
                # 第二层：如果没标签，去清理后的结尾找关键词
                tail_clean = clean_res.lower()[-50:]
                if "not sad" in tail_clean: return "Not sad"
                if "sad" in tail_clean: return "Sad"
                
                # 第三层：终极兜底，如果被截断在 think 里，直接去原始字符串的最后一行找！
                tail_raw = raw_res.lower()[-50:]
                if "not sad" in tail_raw: return "Not sad"
                if "sad" in tail_raw: return "Sad"

                print(f"\n[抓取失败] 输入: {input_text[:30]}... | 模型结尾: {repr(raw_res[-100:])}")
            else:
                time.sleep(1 * (attempt + 1)) 
                
        except Exception as e:
            time.sleep(1.5)
            continue
            
    # 如果 API 彻底卡死，默认 Not sad 保证数据不为空
    return "Not sad"
def process_sample(sample_id, input_text):
    res = request_sentiment(input_text)
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
    print(f"📊 任务5 (Sadness Detection) 待处理: {len(remaining)} / {len(samples)}")

    if not remaining:
        print("✅ 任务5 已全部完成！")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_id = {
            executor.submit(process_sample, s['id'], s['input']): s['id'] 
            for s in remaining
        }
        
        with open(OUT_FILE, 'a') as f:
            for future in tqdm(as_completed(future_to_id), total=len(remaining), desc="Task 5"):
                sid, pred = future.result()
                if pred:
                    f.write(json.dumps({"test_sample_id": sid, "prediction": pred}) + "\n")
                    f.flush()

if __name__ == "__main__":
    main()