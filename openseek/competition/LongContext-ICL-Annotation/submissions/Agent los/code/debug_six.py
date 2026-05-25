import json
import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
TASK_ID = 6
MODEL_NAME = "Qwen3-4B"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# 请确认你的原始文件路径是否匹配
ORIG_FILE = "../data/openseek-6_mnli_same_genre_classification.json"
OUT_FILE = "../outputs/openseek-6-vLLM_Final.jsonl"
WORKERS = 4  # 依然保持 4 并发
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_mnli(input_text, max_retries=3):
    """MNLI 体裁分类专属请求函数"""
    
    # 规则注入：把官方的体裁定义强行塞给模型
    prompt_text = (
        "You are an expert text classifier. Read Sentence 1, Sentence 2, and the given Genre.\n"
        "Your job is to determine if BOTH sentences belong to the given Genre. If yes, output Y. If no, output N.\n\n"
        "Definitions of Genres:\n"
        "- face-to-face: conversations or dialogues\n"
        "- government: information released from public government websites\n"
        "- letters: written work for philanthropic fundraising\n"
        "- 9/11: information pertaining to the 9/11 attacks\n"
        "- slate: cultural topic that appears in the slate magazine\n"
        "- telephone: telephonic dialogue\n"
        "- travel: information in travel guides\n"
        "- verbatim: short posts regarding linguistics\n"
        "- oup: non-fiction works on the textile industry and child development\n"
        "- fiction: popular works of fiction\n\n"
        "You MUST output ONLY 'Y' or 'N'. Wrap your final decision in <label> tags.\n\n"
        "Example 1:\n"
        "Input: Sentence 1: I do not have the energy to remedy these deficiencies now. Sentence 2: I don't have the strength to fix these problems now. Genre: slate.\n"
        "Output: <label>Y</label>\n\n"
        "Example 2:\n"
        "Input: Sentence 1: were you have you i take it you haven't spent any time in the military Sentence 2: Jon said there is nothing else we can do. Genre: telephone.\n"
        "Output: <label>N</label>\n\n"
        f"Input: {input_text}\n"
        "Output:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 1536,  # 允许充分推导对比
        "temperature": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=60) 
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                
                # 切除 <think> 过程
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                # 第一层：精确匹配标签
                label_match = re.search(r'<label>\s*(Y|N)\s*</label>', clean_res, re.I)
                if label_match:
                    return label_match.group(1).upper()
                
                # 第二层：如果没标签，找清洗后的结尾（防截断）
                tail_clean = clean_res.upper()[-30:]
                if re.search(r'\bY\b', tail_clean): return "Y"
                if re.search(r'\bN\b', tail_clean): return "N"
                
                # 第三层：终极在原始文本找
                tail_raw = raw_res.upper()[-30:]
                if re.search(r'\bY\b', tail_raw): return "Y"
                if re.search(r'\bN\b', tail_raw): return "N"

                print(f"\n[抓取失败] 输入: {input_text[:50]}... | 模型结尾: {repr(raw_res[-100:])}")
            else:
                time.sleep(1 * (attempt + 1)) 
                
        except Exception as e:
            time.sleep(1.5)
            continue
            
    # 终极兜底：MNLI里绝大多数不是同一个Genre，返回 N 是最安全的策略
    return "N"

def process_sample(sample_id, input_text):
    res = request_mnli(input_text)
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
    print(f"📊 任务6 (MNLI Genre) 待处理: {len(remaining)} / {len(samples)}")

    if not remaining:
        print("✅ 任务6 已全部完成！")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_id = {
            executor.submit(process_sample, s['id'], s['input']): s['id'] 
            for s in remaining
        }
        
        with open(OUT_FILE, 'a') as f:
            for future in tqdm(as_completed(future_to_id), total=len(remaining), desc="Task 6"):
                sid, pred = future.result()
                if pred:
                    f.write(json.dumps({"test_sample_id": sid, "prediction": pred}) + "\n")
                    f.flush()

if __name__ == "__main__":
    main()