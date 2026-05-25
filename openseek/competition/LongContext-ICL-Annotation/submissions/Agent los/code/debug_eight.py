import json
import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
TASK_ID = 8
MODEL_NAME = "Qwen3-4B"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# 请确保输入文件路径正确
ORIG_FILE = "../data/openseek-8_kernel_generation.json"
OUT_FILE = "../outputs/openseek-8-vLLM_Final.jsonl"
WORKERS = 2  # 长文本生成显存压力极大，建议并发设为 2，最稳
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_triton_code(input_text, max_retries=3):
    """Triton 代码生成专属请求函数（强力镇压思考版）"""
    
    prompt_text = (
        "You are an elite AI system architect and GPU optimization expert.\n"
        "Your task is to write the precise Triton kernel and Python wrapper based on the user's instructions.\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Write ONLY the Python/Triton code.\n"
        "2. Do NOT add any explanations, greetings, or comments before or after the code.\n"
        "3. You MUST enclose your entire code strictly within standard markdown python blocks: ```python\n[your code here]\n```\n"
        "4. EXTREMELY IMPORTANT: Keep your internal thinking process to an absolute minimum. DO NOT write long essays. Output the ```python code block IMMEDIATELY.\n\n"
        f"Instruction:\n{input_text}\n"
        "Output:\n```python\n"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 6144,  # 🔥 显存极限拉升，容纳超长代码
        "temperature": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=180) # 🔥 超时拉到 3 分钟
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                
                # 切除 <think> 过程
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                # 核心抓取
                code_match = re.search(r'```(?:python)?\n(.*?)\n```', clean_res, re.IGNORECASE | re.DOTALL)
                if code_match:
                    return code_match.group(1).strip()
                
                # 兜底抓取
                clean_str = clean_res.replace("```python", "").replace("```", "").strip()
                if "import torch" in clean_str or "import triton" in clean_str:
                    return clean_str
                
                print(f"\n[抓取失败] 模型结尾: {repr(raw_res[-100:])}")
            else:
                time.sleep(2 * (attempt + 1)) 
                
        except Exception as e:
            time.sleep(2)
            continue
            
    # 如果失败，返回 None 而不是空字符串，防止写入废数据
    return None

def process_sample(sample_id, input_text):
    res = request_triton_code(input_text)
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
    print(f"📊 任务8 (Triton Kernel) 待处理: {len(remaining)} / {len(samples)}")

    if not remaining:
        print("✅ 任务8 已全部完成！恭喜通关全量任务！")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_id = {
            executor.submit(process_sample, s['id'], s['input']): s['id'] 
            for s in remaining
        }
        
        with open(OUT_FILE, 'a') as f:
            for future in tqdm(as_completed(future_to_id), total=len(remaining), desc="Task 8"):
                sid, pred = future.result()
                if pred:  # 👈 确保抓取到了真实代码才写入，失败的直接跳过，方便后面补漏
                    f.write(json.dumps({"test_sample_id": sid, "prediction": pred}) + "\n")
                    f.flush()

if __name__ == "__main__":
    main()