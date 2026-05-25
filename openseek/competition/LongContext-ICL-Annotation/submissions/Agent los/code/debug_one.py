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
# 因为每次请求变成了单个数字，负荷变小了，可以把并发提上来
WORKERS = 1
# ==========================================

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def request_single_number(num_str, max_retries=3):
    """顺应推理模型天性：允许充分思考，三层漏斗极限抓取"""
    
    prompt_text = (
        "We are doing a SINGLE STEP math calculation.\n"
        "- If the Input is even, divide it by 2.\n"
        "- If the Input is odd, multiply it by 3 and add 1.\n"
        "Take your time to think step-by-step. "
        "However, you MUST output your final calculated number wrapped in <label> tags at the very end.\n\n"
        f"Input: {num_str}\n"
        "Output:"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 2048,  # 【必须给足】让它把 <think> 念叨完
        "temperature": 0.0
        # ⚠️ 不加任何 stop，防止它在思考中途触发关键词自杀
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=180) 
            
            if response.status_code == 200:
                raw_res = response.json()["choices"][0]["message"]["content"]
                
                # 物理切除思考过程（非贪婪匹配，处理完整或残缺的 think）
                clean_res = re.sub(r'<think>.*?(?:</think>|$)', '', raw_res, flags=re.DOTALL)
                
                # 【第一层漏斗】：乖乖听话带了 <label> 的
                label_match = re.search(r'<label>\s*(\d+)', clean_res, re.I)
                if label_match:
                    return label_match.group(1)
                
                # 【第二层漏斗】：忘记带标签，但在思考结束后输出了数字
                nums_in_clean = re.findall(r'\d+', clean_res)
                if nums_in_clean:
                    return nums_in_clean[-1]
                    
                # 【第三层漏斗】：它把答案写在了 <think> 里面并且忘记闭合了！
                nums_in_raw = re.findall(r'\d+', raw_res)
                if nums_in_raw:
                    # 抓取它整个生命周期里吐出的最后一个数字
                    return nums_in_raw[-1]
                
                print(f"\n[抓取失败] 输入: {num_str} | 模型回复: {repr(raw_res[-100:])}")
            else:
                print(f"\n[API 拒绝] 状态码: {response.status_code}")
                time.sleep(1 * (attempt + 1)) 
                
        except Exception as e:
            print(f"\n[网络异常] 输入: {num_str} | 报错: {e}") 
            time.sleep(1.5)
            continue
            
    return "0"
def process_sample(sample_id, input_text):
    """处理整个数组：拆分开来算，再合并"""
    # 把 "[165, 130, 80]" 拆成 ['165', '130', '80']
    nums_to_process = re.findall(r'\d+', input_text)
    
    results = []
    for num in nums_to_process:
        # 逐个数字丢给 LLM 算
        res = request_single_number(num)
        results.append(res)
        
    # 把算出的单个步数合并成最终格式，例如 "[15, 23, 8]"
    final_prediction = "[" + ", ".join(results) + "]"
    return sample_id, final_prediction

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
    print(f"📊 任务3 (Collatz) 待处理: {len(remaining)} / {len(samples)}")

    if not remaining:
        print("✅ 任务3 已全部完成！")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    # 🌟 调用 process_sample，它会负责把数组拆开算
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