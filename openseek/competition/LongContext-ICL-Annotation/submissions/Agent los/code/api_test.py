import requests
import time
import json

def test_vllm_api():
    # 明确指向咱们刚刚跑通的 vLLM 原生 8000 端口
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # 构造标准的 OpenAI Chat API 请求体
    payload = {
        "model": "Qwen3-4B",
        "messages": [
            {"role": "system", "content": "You are a highly efficient AI assistant running on a Huawei Ascend 910C NPU cluster."},
            {"role": "user", "content": "Hello! Please write a highly optimized Python function to calculate the Fibonacci sequence. Keep your explanation extremely brief."}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": False # 竞赛批量评测时通常不开 stream
    }

    print(f"🚀 发送测试请求至: {url}")
    print(f"📦 目标模型: {payload['model']}")
    start_time = time.time()

    try:
        # 发起 POST 请求，设置 60 秒超时防止死锁
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        elapsed_time = time.time() - start_time

        # 1. 拦截非 200 状态码，直接暴露底层服务器报错
        if response.status_code != 200:
            print(f"\n❌ 请求失败！HTTP 状态码: {response.status_code}")
            print(f"📄 原始响应内容: \n{response.text}")
            return

        # 2. 安全解析 JSON，彻底告别盲目的 resp.json() 崩溃
        response_data = response.json()

        print(f"\n✅ 请求成功！耗时: {elapsed_time:.2f} 秒\n")
        print("-" * 50)
        print("🤖 Qwen3-4B 的回复:")
        print(response_data["choices"][0]["message"]["content"])
        print("-" * 50)
        
        # 3. 打印对评估极其重要的用量信息
        usage = response_data.get('usage', {})
        print(f"📊 资源消耗统计:")
        print(f"   - Prompt Tokens:     {usage.get('prompt_tokens', 'N/A')}")
        print(f"   - Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"   - Total Tokens:      {usage.get('total_tokens', 'N/A')}")

    except requests.exceptions.JSONDecodeError:
        print("\n❌ JSON 解析失败！服务器返回的不是规范的 JSON 数据。")
        print(f"📄 原始响应文本: \n{response.text}")
    except requests.exceptions.ConnectionError:
        print("\n❌ 连接被拒绝！请确认 vLLM 服务是否依然在 8000 端口存活，并且没有发生 OOM 崩溃。")
    except requests.exceptions.Timeout:
        print("\n❌ 请求超时！模型可能在处理超长文本或发生死锁。")
    except Exception as e:
        print(f"\n❌ 发生未预期的异常: {str(e)}")

if __name__ == "__main__":
    test_vllm_api()