import requests

# url = "http://0.0.0.0:2026/v1/completions"
url = "https://api-inference.modelscope.cn/v1/chat/completions"
# url = "https://api-inference.modelscope.cn/v1"

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ms-c429b084-79ba-4a00-a749-aae8681e902d',
}

prompts = [
    # "Hello, FlagScale + vLLM!",
    "你是谁？",
    # "Write a short poem about autumn."
    # '用中文写一首短诗，诗句开头用<label>，结尾用</label>包裹起来'
]

for prompt in prompts:
    data = {
        "model": "Qwen/Qwen3-8B",
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": "什么是大模型"}
        ],
        "max_tokens": 128,
        "enable_thinking": False,
        # "chat_template_kwargs": {
        #     "enable_thinking": False
        # },
        # "extra_body":
        #     {
        #         "chat_template_kwargs": {
        #             "enable_thinking": False
        #         },
        #     },
        "stream": False
    }
    resp = requests.post(url, json=data, headers=headers)
    resp_obj = resp.json()
    # print(f"Prompt: {prompt}")
    print("content:", resp_obj['choices'][0]['message']['content'])
    # print("reasoning", resp_obj['choices'][0]['message']['reasoning'])
    # print("reasoning_content", resp_obj['choices'][0]['message']['reasoning_content'])

    print("*" * 50)

# from openai import OpenAI
#
# client = OpenAI(
#     base_url='https://api-inference.modelscope.cn/v1',
#     api_key='ms-c429b084-79ba-4a00-a749-aae8681e902d',  # ModelScope Token
# )
#
# response = client.chat.completions.create(
#     model='Qwen/Qwen3-8B',  # ModelScope Model-Id
#     messages=[
#         {
#             'role': 'system',
#             'content': 'You are a helpful assistant.'
#         },
#         {
#             'role': 'user',
#             'content': '你好'
#         }
#     ],
#     extra_body={
#         # enable thinking, set to False to disable test
#         "enable_thinking": False
#     },
#     stream=False
# )
#
# print(response.choices[0].message.content)
# print(1)
