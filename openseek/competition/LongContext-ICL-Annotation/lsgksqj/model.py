from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("正在从国内源检查/下载模型...")
model_dir = snapshot_download('qwen/Qwen3-4B')
print(f"模型路径: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(
    model_dir, 
    use_fast=False, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto", 
    device_map="auto",
    trust_remote_code=True
)

prompt = "介绍一下你自己"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True 
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("正在生成回复...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

try:
    stop_token_id = tokenizer.convert_tokens_to_ids("</think>")
    index = len(output_ids) - output_ids[::-1].index(stop_token_id)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n" + "="*20 + " 结果输出 " + "="*20)
print("thinking content:", thinking_content)
print("content:", content)