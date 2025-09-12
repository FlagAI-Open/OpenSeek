from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# export HF_ENDPOINT=https://hf-mirror.com
# export XDG_CACHE_HOME=./cache

tokenizer = AutoTokenizer.from_pretrained("BAAI/OpenSeek-Small-v1-SFT", trust_remote_code=True)
tokenizer.save_pretrained("./models/OpenSeek-Small-v1-SFT")
model = AutoModelForCausalLM.from_pretrained("BAAI/OpenSeek-Small-v1-SFT", trust_remote_code=True).to(torch.bfloat16)
model.save_pretrained("./models/OpenSeek-Small-v1-SFT")

numina_math_cot = load_dataset("AI-MO/NuminaMath-CoT", num_proc=4)
print(numina_math_cot)
numina_math_cot.save_to_disk("./datasets/AI-MO/NuminaMath-CoT")