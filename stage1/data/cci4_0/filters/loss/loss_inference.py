from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
# torch.backends.cuda.matmul.allow_tf32=True
def initialize(model_path, gpu_id):
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=f"cuda:{gpu_id}",
        torch_dtype=torch.bfloat16,
        max_length=4096,
        trust_remote_code=True,
        _attn_implementation="sdpa"
    ).eval()

    return model, tokenizer

import random
import numpy as np
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，确保每个GPU也设置种子

def inference(input_text, gpu_id, model, tokenizer):
    input_ids = tokenizer(input_text, max_length=4096,truncation=True,return_tensors="pt",add_special_tokens=False).to(f"cuda:{gpu_id}")
    if len(input_ids['input_ids'][0]) <= 1:
        return 1e9
    batch = {"input_ids":input_ids.input_ids, "labels":input_ids.input_ids}
    with torch.no_grad():
        try:
            outputs = model.forward(**batch)
        except:
            return 1e9
    return outputs.loss.float().item()

def main():
    import argparse
    import sys
    from tdigest import TDigest

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    # parser.add_argument("--stat_path", type=str, required=True)
    args = parser.parse_args()
    set_seed(13)

#    import jsonlines
    from tqdm import tqdm
    gpu_id = args.gpu_id
    model_path = args.model_path
    model, tokenizer = initialize(model_path, gpu_id)
    data_path = args.data_path
    stat_path = f"{data_path}.stat"
    digest = TDigest()
    with open(stat_path, "w") as f:
        json.dump({"loss_95": None, "loss_99": None, "loss_90": None}, f)
    with open(data_path) as f, open(f"{data_path}.loss", "w") as wf:
        count = 0
        for l in tqdm(f):
            line = json.loads(l)
            input_text = line['text']
            res = inference(input_text , gpu_id, model, tokenizer)
            line["loss"] = res
            wf.write(json.dumps(line, ensure_ascii=False)+'\n')
            digest.update(res)
            count += 1
            print(f"Processed {count} examples")
            if count % 10000 == 0:
                with open(stat_path, "w") as fs:
                    json.dump({"loss_99": digest.percentile(99), "loss_95": digest.percentile(95), "loss_90": digest.percentile(90), "loss_80": digest.percentile(80), "loss_70": digest.percentile(70), "count": count}, fs)
        with open(stat_path, "w") as fs:
            json.dump({"loss_99": digest.percentile(99), "loss_95": digest.percentile(95), "loss_90": digest.percentile(90), "loss_80": digest.percentile(80), "loss_70": digest.percentile(70), "count": count}, fs)

if __name__ == "__main__":
    main()
