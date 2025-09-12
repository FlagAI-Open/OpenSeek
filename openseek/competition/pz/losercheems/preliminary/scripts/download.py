from ..processor.pt_datasets_process import mix_datasets_by_ratio as mix_pt_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# export HF_ENDPOINT=https://hf-mirror.com
# export XDG_CACHE_HOME=cache



if __name__ == "__main__":
    
    datasets_and_ratios = [

        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-high-actual-actual-high_part_142_text_document": 0.02242},
        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-high-synthetic-distill-high_part_76_text_document": 0.00687},
        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-high-synthetic-diverse_qa_pairs-high_part_244_text_document": 0.014466},    
        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-high-synthetic-extract_knowledge-high_part_498_text_document": 0.008715},
        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-high-synthetic-knowledge_list-high_part_86_text_document": 0.006747},
        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-high-synthetic-wrap_medium-high_part_47_text_document": 0.017773},
        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-low-synthetic-wrap_medium-high_part_43_text_document": 0.010979},
        {"JingzeShi/OpenSeek-Pretrain-100B:Nemotron-CC-medium-actual-actual-high_part_92_text_document": 0.027354},


        {"JingzeShi/OpenSeek-Pretrain-100B:arxiv_007_00000_text_document": 0.006414},
        {"JingzeShi/OpenSeek-Pretrain-100B:books_016_00007_text_document": 0.004696},
        {"JingzeShi/OpenSeek-Pretrain-100B:code-high_part_13_text_document": 0.031179},


        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis2_CC-high_23_text_document": 0.022553},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis2_OpenSource-high_1_text_document": 0.007462},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis2_arxiv-high_2_text_document": 0.250676},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis2_code-high_4_text_document": 0.020445},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis2_math-high_12_text_document": 0.033201},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis2_wiki-high_5_text_document": 0.020201},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis_CC-high_74_text_document": 0.006064},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis_OpenSource-high_4_text_document": 0.018568},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis_arxiv-high_2_text_document": 0.221066},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis_code-high_13_text_document": 0.013631},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis_math-high_11_text_document": 0.017917},
        {"JingzeShi/OpenSeek-Pretrain-100B:cot_synthesis_wiki-high_4_text_document": 0.017917},


        {"JingzeShi/OpenSeek-Pretrain-100B:math-high_part_04_text_document": 0.051416},


        {"JingzeShi/OpenSeek-Pretrain-100B:pes2o-full-train_train-0041-of-0136_text_document": 0.00687},
        {"JingzeShi/OpenSeek-Pretrain-100B:pes2o-full-train_train-0125-of-0136_text_document": 0.007387},
        {"JingzeShi/OpenSeek-Pretrain-100B:pes2o-full-val_valid-0034-of-0060_text_document": 0.000143},
        {"JingzeShi/OpenSeek-Pretrain-100B:pes2o_pubmedcentral_3_text_document": 0.061982},
        {"JingzeShi/OpenSeek-Pretrain-100B:stack_018_00000_text_document": 0.004229},
        {"JingzeShi/OpenSeek-Pretrain-100B:wiki_012_00000_text_document": 0.004202},


        {"JingzeShi/OpenSeek-Pretrain-100B:zh_cc-high-loss0_part_28_text_document": 0.031672},
        {"JingzeShi/OpenSeek-Pretrain-100B:zh_cc-high-loss1_part_59_text_document": 0.029371},


    ]

    def calculate_total_ratio(datasets_and_ratios):
        return sum(item for item in datasets_and_ratios.values())

    total_ratio = sum(calculate_total_ratio(dataset) for dataset in datasets_and_ratios)
    print(f"Total ratio: {total_ratio}")

    total_sample_size = 7_500_000
    dataset_text_field = "text"
    max_length = 4096
    packing = True
    dataset_num_proc = 4
    cache_dir = "./cache"
    seed = 233
    model_name_or_path = "BAAI/OpenSeek-Small-v1"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.to(torch.bfloat16)
    print(model)
    model.save_pretrained(f"./models/OpenSeek_Small_v1")
    tokenizer.save_pretrained(f"./models/OpenSeek_Small_v1")

    dataset = mix_pt_datasets(
            datasets_and_ratios=datasets_and_ratios,
            total_sample_size=total_sample_size,
            dataset_text_field=dataset_text_field,
            processing_class=tokenizer,
            max_length=max_length,
            packing=packing,
            formatting_func=None,
            dataset_num_proc=dataset_num_proc,
            seed=seed,
            # cache_dir=cache_dir,
        )
    dataset = dataset.select_columns(["input_ids"])
    print(dataset)

    dataset.save_to_disk("./datasets/OpenSeek-Pretrain-30B", num_proc=8)