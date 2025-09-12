
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk

def process(example):
    # kto
    example["prompt"] = [
        example["messages"][0]
    ]
    example["completion"] = [
        example["messages"][1]
    ]

    example["label"] = True
    return example

numina_math_cot = load_from_disk("/root/code/small-doge/datasets/AI-MO/NuminaMath-CoT")
print(numina_math_cot)
numina_math_cot = numina_math_cot.map(process, num_proc=4).select_columns(["prompt", "completion", "label"])
print(numina_math_cot)
print(numina_math_cot["train"][0])
numina_math_cot.save_to_disk("./datasets/AI-MO/NuminaMath-CoT-preference")