rm ./data_processed/*
rm ./outputs/*

#python LongContext-ICL-Annotation/src/build_sadness_representative_cache.py  --tokenizer_path /FlagRelease/Qwen3-4B-FlagOS-Ascend/
python LongContext-ICL-Annotation/src/build_concat_cache.py  --tokenizer_path /FlagRelease/Qwen3-4B-FlagOS-Ascend/
#python LongContext-ICL-Annotation/src/tag_task2.py
#python LongContext-ICL-Annotation/src/tag_task7_examples.py --num_examples 2000 

for t in 4  
do
	python3.11 LongContext-ICL-Annotation/src/main.py --tokenizer_path /FlagRelease/Qwen3-4B-FlagOS-Ascend/ --task_id $t  --threads 32
done

python LongContext-ICL-Annotation/src/merge_task8_results.py

mv ./outputs/openseek-8-v2.jsonl ./outputs/openseek-8-v1.jsonl
