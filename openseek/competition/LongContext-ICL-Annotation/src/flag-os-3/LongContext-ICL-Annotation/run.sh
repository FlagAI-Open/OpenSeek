for t in 8 8 8 8 
do
	python src/main.py --tokenizer_path /FlagRelease/Qwen3-4B-FlagOS-Ascend/ --task_id $t  --threads 16
done
