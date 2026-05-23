for i in 5
do
    python src/evaluate_accuracy.py --tokenizer_path /FlagRelease/Qwen3-4B-FlagOS-Ascend/ --task_id ${i} --max_examples 64
done
