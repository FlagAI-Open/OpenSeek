import json

import task8_best_realtime_ca2a7b as task8_run


CONFIGS = [
    "pytorch_v1_r3_probe15k_then_r3_repair_semantic40",
    "pytorch_v1_r3_probe15k_then_r3_duibi_targeted166",
]


def main() -> None:
    data = json.loads(task8_run.DATA_PATH.read_text())
    samples = data["test_samples"][:166]
    examples = data["examples"]
    task_description = task8_run._default_task_description(data)
    for config in CONFIGS:
        print(f"=== RUN {config} ===", flush=True)
        task8_run.run_inference(config, samples, examples, task_description=task_description)
        print(f"=== DONE {config} ===", flush=True)


if __name__ == "__main__":
    main()
