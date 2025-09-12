import os
import json
import argparse
import csv

def sort_results(results_list, dataset_order):
    dataset_index_map = {name: i for i, name in enumerate(dataset_order)}

    sorted_results = sorted(results_list,
                            key=lambda x: dataset_index_map.get(x[0], float('inf')))
    return sorted_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    args = parser.parse_args()

    input_folder = args.input_folder
    output_csv = os.path.join(input_folder, "results.csv")

    results_list = []
    header = ["Dataset", "Score"]

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                result = json.load(f)

            result_keys = list(result.get('results', {}).keys())
            if not result_keys:
                continue

            dataset_key = result_keys[0].lower()

            score = None
            if 'hellaswag' in dataset_key:
                score = result['results']['hellaswag']['acc,none']
                dataset_name = 'HellaSwag'
            elif 'truthfulqa' in dataset_key:
                score = result['results']['truthfulqa_mc2']['acc,none']
                dataset_name = 'TruthfulQA'
            elif 'winogrande' in dataset_key:
                score = result['results']['winogrande']['acc,none']
                dataset_name = 'Winogrande'
            elif 'commonsense_qa' in dataset_key:
                score = result['results']['commonsense_qa']['acc,none']
                dataset_name = 'CommonsenseQA'
            elif 'piqa' in dataset_key:
                score = result['results']['piqa']['acc,none']
                dataset_name = 'PIQA'
            elif 'openbookqa' in dataset_key:
                score = result['results']['openbookqa']['acc,none']
                dataset_name = 'OpenBookQA'
            elif 'boolq' in dataset_key:
                score = result['results']['boolq']['acc,none']
                dataset_name = 'BoolQ'
            elif 'arc_easy' in dataset_key:
                score = result['results']['arc_easy']['acc,none']
                dataset_name = 'ARC Easy'
            elif 'arc_challenge' in dataset_key:
                score = result['results']['arc_challenge']['acc,none']
                dataset_name = 'ARC Challenge'
            elif 'mmlu' in dataset_key and 'cmmlu' not in dataset_key:
                dataset_name = 'MMLU'
                group_dict = result.get('groups', {})
                acc_values = []
                for v in group_dict.values():
                    if 'acc,none' in v:
                        acc_values.append(v['acc,none'])
                if acc_values:
                    score = sum(acc_values) / len(acc_values)
            elif 'gsm8k' in dataset_key:
                score = result['results']['gsm8k']['exact_match,strict-match']
                dataset_name = 'GSM8K'
            elif 'minerva_math' in dataset_key:
                score = result['groups']['minerva_math']['exact_match,none']
                dataset_name = 'Minerva Math'
            elif 'ceval' in dataset_key:
                score = result['groups']['ceval-valid']['acc,none']
                dataset_name = 'CEval'
            elif 'cmmlu' in dataset_key:
                score = result['groups']['cmmlu']['acc,none']
                dataset_name = 'CMMLU'
            else:
                continue

            if score is not None:
                results_list.append([dataset_name, score])

    dataset_order = [
        "HellaSwag", "TruthfulQA", "Winogrande", "CommonsenseQA", "PIQA",
        "OpenBookQA", "BoolQ", "ARC Easy", "ARC Challenge", "MMLU",
        "GSM8K", "Minerva Math", "CEval", "CMMLU"
    ]

    sorted_results = sort_results(results_list, dataset_order)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header) 
        csvwriter.writerows(sorted_results)  

    print(f"Results saved to {output_csv}")