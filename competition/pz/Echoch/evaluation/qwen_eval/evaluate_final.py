import argparse
import json
import os
import numpy as np
import math
from tqdm import tqdm
from pebble import ProcessPool
from pathlib import Path
from concurrent.futures import TimeoutError
from typing import Any, Iterable, Union, List, Dict

from grader import *

from parser import *

from math_verify import parse, verify


def calculate_pass_at_k(
    results: List[Dict[str, int]], 
    k: int
) -> float:
    if not results:
        return 0.0

    problem_probabilities = []
    for res in results:
        n = res.get('n')
        c = res.get('c')
        if n is None or c is None:
            raise ValueError("Each result dictionary must contain 'n' and 'c' keys.")
        if n < c:
            raise ValueError(f"The number of correct samples (c={c}) cannot be greater than the total number of samples (n={n}).")
        if n < k:
            continue
        if n - c < k:
            prob = 1.0
        else:
            numerator = math.comb(n - c, k)
            denominator = math.comb(n, k)
            if denominator == 0:
                prob = 1.0
            else:
                prob = 1.0 - (numerator / denominator)
        problem_probabilities.append(prob)

    if not problem_probabilities:
        return 0.0
        
    return np.mean(problem_probabilities)


def math_equal_process(param):
    if param[1] == "qwen":
        return math_equal(param[-2], param[-1])
    else:
        math_verify_parsed, ground_truth = param[-2], [param[-1]]

        if len(math_verify_parsed) < 2:
            return False
        
        # We perform a quick string match first
        if math_verify_parsed[1] in ground_truth:
            return True
        
        # We now fallback to semantic verification
        for gt in ground_truth:
            try:
                print(parse(f"\\boxed{{{gt}}}", parsing_timeout=5))
                if verify(
                    parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                    math_verify_parsed,
                    timeout_seconds=5,
                ):
                    return True
            except Exception:
                continue
        
        # Very unlikely to be correct after the above matches
        return False


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def get_preds(result, data_name):
    report = None
    if '</think>' in result:
        result = result.split('</think>')[1]
    prediction = extract_answer(result, data_name)

    # prediction = strip_string(prediction, skip_unit=data_name == "carp_en")
    prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)

    return prediction,report 

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def evaluate(data_name, k_list: List[int]=None, samples: list=None, file_path: str=None, max_num_samples=None, eval_method="qwen", re_eval=False,execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
        if eval_method == "qwen":
            results = [get_preds(code, data_name) for code in sample['code']]
            print(results)
            # put results back to examples
            preds = [item[0] for item in results]
            reports = [item[1] for item in results]
            for j in range(len(preds)):
                if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ]:
                    preds[j] = choice_answer_clean(sample['code'][j])
                elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                    # remove any non-choice char
                    preds[j] = "".join(
                        [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                    )
                sample.update({ "pred": preds, "report": reports})
        elif eval_method == "math_verify":
            sample['pred'] = [parse(pred,parsing_timeout=5) for pred in sample['code']]
    params = [(idx,eval_method, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]
    print(params)

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=8) as pool:
        future = pool.map(math_equal_process, params, timeout=5)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    pass_at_k_scores = {}
    if score_mat:
        pass_at_k_results = []
        for s in score_mat:
            n = len(s)
            c = sum(s)
            if n > 0:
                pass_at_k_results.append({'n': n, 'c': c})
        print("Pass@k results:", pass_at_k_results)
        k_values = k_list if k_list else [1, 4, 8]
        max_preds = max(len(s) for s in score_mat) if score_mat else 0

        for k in k_values:
            if max_preds >= k:
                score = calculate_pass_at_k(pass_at_k_results, k)
                pass_at_k_scores[f'pass@{k}'] = np.round(score * 100, decimals=5)
            else:
                pass_at_k_scores[f'pass@{k}'] = "Not enough samples (n < k)"

    max_len = max([len(s) for s in score_mat]) if score_mat else 0

    with open(file_path.replace(".jsonl", f"{eval_method}_result.jsonl") if not re_eval else file_path, 'w') as f:
        for sample in samples:
            if eval_method == "math_verify":
                sample['pred'] = [pred[1] if len(pred) == 2 else '' for pred in sample['pred']]
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    if max_len > 0:
        for i, s in enumerate(score_mat):
            if len(s) < max_len:
                score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    if max_len > 0 and score_mat:
        if len(score_mat[0]) > 1:
            row_means= np.array(score_mat).mean(axis=1)
            mean_score = list(np.round(row_means * 100, decimals=5))
            mean_score = np.round(np.mean(mean_score), decimals=5)
        else:
            col_means= np.array(score_mat).mean(axis=0)
            mean_score = list(np.round(col_means * 100, decimals=5))
            mean_score = np.round(np.mean(mean_score), decimals=5)
    else:
        mean_score = 0
    

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'] or not s['pred'][-1]]),
        "acc": mean_score,
        "pass@k": pass_at_k_scores
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            if sample['score']:  # 确保score列表不为空
                type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    # output_path = file_path.replace(".jsonl", f"{eval_method}_eval_result.json")
    # with open(output_path, "w") as f:
    #     json.dump(result_json, f, indent=4)
    return samples, result_json


def merge_all(eval_path):
    final_res = dict()
    final_res['dataset_acc'] = dict()
    for dir in os.listdir(eval_path):
        if os.path.isdir(os.path.join(eval_path, dir)):
            data_name = os.path.basename(dir)
            for file in os.listdir(os.path.join(eval_path, dir)):
                if file.endswith("_metrics.json"):
                    with open(os.path.join(eval_path, dir, file), "r") as f:
                        data = json.load(f)
                    final_res['dataset_acc'][data_name] = data['acc']
    
    final_res['final_acc'] = np.mean(list(final_res['dataset_acc'].values()))
    with open(os.path.join(eval_path, "final_result.json"), "w") as f:
        json.dump(final_res, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, default="./eval_example/")
    parser.add_argument(
        "--k_list",
        nargs='+', 
        type=int,
        default=[1, 4, 8], 
    )
    parser.add_argument("--eval_method", type=str, default="qwen", choices=["qwen", "math_verify"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    for dir in os.listdir(args.eval_path):
        if os.path.isdir(os.path.join(args.eval_path, dir)):
            data_name = os.path.basename(dir)
            for file in os.listdir(os.path.join(args.eval_path, dir)):
                if file.endswith(".jsonl"):
                    file_path = os.path.join(args.eval_path, dir, file)
                    print(f"Evaluating {data_name}")
                    evaluate(data_name=data_name, k_list=args.k_list, file_path=file_path, eval_method=args.eval_method,re_eval=True)
    
    merge_all(args.eval_path)
