import argparse
import json
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from pathlib import Path
from concurrent.futures import TimeoutError

from grader import *

from parser import *


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def evaluate(data_name, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
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
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=8) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
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

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=4))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    output_path = file_path.replace(".jsonl", f"_result.json")
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=4)
    return samples, result_json


def merge_all(eval_path):
    final_res = dict()
    final_res['dataset_acc'] = dict()
    for dir in os.listdir(eval_path):
        if os.path.isdir(os.path.join(eval_path, dir)):
            data_name = os.path.basename(dir)
            for file in os.listdir(os.path.join(eval_path, dir)):
                if file.endswith("_result.json"):
                    with open(os.path.join(eval_path, dir, file), "r") as f:
                        data = json.load(f)
                    final_res['dataset_acc'][data_name] = data['acc']
    
    final_res['final_acc'] = np.mean(list(final_res['dataset_acc'].values()))
    with open(os.path.join(eval_path, "final_result.json"), "w") as f:
        json.dump(final_res, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, default="./eval_example/")
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
                    evaluate(data_name=data_name, file_path=file_path)
    
    merge_all(args.eval_path)
