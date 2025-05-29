# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file-path", type=str, default="", help="file path", required=True
    )
    parser.add_argument(
        "--scorer-model-path",
        type=str,
        default="",
        help="scorer model path",
        required=True,
    )
    parser.add_argument(
        "--score-thres", type=float, default=3.0, help="score thres", required=False
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="max length", required=False
    )
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    model_dir = args.scorer_model_path
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=False,
        ignore_mismatched_sizes=False,
    )
    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        token=None,
        trust_remote_code=False,
    )

    max_length = args.max_length

    y_true = []
    y_pred = []

    import jsonlines

    input_file = args.input_file_path
    lines = 0
    with jsonlines.open(input_file) as reader:
        for line in reader:
            lines += 1
            if lines % 500 == 0:
                print(f"Processed {lines} lines.", flush=True)
            sentecnce = line["text"]
            score = line["score"]
            if score < 3:
                y_true.append(1)
            else:
                y_true.append(0)

            result = tokenizer(
                [sentecnce],
                padding=False,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            for key in result:
                result[key] = torch.tensor(result[key])

            model_out = model(**result)
            pred_score = float(model_out.logits.tolist()[0][0])
            if pred_score < args.score_thres:
                y_pred.append(1)
            else:
                y_pred.append(0)
            del model_out, pred_score

    from sklearn.metrics import confusion_matrix, classification_report

    matrix = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix: \n{matrix}")
    report = classification_report(y_true, y_pred)
    print("Validation Report:\n" + report, flush=True)
    TP, FP, FN, TN = matrix[0][0], matrix[1][0], matrix[0][1], matrix[1][1]
    samples = TP + FP + FN + TN
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1_score = 2 * (precision * recall) / (recall + precision)
    print(f"Accuracy: {(TP + TN) / samples}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1_score:.4f}")
