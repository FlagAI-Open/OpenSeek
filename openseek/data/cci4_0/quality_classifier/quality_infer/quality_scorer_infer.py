# -*- coding: utf-8 -*-
# @Time       : 2025/2/19 9:53
# @Author     : Marverlises
# @File       : roberta.py
# @Description: PyCharm

import os
import time
import torch
import jsonlines
import argparse
import logging
import fasttext
import tqdm
import utils

from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from utils import Utils


def setup_logging(log_filename='inference.log'):
    """
    设置日志记录，将日志输出到指定文件并打印到控制台。

    :param log_filename: 日志文件的名称，默认为'app.log'
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, log_filename)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        handlers=[logging.FileHandler(log_filename),  # 输出到文件
                                  logging.StreamHandler()])  # 输出到控制台
    return logging.getLogger(__name__)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.custom_tokenizer = None
        self.custom_model = None
        self.custom_config = None
        config["base_model"] = deberta_base_model_path
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.model.to(device)
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

    def predict_result(self, request_data):
        """ request_data: list of dictionaries containing text data """
        if isinstance(request_data, str):
            request_data = [request_data]

        input_token = self.custom_tokenizer(request_data, return_tensors="pt", padding="longest", truncation=True).to(
            device)
        with torch.no_grad():
            outputs = self.custom_model(input_token["input_ids"], input_token["attention_mask"])
        # Predict and display results
        predicted_classes = torch.argmax(outputs, dim=1)
        predicted_domains = [self.custom_config.id2label[class_idx.item()] for class_idx in
                             predicted_classes.cpu().numpy()]
        return predicted_domains

    def load_custom_model(self, model_path):
        logger.info(f"Loading Deberta Custom Model...{model_path}")
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = CustomModel.from_pretrained(model_path)
        model.eval()
        model.to(device)
        self.custom_tokenizer = tokenizer
        self.custom_model = model
        self.custom_config = config


class FastTextModel:
    def __init__(self):
        self.model = None

    def load_fasttext_model(self, model_path):
        logger.info(f"Loading FastText model...{model_path}")
        self.model = fasttext.load_model(model_path)

    def predict_result(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        scores = []
        for text in texts:
            seg_text = Utils.segment(text)  # Segment text using Utils
            seg_text = Utils.remove_item_ngram(seg_text)
            result = self.model.predict(seg_text)
            if result[0][0] == '__label__1':
                score = result[1][0]
            else:
                score = 1 - result[1][0]
            scores.append(score)
        return scores


class RobertaModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = 2048

    def load_roberta_model(self, model_path):
        logger.info(f"Loading Roberta model...{model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=False,
                                                                   ignore_mismatched_sizes=False)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=None, trust_remote_code=False)
        self.model = model
        self.tokenizer = tokenizer

    def predict_result(self, text):
        if isinstance(text, str):
            text = [text]
        result = self.tokenizer(text, padding=True, max_length=self.max_length, truncation=True,
                                return_tensors="pt").to(device)

        with torch.no_grad():
            model_out = self.model(**result)
        score = [float(item[0]) for item in model_out.logits.tolist()]
        return score


class FineWebModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_fine_web_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)

    def predict_result(self, text):
        if isinstance(text, str):
            text = [text]
        scores = []
        for text_item in text:
            inputs = self.tokenizer(text_item, return_tensors="pt", padding="longest", truncation=True).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # logits = outputs.logits.squeeze(-1).float().detach().numpy()
            logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()

            score = logits.item()
            scores.append(score)
        return scores


def get_output_file_writer_line(output_file_path):
    """
    This function is used to get the output file writer line. If the output file already exists, get the number of lines in the file. Otherwise, return 0.
    :param output_file_path:    Path to the output file
    :return:                    Number of lines in the output file
    """
    if os.path.exists(output_file_path):
        line_num = 0
        with jsonlines.open(output_file_path) as reader:
            try:
                for _ in reader:
                    line_num += 1
            except Exception as e:
                return line_num
        return line_num
    else:
        return 0


def process_file(file_path, output_file_path, models,
                 batch_size=256):
    """
    Process a file and write the output to a new file.
    :param file_path:           Path to the input file
    :param output_file_path:    Path to the output file
    :param models:              Dictionary containing the models to use
    :param batch_size:          Batch size for processing the file
    :return:
    """
    lines = 0
    batch = []
    total_start_time = time.time()
    start_time = time.time()
    # get the number of lines in the output file
    # already_processed_lines = get_output_file_writer_line(output_file_path)
    # logger.info(f"Continue from lines {already_processed_lines}...")

    try:
        logger.info(f"Processing {file_path}...")
        # with jsonlines.open(file_path) as reader, jsonlines.open(output_file_path, mode='a') as writer:
        with jsonlines.open(file_path) as reader, jsonlines.open(output_file_path, mode='w') as writer:
            for line in reader:
                try:
                    # # Skip already processed lines
                    # if lines < already_processed_lines:
                    #     lines += 1
                    #     continue

                    text = line["text"]
                    batch.append((line, text))

                    if len(batch) >= batch_size:
                        process_batch(batch, writer, models)
                        batch = []

                    if lines % 500 == 0 and lines != 0:
                        logger.info(
                            f"Processed {lines} lines. Average time cost: {500 / (time.time() - start_time):.2f} lines/s.")
                        start_time = time.time()

                    lines += 1
                except Exception as e:
                    logger.error(f"Error processing line {lines}: {e}, output file: {output_file_path}")
                    continue
            # Process remaining batch
            if batch:
                process_batch(batch, writer, models)

        logger.info(f"Processed {lines} lines.")
        logger.info(f"Average time cost: {lines / (time.time() - total_start_time):.2f} lines/s.")
        logger.info(f"Output file saved to {output_file_path}")
        return False
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}, Stopping at line {lines}")
        raise e


def process_batch(batch, writer, models):
    """
    Process a batch of data and write the output to a file.
    :param batch:       List of tuples containing the data —— (line, text)
    :param writer:      JSONL writer object
    :param models:      Dictionary containing the models to use
    :return:            None
    """
    texts = [item[1] for item in batch]
    result = {}

    for model_name, model in models.items():
        result[model_name] = model.predict_result(texts)

    for i, (line, _) in enumerate(batch):
        line["quality"] = {}

        if 'deberta' in result.keys():
            line["domain"] = result['deberta'][i]

        if 'fasttext' in result.keys():
            line['quality']['fasttext_400k_train_1gram'] = result['fasttext'][i]
            line['quality']['fasttext_400k_train_1gram_normal'] = max(0, min(5, 5 * result['fasttext'][i]))

        if 'roberta-qwen' in result.keys():
            line['quality']['roberta_qwen'] = result['roberta-qwen'][i]
            line['quality']['roberta_qwen_normal'] = max(0, min(5, result['roberta-qwen'][i]))

        if 'roberta-deepseek' in result.keys():
            line['quality']['roberta_deepseek'] = result['roberta-deepseek'][i]
            line['quality']['roberta_deepseek_normal'] = max(0, min(5, result['roberta-deepseek'][i]))

        if 'fine-web' in result.keys():
            line['quality']['fine_web'] = result['fine-web'][i]
            line['quality']['fine_web_normal'] = max(0, min(5, result['fine-web'][i]))

        writer.write(line)


def get_models(need_models_list):
    models = {}
    if 'roberta-deepseek' in need_models_list:
        RobertaModel1 = RobertaModel()
        RobertaModel1.load_roberta_model(args.roberta_model_path1)
        models['roberta-deepseek'] = RobertaModel1

    if 'roberta-qwen' in need_models_list:
        RobertaModel2 = RobertaModel()
        RobertaModel2.load_roberta_model(args.roberta_model_path2)
        models['roberta-qwen'] = RobertaModel2

    if 'fasttext' in need_models_list:
        Utils.get_stopwords()
        FasttextModel = FastTextModel()
        FasttextModel.load_fasttext_model(args.fasttext_model_path)
        models['fasttext'] = FasttextModel

    if 'fine-web' in need_models_list:
        FinewebModel = FineWebModel()
        FinewebModel.load_fine_web_model(args.fineweb_model_path)
        models['fine-web'] = FinewebModel

    if 'deberta' in need_models_list:
        LingualModel = CustomModel.from_pretrained(args.deberta_model_path)
        LingualModel.load_custom_model(args.deberta_model_path)
        models['deberta'] = LingualModel

    return models


def run_inference(args):
    os.makedirs(args.output_file_dir, exist_ok=True)

    try:
        models = get_models(args.model)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
    failed_files = []

    batch_size = args.batch_size
    logger.info(f"Batch size: {batch_size}")
    for input_file in args.input_file_path:
        output_file_path = os.path.join(args.output_file_dir,
                                        f"{os.path.basename(input_file).split('.')[0]}_output.jsonl")
        fail_flag = process_file(input_file, output_file_path, models, batch_size=batch_size)
        if fail_flag:
            failed_files.append(input_file)

    # Log failed files
    if failed_files:
        logger.info(f"Failed to process the following files: {', '.join(failed_files)}")
        # save failed files to a text file
        with open("failed_files.txt", "w") as f:
            f.write("\n".join(failed_files))
    else:
        logger.info("All files processed successfully.")


"""
示例脚本：
python data_infer.py     --roberta-model-path1 "/path/to/roberta_model (Scorer 1)" \
                         --roberta-model-path2 "/path/to/roberta_model (Scorer 2)" \
                         --deberta-model-path "/path/to/deberta_model (Domain Classifier)" \
                         --fasttext-model-path "/path/to/fasttext_model (Scorer 3)" \
                         --input-file-path "/path/to/input1.jsonl" "/path/to/input2.jsonl" \
                         --output-file-dir "/path/to/output" \
                         --batch-size 16 \
                         --model "roberta-qwen" "roberta-deepseek" "deberta" "fasttext" \
                         --cuda-device 1
                         --stopwords-path "/path/to/stopwords"
                         
"roberta-qwen"          :   Scorer 1
"roberta-deepseek"      :   Scorer 2
"deberta"               :   Domain Classifier
"fasttext"              :   Scorer 3
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--roberta-model-path1', type=str,
                        default='/baai/baai-sailling/ldwang/quality_models/Roberta_deepseek',
                        help="Path to the Roberta1 model")
    parser.add_argument('--roberta-model-path2', type=str,
                        default='/baai/baai-sailling/ldwang/quality_models/Roberta_qwen',
                        help="Path to the Roberta2 model")
    parser.add_argument('--deberta-model-path', type=str, default='/baai/baai-sailling/ldwang/quality_models/deberta',
                        help="Path to the deberta model")
    parser.add_argument('--fasttext-model-path', type=str,
                        default='/baai/baai-sailling/ldwang/quality_models/fasttext-score/bigram_400k_model_1gram_no_pattern.bin',
                        help="Path to the FastText model")
    parser.add_argument('--fineweb-model-path', type=str,
                        default='/baai/baai-sailling/ldwang/quality_models/fineweb-edu-classifier',
                        help="Path to the FineWeb Roberta model")
    # set stopwords path
    parser.add_argument('--stopwords-path', type=str,
                        default='/baai/baai-sailling/ldwang/stopwords/',
                        help="Path to the stopwords")

    parser.add_argument('--input-file-path', type=str, nargs='+', help="Path to the input JSONL files",
                        default=['./temp_save_path/test_speed.jsonl'])
    parser.add_argument('--output-file-dir', type=str, help="Directory to save the output JSONL files",
                        default='./temp_save_path')
    parser.add_argument('--logs-dir', type=str, help="Directory to save the output JSONL files",
                        default='./logs')
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for processing files")
    # choose the model to use
    parser.add_argument('--model', type=str, nargs='+',
                        default=['roberta-qwen', 'roberta-deepseek', 'deberta', 'fasttext', 'fine-web'],
                        help="Choose the model to use")

    # choose the CUDA device
    parser.add_argument('--cuda-device', type=int, default=0, help="Choose the CUDA device")
    # add info
    parser.add_argument('--info', type=str, default='test',
                        help="Add info to the log file to identify the running machine")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    # get the base model path for deberta
    deberta_base_model_path = os.path.join(os.path.split(args.deberta_model_path)[0], 'deberta_base')
    # set the basedir for utils and set add_stopwords
    utils.basedir = args.stopwords_path
    need_remove_grams_path = os.path.join(args.stopwords_path, 'add_stopwords.json')
    with open(need_remove_grams_path, 'r') as f:
        for line in f:
            utils.need_remove_ngrams.append(line.strip())

    log_dir = args.logs_dir
    logger = setup_logging(f"{args.info}_device_{args.cuda_device}.log")
    # log the arguments
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # run inference
    run_inference(args)
