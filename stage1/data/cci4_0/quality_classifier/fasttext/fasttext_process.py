# -*- coding: utf-8 -*-
# @Time       : 2025/1/17 10:25
# @Author     : Marverlises
# @File       : round.py
# @Description: PyCharm

import json
import logging
import random
import os
import shutil
import math
from pathlib import Path
from collections import Counter

import jieba  # Chinese text segmentation library
import pandas as pd  # Data manipulation and analysis
import fasttext  # Library for efficient text classification
import matplotlib.pyplot as plt  # Plotting library
from nltk.util import ngrams  # N-gram generation utility

# Configuration Parameters
# This dictionary centralizes all configurable settings for the script, making it easy to modify paths, training parameters, etc.
CONFIG = {
    # Directory Settings: Define base paths for data, source files, and stopwords
    "base_dir": "/share/project/baiyu/files/round16",  # Working directory for this round
    "src_dir": "/share/project/baiyu/files/round13",  # Source directory for copying training/test files
    "stopwords_dir": "./stopwords/",  # Directory for stopwords files
    "predict_data_file": "/share/project/baiyu/files/predict_data.json",  # File for prediction input

    # File Names: Define specific filenames used throughout the script
    "train_200k_pos": "train_positive_text_200k.json",  # Positive training data (200k samples)
    "train_200k_neg": "train_negative_text_200k_same_len_distribution.json",  # Negative training data (200k samples)
    "test_20k_pos": "test_positive_text_20k.json",  # Positive test data (20k samples)
    "test_20k_neg": "test_negative_text_20k_same_len_distribution.json",  # Negative test data (20k samples)
    "train_data_json": "train_data.json",  # Combined training data file
    "test_data_json": "test_data.json",  # Combined test data file
    "train_seg_file": "train_seg.txt.shuf",  # Segmented and shuffled training data
    "test_seg_file": "test_seg.txt",  # Segmented test data
    "train_seg_no_pattern": "train_seg_no_pattern.txt",  # Training data with n-grams removed
    "test_seg_no_pattern": "test_seg_no_pattern.txt",  # Test data with n-grams removed
    "model_file": "bigram_400k_model_1gram_no_pattern.bin",  # Trained model file
    "ngram_file": "ngram_3_5.json",  # File to store frequent n-grams to remove
    "predict_result_file": "predict_result.json",  # Prediction results
    "positive_data_file": "positive_data.txt",  # Positive samples from pool
    "negative_data_file": "negative_data.json",  # Negative samples from pool
    "external_test_file": "./100000_test_data_normal.txt",  # External test data
    "external_test_no_pattern": "./100000_test_data_normal_no_pattern.txt",  # External test data with n-grams removed
    "plot_image": "100000_test_data_normal_no_pattern.png",  # Output image for score distribution

    # Training Parameters: Settings for the fastText model
    "word_ngrams": 1,  # Number of word n-grams to consider (1 = unigrams)
    "epoch": 25,  # Number of training epochs
    "lr": 0.1,  # Learning rate
    "min_count": 5,  # Minimum word count threshold
    "bucket": 2000000,  # Number of buckets for hashing

    # Logging Settings: Configuration for logging output
    "log_level": logging.INFO,  # Logging verbosity level

    # Ngram Settings: Parameters for n-gram analysis
    "min_n": 3,  # Minimum n-gram size
    "max_n": 5,  # Maximum n-gram size
    "top_n": 100,  # Number of top n-grams to extract

    # Thresholds: Cutoff values for filtering
    "score_threshold": 0.98,  # Threshold for high-quality predictions
}

# Global Stopwords Set
# A set to store stopwords loaded from files, used to filter out common words during text processing
STOPWORDS_SET = set()

# Setup Logging
# Configure logging to write to both a file and the console for tracking script execution
log_file = os.path.join(CONFIG["base_dir"], f"{os.path.basename(CONFIG['base_dir'])}.log")
logging.basicConfig(
    level=CONFIG["log_level"],
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Ensure Base Directory Exists
# Create the base directory if it doesn't exist to avoid file writing errors
os.makedirs(CONFIG["base_dir"], exist_ok=True)


class Utils:
    """Utility functions for file handling, text processing, and n-gram analysis."""

    @staticmethod
    def get_all_jsonl_files(directory, suffix=".jsonl"):
        """Recursively find all files with a given suffix in a directory."""
        path = Path(directory)
        return [str(file) for file in path.rglob(f"*{suffix}")]

    @staticmethod
    def load_stopwords():
        """Load stopwords from multiple files into the global STOPWORDS_SET."""
        global STOPWORDS_SET
        stopwords_files = [
            "baidu_stopwords.txt", "cn_stopwords.txt", "hit_stopwords.txt",
            "scu_stopwords.txt", "emoji_stopwords.txt"
        ]
        # Iterate through each stopwords file and add its contents to the set
        for file in stopwords_files:
            with open(os.path.join(CONFIG["stopwords_dir"], file), "r", encoding="utf-8") as f:
                STOPWORDS_SET.update(line.strip() for line in f)
        return STOPWORDS_SET

    @staticmethod
    def segment(text, stopwords_set):
        """Segment Chinese text using jieba and remove stopwords."""
        # Replace tabs and newlines with spaces, then segment with jieba
        seg_text = jieba.cut(text.replace("\t", " ").replace("\n", " "))
        # Join segmented words and filter out stopwords
        outline = " ".join(seg_text).split()
        outline = " ".join(word for word in outline if word not in stopwords_set)
        return outline

    @staticmethod
    def calculate_ngram_frequencies(file_path, min_n=CONFIG["min_n"], max_n=CONFIG["max_n"], top_n=CONFIG["top_n"],
                                    have_label=True):
        """Calculate the frequency of n-grams in a text file."""
        ngram_counter = Counter()
        file_path = Utils.resolve_path(file_path)  # Ensure correct file path
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if have_label:  # Remove label if present
                    tokens = tokens[:-1]
                # Count n-grams for each size from min_n to max_n
                for n in range(min_n, max_n + 1):
                    ngram_counter.update(ngrams(tokens, n))
        return ngram_counter.most_common(top_n)  # Return top N most frequent n-grams

    @staticmethod
    def remove_ngram(need_remove_grams_path, ori_train_path, new_train_path, have_label=True):
        """Remove specified n-grams from a training file and save the result."""
        need_remove_grams_path = Utils.resolve_path(need_remove_grams_path)
        ori_train_path = Utils.resolve_path(ori_train_path)
        # Load n-grams to remove
        with open(need_remove_grams_path, "r") as f:
            need_remove = [line.strip() for line in f]
        logging.info(f"Total n-grams to remove: {len(need_remove)}")

        new_train_data = []
        with open(ori_train_path, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if have_label:  # Handle labeled data
                    label, tokens = tokens[-1], tokens[:-1]
                    text = " ".join(tokens)
                    for item in need_remove:
                        text = text.replace(item, "")
                    new_train_data.append(f"{' '.join(text.split())} {label}")
                else:  # Handle unlabeled data
                    text = " ".join(tokens)
                    for item in need_remove:
                        text = text.replace(item, "")
                    new_train_data.append(" ".join(text.split()))

        # Write the processed data to a new file
        with open(new_train_path, "w") as f:
            f.write("\n".join(new_train_data) + "\n")
        logging.info(f"N-gram removal completed. New file: {new_train_path}")

    @staticmethod
    def resolve_path(file_path):
        """Resolve file path, checking if it exists or adjusting to base directory."""
        if not os.path.exists(file_path):
            return os.path.join(os.path.dirname(CONFIG["base_dir"]), os.path.basename(file_path))
        return file_path

    @staticmethod
    def remove_item_ngram(item_data, need_remove_grams_path):
        """ Remove specified n-grams from a text item."""
        need_remove = []
        with open(need_remove_grams_path, 'r') as f:
            for line in f:
                need_remove.append(line.strip())
        tokens = item_data.strip().split()
        tokens_sentence = " ".join(tokens)
        for item in need_remove:
            tokens_sentence = tokens_sentence.replace(item, '')
        new_tokens = tokens_sentence.split()
        return " ".join(new_tokens)


class DataProcess:
    """Class for data preprocessing and file management."""

    @staticmethod
    def copy_files():
        """Copy training and test files from source directory to base directory."""
        files_to_copy = [
            CONFIG["train_200k_pos"], CONFIG["train_200k_neg"],
            CONFIG["test_20k_pos"], CONFIG["test_20k_neg"]
        ]
        for file_name in files_to_copy:
            src_path = os.path.join(CONFIG["src_dir"], file_name)
            dest_path = os.path.join(CONFIG["base_dir"], file_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                logging.info(f"Copied {file_name} to {CONFIG['base_dir']} successfully.")
            else:
                logging.error(f"Source file {src_path} does not exist. Copy failed.")

    @staticmethod
    def create_negative_200k_same_len_as_positive():
        # Note: This is not necessary for the current implementation, so it can be removed
        """Create a negative dataset of 200k samples matching the length distribution of positive data.

        Note: This method was intended to eliminate potential biases due to text length differences
        between positive and negative datasets. However, validation showed that length distribution
        has minimal impact on model performance in this context, so this step can be skipped if desired.
        """
        # Load pool of samples
        with open(CONFIG["pool_samples_file"], "r", encoding="utf-8") as f:
            data = [{"text": json.loads(line)["text"]} for line in f]
        logging.info(f"Total data loaded: {len(data)}")

        df = pd.DataFrame(data)
        df["length"] = df["text"].apply(len)  # Calculate text length for each sample

        # Define length distribution to match positive data
        length_distribution = [
            {"length": 100.9555, "count": 22000}, {"length": 197.2439, "count": 22000},
            {"length": 296.25115, "count": 22000}, {"length": 409.88875, "count": 22000},
            {"length": 545.4919, "count": 22000}, {"length": 707.7095, "count": 22000},
            {"length": 920.63885, "count": 22000}, {"length": 1382.10835, "count": 22000},
            {"length": 2351.31205, "count": 22000}, {"length": 4043.6189, "count": 22000}
        ]
        DataProcess.extract_same_len_save(df, length_distribution)

    @staticmethod
    def extract_same_len_save(df, length_distribution):
        """Extract and save negative samples according to the specified length distribution."""
        negative_samples = pd.DataFrame()
        available_indices = set(df.index)  # Track available samples to avoid overlap

        # Sample data for each length bin
        for bin_info in length_distribution:
            target_length, sample_count = bin_info["length"], bin_info["count"]
            min_length, max_length = DataProcess.get_length_range(target_length)

            subset = df[
                (df["length"] >= min_length) &
                (df["length"] <= max_length) &
                (df.index.isin(available_indices))
                ]
            if len(subset) < sample_count:
                raise ValueError(
                    f"Not enough samples for length {min_length}-{max_length}: {len(subset)} available, {sample_count} needed.")

            sampled = subset.sample(n=sample_count, random_state=42)  # Random sampling with fixed seed
            negative_samples = pd.concat([negative_samples, sampled], ignore_index=True)
            available_indices -= set(sampled.index)  # Remove sampled indices

        logging.info(f"Total negative samples extracted: {len(negative_samples)}")
        negative_samples_list = negative_samples.to_dict(orient="records")
        random.shuffle(negative_samples_list)  # Shuffle to randomize order

        # Split into train (200k) and test (20k) sets
        train_data = negative_samples_list[:200000]
        test_data = negative_samples_list[200000:220000]

        # Save to files
        with open(os.path.join(CONFIG["base_dir"], CONFIG["train_200k_neg"]), "w") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(os.path.join(CONFIG["base_dir"], CONFIG["test_20k_neg"]), "w") as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def get_length_range(target_length):
        """Define length range for sampling based on target length."""
        if target_length == 1382.10835:
            return math.floor(target_length * 0.4), math.ceil(target_length * 2)
        elif target_length == 2351.31205:
            return math.floor(target_length * 0.5), math.ceil(target_length * 1.5)
        elif target_length == 4043.6189:
            return math.floor(target_length * 0.5), 24608
        elif target_length == 100.9555:
            return 0, math.ceil(target_length)
        elif target_length == 197.2439:
            return 100, target_length
        return math.floor(target_length * 0.8), math.ceil(target_length * 1.2)


class Round:
    """Class to manage a training round, including data combination, segmentation, training, and evaluation."""

    def __init__(self):
        """Initialize with loaded stopwords."""
        self.stopwords = Utils.load_stopwords()

    def combine_train_test_data(self):
        """Combine positive and negative training/test data into single files."""
        # Load and label training data
        train_data = self._load_and_label_data(
            [CONFIG["train_200k_pos"], CONFIG["train_200k_neg"]],
            ["__label__1", "__label__0"]
        )
        random.shuffle(train_data)  # Shuffle to mix positive and negative samples
        self._save_data(train_data, CONFIG["train_data_json"])

        # Load and label test data
        test_data = self._load_and_label_data(
            [CONFIG["test_20k_pos"], CONFIG["test_20k_neg"]],
            ["__label__1", "__label__0"]
        )
        self._save_data(test_data, CONFIG["test_data_json"])

    def _load_and_label_data(self, files, labels):
        """Load data from files and assign labels."""
        data = []
        for file, label in zip(files, labels):
            with open(os.path.join(CONFIG["base_dir"], file), "r") as f:
                for line in f:
                    item = json.loads(line)
                    item["tag"] = label  # Assign positive or negative label
                    data.append(item)
        return data

    def _save_data(self, data, filename):
        """Save data to a JSON file."""
        with open(os.path.join(CONFIG["base_dir"], filename), "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def segment(self, file_path, save_path):
        """Segment text data and save with labels."""
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        random.shuffle(data)  # Shuffle data for randomness
        for item in data:
            item["text"] = Utils.segment(item["text"], self.stopwords) + "\t" + item["tag"]
        with open(save_path, "w") as f:
            f.write("\n".join(item["text"] for item in data) + "\n")

    def train(self, train_data, model_save_name):
        """Train a fastText model with the specified parameters."""
        logging.info(f"Training model with parameters: {CONFIG['epoch']}, {CONFIG['lr']}, {CONFIG['word_ngrams']}")
        classifier = fasttext.train_supervised(
            input=os.path.join(CONFIG["base_dir"], train_data),
            epoch=CONFIG["epoch"],
            lr=CONFIG["lr"],
            wordNgrams=CONFIG["word_ngrams"],
            verbose=2,
            minCount=CONFIG["min_count"],
            loss="hs",  # Hierarchical softmax loss
            bucket=CONFIG["bucket"]
        )
        classifier.save_model(os.path.join(CONFIG["base_dir"], model_save_name))

    def test_by_predict(self, model_path, test_name):
        """Evaluate the model by predicting on test data and calculating metrics."""
        classifier = fasttext.load_model(model_path)
        cal_dict = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}  # Confusion matrix counters
        test_name = Utils.resolve_path(test_name)

        with open(test_name, "r") as f:
            data = f.readlines()
            for item in data:
                label = "__label__1" if item.strip().endswith("__label__1") else "__label__0"
                text = item.strip().replace(label, "")  # Remove label for prediction
                pred_label = classifier.predict(text)[0][0]
                self._update_confusion_matrix(cal_dict, label, pred_label)

        self._log_evaluation_metrics(model_path, len(data), cal_dict)

    def _update_confusion_matrix(self, cal_dict, true_label, pred_label):
        """Update confusion matrix based on true and predicted labels."""
        if true_label == "__label__1":
            cal_dict["TP" if pred_label == "__label__1" else "FN"] += 1
        else:
            cal_dict["TN" if pred_label == "__label__0" else "FP"] += 1

    def _log_evaluation_metrics(self, model_path, total_samples, cal_dict):
        """Log evaluation metrics including precision, recall, and F1-score."""
        logging.info(f"==================== {model_path} ====================")
        logging.info(f"Test samples: {total_samples}")
        logging.info(f"TP: {cal_dict['TP']} | FP: {cal_dict['FP']} | TN: {cal_dict['TN']} | FN: {cal_dict['FN']}")
        precision = cal_dict["TP"] / (cal_dict["TP"] + cal_dict["FP"]) if (cal_dict["TP"] + cal_dict["FP"]) > 0 else 0
        recall = cal_dict["TP"] / (cal_dict["TP"] + cal_dict["FN"]) if (cal_dict["TP"] + cal_dict["FN"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-score: {f1:.4f}")

    def plot_pool(self, model_path, data_path, img_name, save_file):
        """Plot score distribution of predictions and save positive samples."""
        classifier = fasttext.load_model(model_path)
        with open(data_path, "r") as f:
            result = [line.strip() for line in f]

        all_scores, positive_samples = [], []
        for item in result:
            pred_label, pred_score = classifier.predict(item)
            score = pred_score[0] if pred_label[0] == "__label__1" else 1 - pred_score[0]
            all_scores.append(score)
            if pred_label[0] == "__label__1":
                positive_samples.append(f"{item}\t{score}")

        if save_file:
            with open(save_file, "w") as f:
                f.write("\n".join(positive_samples) + "\n")

        # Generate and save histogram
        plt.hist(all_scores, bins=100)
        plt.title("Score Distribution of No Pattern")
        plt.savefig(os.path.join(CONFIG["base_dir"], img_name))
        plt.close()

    def predict_by_data(self, model_path, data):
        """Predict labels and scores for a list of data items."""
        classifier = fasttext.load_model(model_path)
        stopwords_ngram_path = os.path.join(CONFIG["stopwords_dir"], "add_stopwords.json")
        result = []
        for item in data:
            seg_text = Utils.segment(item["ori_text"], self.stopwords)
            seg_text = Utils.remove_item_ngram(seg_text, stopwords_ngram_path)  # Remove additional n-grams
            pred_label, pred_score = classifier.predict(seg_text)
            score = pred_score[0] if pred_label[0] == "__label__1" else 1 - pred_score[0]
            result.append({
                "id": item["id"],
                "label": pred_label[0],
                "score": score,
                "seg_text": seg_text,
                "ori_text": item["ori_text"]
            })
        return result


def round_func():
    """Execute a full training and evaluation round."""
    # Step 1: Copy necessary files from the source directory
    DataProcess.copy_files()

    # Step 2: Initialize the Round class for training and evaluation
    round = Round()

    # Step 3: Combine positive and negative data into single files
    round.combine_train_test_data()

    # Step 4: Segment training and test data
    round.segment(os.path.join(CONFIG["base_dir"], CONFIG["train_data_json"]),
                  os.path.join(CONFIG["base_dir"], CONFIG["train_seg_file"]))
    round.segment(os.path.join(CONFIG["base_dir"], CONFIG["test_data_json"]),
                  os.path.join(CONFIG["base_dir"], CONFIG["test_seg_file"]))

    # Step 5: Remove unwanted n-grams from segmented data
    Utils.remove_ngram(
        os.path.join(CONFIG["stopwords_dir"], "add_stopwords.json"),
        os.path.join(CONFIG["base_dir"], CONFIG["train_seg_file"]),
        os.path.join(CONFIG["base_dir"], CONFIG["train_seg_no_pattern"])
    )
    Utils.remove_ngram(
        os.path.join(CONFIG["stopwords_dir"], "add_stopwords.json"),
        os.path.join(CONFIG["base_dir"], CONFIG["test_seg_file"]),
        os.path.join(CONFIG["base_dir"], CONFIG["test_seg_no_pattern"])
    )

    # Step 6: Train the model
    round.train(CONFIG["train_seg_no_pattern"], CONFIG["model_file"])

    # Step 7: Test the model on the test set
    round.test_by_predict(os.path.join(CONFIG["base_dir"], CONFIG["model_file"]),
                          os.path.join(CONFIG["base_dir"], CONFIG["test_seg_no_pattern"]))

    # Step 8: Process external test data and plot score distribution
    Utils.remove_ngram(
        os.path.join(CONFIG["stopwords_dir"], "add_stopwords.json"),
        CONFIG["external_test_file"],
        CONFIG["external_test_no_pattern"],
        have_label=False
    )
    round.plot_pool(
        os.path.join(CONFIG["base_dir"], CONFIG["model_file"]),
        CONFIG["external_test_no_pattern"],
        CONFIG["plot_image"],
        os.path.join(CONFIG["base_dir"], CONFIG["positive_data_file"])
    )

    # Step 9: Predict on external data and save results
    with open(CONFIG["predict_data_file"], "r") as f:
        all_data = [json.loads(line) for line in f]
    result = round.predict_by_data(os.path.join(CONFIG["base_dir"], CONFIG["model_file"]), all_data)
    with open(os.path.join(CONFIG["base_dir"], CONFIG["predict_result_file"]), "w") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info("Prediction completed!")


if __name__ == "__main__":
    """Entry point of the script."""
    round_func()