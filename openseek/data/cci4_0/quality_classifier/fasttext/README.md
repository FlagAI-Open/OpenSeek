# FastText Quality Classifier

A text quality classification tool based on FastText for evaluating and filtering text data. This tool supports training models to classify text as high-quality (positive) or low-quality (negative), and provides utilities for text preprocessing, model training, and prediction.

> Notes: We add some stopwords based on the top frequency of the predicted high-quality data that might influence the model performance during our iteration. If you want to use the original stopwords, please **remove** the **add_stopwords.json** file and **emoji_stopwords.txt** file in fasttext_process.py during your process.

## Features

- Text preprocessing with stopword removal and word segmentation
- N-gram feature extraction and filtering
- FastText model training with customizable parameters
- Model evaluation with precision, recall, and F1 metrics
- Batch prediction capabilities
- Support for both Chinese and English text

## Requirements

```
fasttext_wheel==0.9.2
jieba==0.42.1
matplotlib==3.10.1
nltk==3.8
pandas==1.5.3
```

## Directory Structure

```
./
├── fasttext_predict.py      # Script for making predictions with trained models
├── fasttext_process.py      # Main module with preprocessing, training, and evaluation
├── requirements.txt         # Required Python packages
└── stopwords/               # Stopword files for text preprocessing
    ├── add_stopwords.json   # Additional custom stopwords
    ├── baidu_stopwords.txt  # Baidu's stopword list
    ├── cn_stopwords.txt     # Chinese stopwords
    ├── emoji_stopwords.txt  # Emoji characters to filter
    ├── hit_stopwords.txt    # Harbin Institute of Technology stopwords
    └── scu_stopwords.txt    # Sichuan University stopwords
```

## Usage

### 1. Training a New Model

```python
from fasttext_process import Round

# Initialize
round = Round()

# Combine and preprocess training/test data
round.combine_train_test_data()

# Segment text for training
round.segment(file_path="train_data.json", save_path="train_seg.txt")

# Train the model
round.train(train_data="train_seg.txt", model_save_name="my_model.bin")

# Evaluate on test data
round.test_by_predict(model_path="my_model.bin", test_name="test_seg.txt")
```

### 2. Making Predictions

```python
from fasttext_process import Round
import os

# Initialize
round = Round()

# Method 1: Predict from a list of texts
texts = ["This is a high-quality text", "This is a low-quality text"]
results = round.predict_by_data(
    model_path="path/to/model.bin",
    data=texts
)

# Method 2: Predict from a file
round.test_by_predict(
    model_path="path/to/model.bin",
    test_name="path/to/test_file.txt"
)
```

## Configuration

The tool uses a centralized configuration dictionary in `fasttext_process.py` that can be modified to adjust:

- File paths for training and test data
- Training parameters (learning rate, epochs, etc.)
- N-gram processing settings
- Thresholds for quality classification

## Model Training Parameters

- Word n-grams: 1-2 (default)
- Learning rate: 0.1 (default)
- Epochs: 25 (default)
- Vector dimension: 100 (default)

