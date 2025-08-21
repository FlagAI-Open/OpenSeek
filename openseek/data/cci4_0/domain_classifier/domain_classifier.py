# -*- coding: utf-8 -*-
# @Time       : 2025/3/14 9:06
# @Author     : Marverlises
# @File       : domain_classifier.py
# @Description: PyCharm

import torch
import logging
import argparse

from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    """
    This class is used to load the domain classifier model and predict the domain of the given text data.
    """

    def __init__(self, config):
        """ Initialize the model """
        super(CustomModel, self).__init__()
        self.custom_tokenizer = None
        self.custom_model = None
        self.custom_config = None

        # Load the base model from  local path
        if deberta_base_model_path:
            config["base_model"] = deberta_base_model_path

        self.model = AutoModel.from_pretrained(config["base_model"])
        self.model.to(device)
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        """ Forward pass """
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

    def predict_result(self, request_data):
        """ request_data: list of str containing text data """
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
        """ Load domain classifier model """
        logger.info(f"Loading Domain Classifier Model...{model_path}")
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = CustomModel.from_pretrained(model_path)
        model.eval()
        model.to(device)
        self.custom_tokenizer = tokenizer
        self.custom_model = model
        self.custom_config = config


if __name__ == '__main__':
    # You can run this script to test the domain classifier model with default arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--deberta-model-path', type=str, default='nvidia/multilingual-domain-classifier',
                        help="Path to the domain classifier model")
    parser.add_argument('--deberta-base-model-path', type=str, default='',
                        help="Your local model path to the 'https://huggingface.co/microsoft/mdeberta-v3-base' model")

    args = parser.parse_args()
    deberta_base_model_path = args.deberta_base_model_path

    domain_classifier = CustomModel.from_pretrained(args.deberta_model_path)
    domain_classifier.load_custom_model(args.deberta_model_path)

    data = ["Los deportes son un dominio popular", "La política es un dominio popular"]
    result = domain_classifier.predict_result(data)
    print(result)
