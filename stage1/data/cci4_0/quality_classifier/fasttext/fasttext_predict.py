# -*- coding: utf-8 -*-
# @Time       : 2025/3/19 15:11
# @Author     : Marverlises
# @File       : fasttext_predict.py
# @Description: PyCharm

import os
from fasttext_process import Round

if __name__ == '__main__':
    base_dir = 'The path of the model file'

    # Method 1: Predict by list of text data
    round = Round()
    # Open your data file and read it into all_data as ['text1', 'text2', ...]
    all_data = []
    result = round.predict_by_data(model_path=os.path.join(base_dir, 'bigram_400k_model_1gram_no_pattern.bin'),
                                   data=all_data)
    # Method 2: Predict by seg text file, YOUR_TEST_FILE such as ['word1 word2 label', 'word3 word4 label', ...]
    round.test_by_predict(model_path=os.path.join(base_dir, 'bigram_400k_model_1gram_no_pattern.bin'),
                          test_name='YOUR_TEST_FILE_PATH')