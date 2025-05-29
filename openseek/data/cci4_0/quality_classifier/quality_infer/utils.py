# -*- coding: utf-8 -*-
# @Time       : 2025/2/19 11:41
# @Author     : Marverlises
# @File       : utils.py
# @Description: PyCharm
import logging
import os
import jieba

from pathlib import Path

stopwords_set = set()
need_remove_ngrams = []
basedir = None


class Utils:
    @staticmethod
    def get_all_jsonl_files(directory, suffix='.jsonl'):
        """
        获取指定目录及其所有子目录下的所有以 某个后缀结尾的文件。

        :param directory: 要搜索的根目录路径。
        :param suffix: 文件后缀。默认为 '.jsonl'。
        :return: 包含所有 .jsonl 文件路径的列表。
        """
        path = Path(directory)
        return [str(file) for file in path.rglob(f'*{suffix}')]

    @staticmethod
    def get_stopwords():
        """
        获取停用词
        :return:
        """
        global stopwords_set
        assert basedir is not None, "stopwords_path is not set"
        # 停用词文件
        with open(os.path.join(basedir, 'baidu_stopwords.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                stopwords_set.add(line.strip())
        with open(os.path.join(basedir, 'cn_stopwords.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                stopwords_set.add(line.strip())
        with open(os.path.join(basedir, 'hit_stopwords.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                stopwords_set.add(line.strip())
        with open(os.path.join(basedir, 'scu_stopwords.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                stopwords_set.add(line.strip())
        with open(os.path.join(basedir, 'emoji_stopwords.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                stopwords_set.add(line.strip())

        return stopwords_set

    @staticmethod
    def segment(text):
        """
        分词
        :param text:            待分词的文本
        :param stopwords_set:   停用词
        :return:
        """
        # 结巴分词
        seg_text = jieba.cut(text.replace("\t", " ").replace("\n", " "))
        outline = " ".join(seg_text)
        outline = " ".join(outline.split())

        # 去停用词与HTML标签
        outline_list = outline.split(" ")
        outline_list_filter = [item for item in outline_list if item not in stopwords_set]
        outline = " ".join(outline_list_filter)

        return outline

    @staticmethod
    def remove_item_ngram(item_data):
        """
        删除item_data中的ngram
        :param item_data:               item数据
        :return:
        """
        global need_remove_ngrams

        tokens = item_data.strip().split()
        tokens_sentence = " ".join(tokens)
        for item in need_remove_ngrams:
            tokens_sentence = tokens_sentence.replace(item, '')
        new_tokens = tokens_sentence.split()
        return " ".join(new_tokens)
