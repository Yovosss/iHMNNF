#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @File     tokenization.py
    @Author   WZX
    @Date     2020/12/17 15:59
    @Describe 主要用于对特定表，特定文本字段进行分词，调用的分词接口是结巴分词
    @Version  1.0
"""

import re
import os
import jieba
import codecs
import matplotlib as plt

from utils import FileUtils
from datanalysis import DataAnalysis


# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class Tokenizer():
    """
    param: db_name 数据库名称, intable 表名, subtype_name 文本类型, word_segment_flag 是否做分词, sentence_break_flag 是否做句子切分
    """

    def __init__(self, db_name, in_table, subtype_name, word_segment_flag=True, sentence_break_flag=False):

        self.db_name = db_name
        self.in_table = in_table
        self.subtype_name = subtype_name

        self.sentence_break_flag = sentence_break_flag
        self.word_segment_flag = word_segment_flag
        self.sent_break = "。"

    def sent_segment(self, input_str):
        """切句，返回list"""
        sent_list = []

        input_str = re.sub(r"\\+r\\+n", r"\r\n", input_str)
        input_str = re.sub(r"\\+n", r"\n", input_str)

        break_str = self.sent_break + "\r\n"
        pattern = re.compile(r"[^" + break_str + r"]*[" + break_str + r"]")
        search_obj = pattern.search(input_str)
        while search_obj:
            if search_obj.group().strip(break_str):
                sent_list.append(search_obj.group().strip())
            input_str = input_str[search_obj.end():]
            search_obj = pattern.search(input_str)
        if input_str.strip(break_str):
            sent_list.append(input_str.strip())
        return sent_list

    def word_segment(self, input_str):
        """
        分词
        :param input_str:待处理字符串
        :return: 分词后的结果，以/ 为分隔符
        """
        cont = input_str
        cont_res = "/ ".join(jieba.cut(cont, cut_all=False))

        return cont_res


    def tokenizer_pipline(self):
        """
        param: db_name, in_table, subtype_name ===>> list形式的文本
        return: 分词后的文本list, 可以作为输入进行词语分析
        """

        dataset = FileUtils.read_db_file(self.db_name, self.in_table, self.subtype_name)
        new_dataset = []
        
        if dataset:
            for line in dataset:
                if self.sentence_break_flag:
                    sent_list = self.sent_segment(line)
                else:
                    sent_list = [line]
                for sent in sent_list:
                    if self.word_segment_flag:
                        sent = self.word_segment(sent)
                    new_dataset.append(sent)
        else:
            print("[WARN]: The list of input is null")

        return new_dataset

if __name__ == '__main__':
    """
    下述只是测试用，后续会写到main.py文件中，单独写一个函数进行封装
    """

    #定义预分词的表和文本类型
    data = Tokenizer('fuo', 'ORIGIN_NOTE', '主诉')
    #获取分词后的文本list
    result = data.tokenizer_pipline()

    #定义数据分析实例
    ana = DataAnalysis()
    ana.get_basic_info(result)
    ana.output_info('outest')
