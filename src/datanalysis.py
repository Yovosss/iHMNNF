#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @File     datanalysis.py
    @Author   WZX
    @Date     2020/12/18 13:51
    @Describe 主要用于对分词后的数据进行文本统计分析
    @Version  1.0
"""
import re
import os
import codecs
import collections
from utils import FileUtils
import matplotlib.pyplot as plt

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class DataAnalysis():

    def __init__(self):

        """text basic information"""
        # 按词数不同的句子数分布统计，key=单个句子内词数，value=对应词数的句子数
        self.sent_distribution_by_word_count_dict = {}
        # 词频，key=词，value=词频
        self.word_fre = {}
        # 词表大小，值为 len(word_fre.keys())
        self.word_uniq_count = 0
        # 句子数统计
        self.sent_count = 0
        # 平均句长（句子长度和除以句子总数）
        self.avg_sent_len = 0
        # 最大句长
        self.max_sent_len = 0
        #是否将结果输出
        self.info_output = True

        return

    def get_basic_info(self, input_list):

        """统计文本数据的基本信息，如词频、词表大小、句子数统计等"""
        total_sent_len = 0
        for line in input_list:
            line_arr = line.strip()
            if not line_arr:
                continue
            self.sent_count += 1
            line_arr = [x for x in line_arr.split("/ ") if x.strip()]

            cur_sent_len = len(line_arr)
            if cur_sent_len in self.sent_distribution_by_word_count_dict.keys():
                self.sent_distribution_by_word_count_dict[cur_sent_len] += 1
            else:
                self.sent_distribution_by_word_count_dict[cur_sent_len] = 1
            if cur_sent_len > self.max_sent_len:
                self.max_sent_len = cur_sent_len
            total_sent_len += cur_sent_len
            for word in line_arr:
                if word in self.word_fre.keys():
                    self.word_fre[word] += 1
                else:
                    self.word_fre[word] = 1
        self.word_uniq_count = len(self.word_fre.keys())
        self.avg_sent_len = total_sent_len * 1.0 / self.sent_count


    def output_info(self, output_filename):
        """输出基本信息统计结果"""
        if self.info_output:
            FileUtils.mkdir(os.path.dirname(__file__) + "/output")
            with codecs.open(os.path.dirname(__file__) + "/output/{}.txt".format(output_filename), "w", encoding="utf-8") as fout:
                fout.write("\n".join(self.word_fre.keys()))

        with codecs.open(os.path.dirname(__file__) + "/output/{}.txt".format(output_filename+"_1"), "w", encoding="utf-8") as fout:
            fout.write("词表大小\t:\t{}\n".format(self.word_uniq_count))
            fout.write("句子总数\t:\t{}\n".format(self.sent_count))
            fout.write("平均句长\t:\t{}\n".format(self.avg_sent_len))
            fout.write("最大句长\t:\t{}\n".format(self.max_sent_len))

        with codecs.open(os.path.dirname(__file__) + "/output/{}.txt".format(output_filename+"_2"), "w", encoding="utf-8") as fout:
            fout.writelines("{0}\t:\t{1}\n".format(k, v) for k,v in sorted(self.word_fre.items(), key=lambda item:item[1], reverse=True))
            
        """画图"""
        # 容器准备
        # 额外有一个信息汇总表
        row_num = 2 + 1
        col_num = 1

        fig, ax = plt.subplots(nrows=row_num, ncols=col_num)
        fig.set_size_inches(col_num * 10, row_num * 10)
        fig.set_dpi(100)
        axes = ax.flatten()

        # 信息汇总表
        axes[0].axis('tight')
        axes[0].axis('off')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['bottom'].set_visible(False)
        axes[0].spines['left'].set_visible(False)

        axes[0].set_title("基本信息")
        col_labels = ["数值"]
        row_labels = ["词表大小", "句子总数", "平均句长", "最大句长"]
        table_val = [[self.word_uniq_count], [self.sent_count], [self.avg_sent_len], [self.max_sent_len]]

        axes[0].table(cellText=table_val, rowLabels=row_labels, colLabels=col_labels, colWidths=[1]*3,
                      rowLoc='center', colLoc='center', loc='upper center', cellLoc='center')

        # 不同长度的句子数分布
        # 数据准备
        top_k = min(10, len(self.sent_distribution_by_word_count_dict.keys()))
        # 按value降序排列
        sorted_sent_distribution_by_word_count_dict = collections.OrderedDict(
            sorted(self.sent_distribution_by_word_count_dict.items(), key=lambda t: t[1], reverse=True)
        )
        x_list = list(sorted_sent_distribution_by_word_count_dict.values())[:top_k]
        y_list = list(sorted_sent_distribution_by_word_count_dict.keys())[:top_k]
        # 画图
        self.draw_barh(axes[1], x_list, y_list, "句子数", "句长", "不同长度的句子数分布(top {})".format(top_k))

        # 词频分布
        # 数据准备
        top_k = min(50, len(self.word_fre.keys()))
        # 按 value 降序排列
        sorted_word_fre = collections.OrderedDict(
            sorted(self.word_fre.items(), key=lambda t: t[1], reverse=True)
        )
        x_list = list(sorted_word_fre.values())[:top_k]
        y_list = list(sorted_word_fre.keys())[:top_k]
        # 画图
        self.draw_barh(axes[2], x_list, y_list, "频次", "词", "词频分布(top {})".format(top_k))

        plt.savefig(os.path.dirname(__file__) + "/output/{}".format(output_filename+".png"))
        plt.show()

    def draw_barh(self, ax, x_list, y_list, x_name, y_name, title):
        ax.barh(range(len(y_list)), x_list)
        ax.set_title(title)
        # x 轴标题
        ax.set_xlabel(x_name)
        # y 轴标题
        ax.set_ylabel(y_name)
        # y 轴柱的数量及对应的label
        ax.set_yticks(range(len(y_list)))
        ax.set_yticklabels(y_list)
        min_sent_count = min(x_list)
        max_sent_count = max(x_list)
        # x 轴范围
        ax.set_xlim(min_sent_count - 100, max_sent_count + 100)
        for i, j in enumerate(x_list):
            ax.text(j + 0.2, i - 0.1, "{}".format(j))