
# coding = utf-8

import yaml
import codecs
import pickle
import datetime
import pandas as pd
import math

import numpy as np

# 读取yml配置文件
def read_yml(yml_file):
    return yaml.load(codecs.open(yml_file, encoding = 'utf-8'), Loader=yaml.FullLoader)

# 保存读取pkl文件
def pickle_save(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))

def now():
    now = datetime.datetime.now()
    return str(now.strftime("%Y%m%d_%H%M%S"))

def frame_cut(dataframe, cut_n):
	d = ((pd.Series(dataframe.index).astype(float) + 1)/cut_n).apply(lambda x: math.ceil(x))
	cut = {}
	for i in d.unique():
	    cut[i] = dataframe[d==i]
	return cut