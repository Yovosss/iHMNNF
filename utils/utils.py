#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @File     utils.py
    @Author   WZX
    @Date     2020/12/17 15:59
    @Describe 
    @Version  1.0
"""
import os
import gzip
import pickle
import codecs
from config.orcl import db_connection
from sqlalchemy import Table, select, text, distinct, types

class FileUtils():

    @staticmethod
    def read_db_file(db_name, in_table, subtype_name):
        """
        读取数据库文件，返回由每行文本组成的list
        :db_name: 数据库名称 in_table: 要读取的表名
        :return: list
        """
        db_conn = db_connection(db_name)
        in_table_name = Table(in_table, db_conn.metadata, autoload=True)

        s1 = select([in_table_name.c.record
                     ]).where(in_table_name.c.subtype_name == subtype_name)

        result = db_conn.conn.execute(s1).fetchall()
        results = [re.strip() for (re,) in result if re]
        return results


    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print("[INFO]: path={} does not exist, has been created.".format(path))
        else:
            print("[INFO]: path={} already exists".format(path))

    @staticmethod
    def read_txt_file(file_name):
        """
        读取文本文件，返回由每行文本组成的list
        :param filename: 文件名
        :return: list
        """
        lines = []
        with codecs.open(file_name, "r", encoding = "utf-8") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                lines.append(line)
            return lines

    @staticmethod
    def write_txt_file(cont_list, file_name):
        """
        将list写入文本文件，并换行
        :param: list
        :return: txt file
        """
        with codecs.open(file_name, "w+", encoding = "utf-8") as outfile:
            for i in cont_list:
                outfile.write(i)
                outfile.write('\r\n')


    @staticmethod
    def df2db(db_name, table_name, df):
        """
        describe:将dataframe数据插入到oracle表
        param: db_name 数据库, table 目标表, df 要插回的dataframe
        """
        db_conn = db_connection(db_name)
        df.to_sql(table_name, db_conn.engine, index=False, if_exists='append', dtype=None, chunksize=100)

    @staticmethod
    def read_mnist(file_path):
        with gzip.open(file_path) as fp:
            training_data, valid_data = pickle.load(fp, encoding='bytes')
        return training_data, valid_data

def sparsify(mat):
    coocode = []
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if(mat[i][j]==None):
                continue;
            else:
                coocode.append((i,j,mat[i][j]))
    return {'timestep':len(mat),'features':len(mat[0]),'codes':coocode};

