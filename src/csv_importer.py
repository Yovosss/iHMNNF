#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @File     csv_importer.py
    @Author   WZX
    @Date     2020/01/18 13:51
    @Describe 对plsql和sqldev无法导入数据库的包含CLOB字段的csv文件进行单独处理，利用该脚本将其导入oracle数据库
    @Version  1.0
"""

import cx_Oracle
import csv
import datetime
import os
import codecs
from itertools import islice
import pandas as pd
import numpy as np
import sys

from config.orcl import db_connection
from sqlalchemy import Table, select, text, distinct, types
from sqlalchemy.dialects.oracle import BFILE, BLOB, CHAR, CLOB, DATE, DOUBLE_PRECISION, FLOAT, INTERVAL, LONG, NCLOB, NUMBER, NVARCHAR, NVARCHAR2, RAW, TIMESTAMP, VARCHAR, VARCHAR2

class CsvImporter():

    def __init__(self, db_name, db_name_sys, file_path, column_query_table, out_table):
        """
        input: db_name(数据库名称)，db_name_sys(数据库sys账户), file_path(要导入的csv文件地址), column_query_table(查询列名表名称), out_table(输出表名称)
        """
        #db connection
        self.db_conn = db_connection(db_name)
        self.db_conn_sys = db_connection(db_name_sys)
        self.column_query = Table(column_query_table, self.db_conn_sys.metadata, autoload = True)
        self.out_table_query = Table(out_table, self.db_conn.metadata, autoload = True)
        self.out_table_name = out_table
        self.path = file_path

    def getcolumn(self):
        #get column name list
        s1 = select([self.column_query.c.column_name]).where(self.column_query.c.table_name == self.out_table_name)
        table_column = self.db_conn_sys.conn.execute(s1).fetchall()
        column = [col for (col,) in table_column]
        return column

    #定义后续插入数据库时的字符类型
    def mapping_df_types(self, df):
        """
        INPUT: df(要插回数据库的DataFrame数据)
        OUTPUT: dtypedict(指定每个字段要插入到数据库的字段类型)
        """
        #定义空字典
        dtypedict = {}

        for i, j in zip(df.columns, df.dtypes):
            if str(i) == "note_id":
                dtypedict.update({i: VARCHAR2(50)})
            if str(i) =="pno":
                dtypedict.update({i: VARCHAR2(10)})
            if str(i) == "type":
                dtypedict.update({i: VARCHAR2(20)})
            if str(i) == "subtype_name":
                dtypedict.update({i: VARCHAR2(60)})
            if str(i) == "record":
                dtypedict.update({i: CLOB})
            if str(i) == "time":
                dtypedict.update({i: VARCHAR2(60)})
            if str(i) == "provider":
                dtypedict.update({i: VARCHAR2(60)})
        return dtypedict

    #导入数据库
    def insertdata(self, df_data, dtypedict_status=True):
        """
        INPUT: df_data(要插回数据库的数据)
        """
        #获取定义的字典
        if dtypedict_status:
            dtypedict = self.mapping_df_types(df_data)
        else:
            dtypedict = None
        #写入数据库  方式一
        df_data.to_sql(self.out_table_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)

        #写入数据库 方式二
        # print(df_data.to_dict(orient='records'))
        # self.db_conn.engine.execute(self.out_table_query.insert(), df_data.to_dict(orient='records'))  #有问题
        #写入数据库 方式三
        # conn = self.db_conn.engine.raw_connection()
        # cursor = conn.cursor()
        # col = ', '.join(df_data.columns.tolist())
        # print(col)
        # s = ', '.join([':'+str(i) for i in range(1, df_data.shape[1]+1)])
        # print(s)
        # sql = 'insert into {}({}) values({})'.format(self.out_table_name, col, s)
        # print(sql)
        # cursor.executemany(sql, df_data.values.tolist())
        # if count % 100 == 0 or count == 116553:
        #     conn.commit()
        # conn.commit()
        # cursor.close()


    def csvimporter(self):
        columns = self.getcolumn()
        columns.reverse()

        with codecs.open(self.path, "rb", encoding='gb18030', errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            results = pd.DataFrame([], columns = columns)
            for line in islice(reader, 1, None):
                sys.stderr.write("[INFO]: processing " + str(count+1) + " lines\r\n")
                # count += 1
                result = pd.DataFrame([line], columns = columns)
                results = results.append(result)
                count += 1
                if len(results) == 1000 and count/1000 < round(5160767/1000):
                    self.insertdata(results, dtypedict_status=False)
                    sys.stderr.write("[INFO]: submitted " + str(count) + " lines\r\n")
                    results.drop(results.index, inplace=True)

                elif len(results) < 1000 and count/1000 > round(5160767/1000)-1:
                    self.insertdata(result, dtypedict_status=False)
                    sys.stderr.write("[INFO]: submitted " + str(count) + " lines\r\n")
                    result.drop(result.index, inplace=True)


if __name__ == '__main__':

    #导入ORIGIN_MEDITECH
    # infile = r"E:\jupyter-notebook\Regex Test\MEDITECH.csv"
    # c = CsvImporter('fuo', 'fuo_sys', infile, 'ALL_TAB_COLUMNS', 'ORIGIN_MEDITECH')
    # c.csvimporter()
    
    # 导入文本数据
    # infile = r"E:\jupyter-notebook\Regex Test\NOTE1.csv"
    # c = CsvImporter('fuo', 'fuo_sys', infile, 'ALL_TAB_COLUMNS', 'ORIGIN_NOTE_1')
    # c.csvimporter()

    # infile = r"E:\jupyter-notebook\Regex Test\NOTE2.csv"
    # c = CsvImporter('fuo', 'fuo_sys', infile, 'ALL_TAB_COLUMNS', 'ORIGIN_NOTE_2')
    # c.csvimporter()

    # 导入医嘱数据    执行完毕
    infile = r"/mnt/data/wzx/0-data/IM_DRUG_V2.csv"
    c = CsvImporter('fuo', 'fuo_sys', infile, 'ALL_TAB_COLUMNS', 'ORIGIN_DRUG_V4')
    c.csvimporter()