#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author: wang zhixiao
@time  : 2020-12-10 11:36
@detail: (NoDup)对首次病程记录文本数据进行去重||
         (FirstRecordSplit)对首次病程记录文本数据做初步切分||
         (HeightAndWeight)对护理生命体征内的身高体重更新到患者就诊表||
         (LabelProcess)对标签数据进行处理||
         (SymAndChiefcom)对结构化之后的症状与主诉数据进行再处理与合并
         (FeatureAggre)读取个人史、既往史、主诉与症状合并、就诊、化验数据进行合并与转置
         "======================================================================="
         (LabelComplement)提取condition表的diag_append字段内的指示性信息，对发热患者的标签进行补全
         (LabelCV)依据药物类型提取可能的标签，并对source_i, source_ii, source_liu和source_iii进行交叉合并
         (MeasurementProcess)对化验数据内的分类变量与数值变量的异常值进行处理
         (SymAndChiefcomV2)二次对症状和主诉做处理
"""

import re
import os
import xlrd
import pickle
import codecs
import openpyxl
import cx_Oracle
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import matplotlib as mpl
from collections import Counter
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from itertools import cycle
from scipy import interp
from utils.utils import FileUtils
from config.orcl import db_connection
from sqlalchemy import Table, select, text, distinct, types
from sqlalchemy.dialects.oracle import BFILE, BLOB, CHAR, CLOB, DATE, DOUBLE_PRECISION, FLOAT, INTERVAL, LONG, NCLOB, NUMBER, NVARCHAR, NVARCHAR2, RAW, TIMESTAMP, VARCHAR, VARCHAR2

class NoDup():
    """
    Args:
        db_name: The orcl of the database
        in_table: The target table to precess, here is ORIGIN_NOTE_DUP
    Returns:
        No returns, all the changes will be updated into the table ORIGIN_NOTE_DUP
    """
    
    def __init__(self, db_name, in_table):

        self.db_conn = db_connection(db_name)
        self.table = Table(in_table, self.db_conn.metadata, autoload = True)
        self.table_inference = Table('ORIGIN_NOTE', self.db_conn.metadata, autoload=True)
        self.subtype_name ='主诉'

    def getvrilist(self):

        s1 = select([distinct(self.table.c.visit_record_id)])
        result = self.db_conn.conn.execute(s1).fetchall()
        vris = [vri for (vri,) in result]
        return vris

    def getalldata(self, vri):

        s2 = select([self.table.c.note_id,
                     self.table.c.visit_record_id,
                     self.table.c.record,
                     self.table.c.time,
                     self.table.c.flag
                     ]).where(self.table.c.visit_record_id == vri)

        result = self.db_conn.conn.execute(s2).fetchall()
        result = pd.DataFrame(result, columns=['note_id', 'visit_record_id', 'record', 'time', 'flag'])
        
        return result

    def getalldata2(self,vri):

        s3 = select([self.table_inference.c.note_id,
                     self.table_inference.c.visit_record_id,
                     self.table_inference.c.record
                     ]).where(self.table_inference.c.subtype_name == self.subtype_name).where(self.table_inference.c.visit_record_id==vri)
        result = self.db_conn.conn.execute(s3).fetchall()
        result = pd.DataFrame(result, columns = ['note_id', 'visit_record_id', 'record'])

        return result

    def process(self):

        vri = self.getvrilist()

        for v in vri:

            print("[VISIT_RECORD_ID]:{}".format(v))
            dicts = {}
            dicts_diff = {}
            data = self.getalldata(v)
            data_inference = self.getalldata2(v)

            if not data_inference.empty:
                record_inference = data_inference['record'].values[0]

                if not data_inference.empty and data_inference['record'].values[0] is not None:
                    record_inference = re.sub(r"\\r\\n", r"\r\n", record_inference)   #回车和换行，统一替换为\r或\n
                    record_inference = re.sub(r"\s+", "", record_inference)  #去除空格
                    record_inference = re.sub(r"\s{0,1}[,|，]\s{0,1}", "，", record_inference)   #将英文与中文冒号，及其带空格的部分，全部替换为英文:
                    if not data.empty:

                    # if not data.empty and data['record'].values[0] is not None:
                        for index, row in data.iterrows():

                            record = row['record']
                            record = re.sub(r"\\r\\n", r"\r\n", record)   #回车和换行，统一替换为\r或\n
                            record = re.sub(r"\s+", "", record)  #去除空格
                            record = re.sub(r"\s{0,1}[,|，]\s{0,1}", "，", record)

                            record_regex = re.search(r"(?<=因“)(\S)*?(?=”入院)", record)

                            if record_regex.group() == record_inference:

                                dicts.setdefault(row['note_id'], {})
                                dicts[row['note_id']]['record'] = row['record']
                                dicts[row['note_id']]['length'] = len(row['record'])
                                dicts[row['note_id']]['time'] = row['time']

                            else:
                                dicts_diff.setdefault(row['note_id'], {})
                                dicts_diff[row['note_id']]['record'] = record_regex.group()
                                dicts_diff[row['note_id']]['length'] = len(row['record'])
                                dicts_diff[row['note_id']]['time'] = row['time']

                    
                        if len(dicts) != 0:
                    
                            if len(dicts) == 1:
                                sql = self.table.update().where(self.table.c.note_id == max(dicts, key=lambda v: dicts[v]['length'])).values(flag='1')
                                self.db_conn.conn.execute(sql)
                            else:
                                ##从多个键值对内取出record长度最长或时间较大的键值对，更新
                                sql = self.table.update().where(self.table.c.note_id == max(dicts, key=lambda v: dicts[v]['length'])).values(flag='1')
                                self.db_conn.conn.execute(sql)                         

                        elif len(dicts_diff) != 0:

                            ## 取正则化后的症状相等的，取其最长的记录flag更新为1，否则人工处理
                            rec = []
                            result = True
                            for key, value in dicts_diff.items():
                                rec.append(value['record'])
                            
                            for r in range(len(rec)-1):
                                for c in range(r + 1, len(rec)):
                                    if rec[r] != rec[c]:
                                        result = False
                                        break
                            ## 如果有一个不相同的元素，则result为False
                            if result:
                                ## 全部相同则取length最长的元素更新为1
                                sql = self.table.update().where(self.table.c.note_id == max(dicts_diff, key=lambda v: dicts_diff[v]['length'])).values(flag='1')
                                self.db_conn.conn.execute(sql)
                            else:
                                print("[message]: the data of {} need to be reviewed by engineer!!".format(v))

                        else:

                            print("[error]: the first records of [{}] are wrong".format(v))
            else:
                print("[error]: the chief conplaint of  {} is None".format(v))

class FirstRecordSplit():

    def __init__(self, db_name, in_table, out_table):
        """
        input: db_name(数据库名称)，in_table(输入表名称), out_table(输出表名称)
        """
        self.db_conn = db_connection(db_name)
        self.in_table_name = Table(in_table, self.db_conn.metadata, autoload = True)
        self.out_table_name = out_table
    
    def getnoteidlist(self, subtype_name):
        s1 = select([distinct(self.in_table_name.c.note_id)]).where(self.in_table_name.c.subtype_name == subtype_name)
        result = self.db_conn.conn.execute(s1).fetchall()  #fetch可以改为fetchall或fetchone
        vris = [vri for (vri,) in result]
        return vris

    def getalldata(self, vri, subtype_name):
        """
        INPUT: vri(文本序号), subtype_name(文本类型名称)
        OUTPUT: result(文本序号对应的文本数据)
        """
        s1 = select([self.in_table_name.c.note_id, 
                     self.in_table_name.c.person_id, 
                     self.in_table_name.c.visit_record_id,
                     self.in_table_name.c.visit_record_id_new,
                     self.in_table_name.c.subtype_name, 
                     self.in_table_name.c.record,
                     self.in_table_name.c.time,
                     self.in_table_name.c.provider
                     ]).where(self.in_table_name.c.note_id == vri).where(self.in_table_name.c.subtype_name == subtype_name)
        result = self.db_conn.conn.execute(s1).fetchall()
        result = pd.DataFrame(result, columns=['note_id', 'person_id', 'visit_record_id', 'visit_record_id_new', 'subtype_name', 'record', 'time', 'provider'])
        return result

    def recordsplit(self, infile):
        """
        INPUT: infile(数据库内获取的患者就诊对应的文本数据)
        OUTPUT: record_split_dict(切分后的字典，key为文本类型，value为文本)
        """
        record = infile['record'].values[0]
        record = re.sub(r"\\r\\n", r"\r\n", record)
        record = record.strip()
        record = re.sub(r"\s{0,1}[:|：]\s{0,1}", ":", record)

        record_split_dict = {}
        rec_pattern = re.compile(r"患者{1}(.+)因{1}(.*)(?=(初步诊断))", re.DOTALL)
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['ZS'] =  obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-主诉）" %(infile['note_id'].values[0]))

        rec_pattern = re.compile(r"(?<=初步诊断:)(.*)(?=(本病例特点))", re.DOTALL)  #如冒号未统一为英文，则需要修改
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['CBZD'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-初步诊断）" %(infile['note_id'].values[0]))
                    
        rec_pattern = re.compile(r"(?<=本病例特点:)(.*)(?=(诊断依据))", re.DOTALL)  #如冒号未统一为英文，则需要修改
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['BLTD'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-病例特点）" %(infile['note_id'].values[0]))

        rec_pattern = re.compile(r"(?<=(诊断依据:1．病史:))(.*)(?=(2{1}[．|.|。]症状))", re.DOTALL)  
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['BS'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-病史）" %(infile['note_id'].values[0]))        

        rec_pattern = re.compile(r"(?<=(2．症状:))(.*)(?=(3{1}[．|.|。]体征))", re.DOTALL)  
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['ZZ'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-症状）" %(infile['note_id'].values[0])) 

        rec_pattern = re.compile(r"(?<=(3．体征:))(.*)(?=(4{1}[．|.|。]辅助检查))", re.DOTALL)  
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['TZ'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-体征）" %(infile['note_id'].values[0])) 

        rec_pattern = re.compile(r"(?<=(4．辅助检查:))(.*)(?=(鉴别诊断))", re.DOTALL)  
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['FZJC'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-辅助检查）" %(infile['note_id'].values[0])) 

        rec_pattern = re.compile(r"(?<=(鉴别诊断:))(.*)(?=(诊疗措施:|诊疗计划:))", re.DOTALL)  
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['JBZD'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-鉴别诊断）" %(infile['note_id'].values[0]))   

        rec_pattern = re.compile(r"(?<=(诊疗措施:|诊疗计划:))(.*)", re.DOTALL)  
        obj = rec_pattern.search(record)
        if obj is not None:
            record_split_dict['ZLCS'] = obj.group()
        else:
            print("患者 %s 未提取到（首次病程记录-诊疗措施）" %(infile['note_id'].values[0]))

        return record_split_dict

    def semistrucinsert(self, infile, semi_struc):
        """
        INPUT: infile(数据库内获取到的患者首次病程记录文本数据), semi_struc(文本切分后的字典)
        OUTPUT: infile(重新组合成DataFrame的切分后的数据)
        """
        count = 1
        for i, j in semi_struc.items():
            new = pd.DataFrame({'note_id': infile['note_id'].values[0],
                        'person_id': infile['person_id'].values[0],
                        'visit_record_id':infile['visit_record_id'].values[0],
                        'visit_record_id_new':infile['visit_record_id_new'].values[0],
                        'subtype_name': str(infile['subtype_name'].values[0]) + '_' + i,
                        'record': j,
                        'time':infile['time'].values[0],
                        'provider': infile['provider'].values[0]}, index = [0])
            count += 1
            infile = infile.append(new, ignore_index = True)

        return infile
    
    def mapping_df_types(self, df):
        """
        INPUT: df(要插回数据库的DataFrame数据)
        OUTPUT: dtypedict(指定每个字段要插入到数据库的字段类型)
        """
        dtypedict = {}

        for i, j in zip(df.columns, df.dtypes):
            if str(i) == "note_id":
                dtypedict.update({i: VARCHAR2(20)})
            if str(i) =="person_id":
                dtypedict.update({i: VARCHAR2(54)})
            if str(i) == "visit_record_id":
                dtypedict.update({i: VARCHAR2(103)})
            if str(i) == "visit_record_id_new":
                dtypedict.update({i: VARCHAR2(53)})
            if str(i) == "subtype_name":
                dtypedict.update({i: VARCHAR2(50)})
            if str(i) == "record":
                dtypedict.update({i: CLOB})
            if str(i) == "time":
                dtypedict.update({i: VARCHAR2(19)})
            if str(i) == "provider":
                dtypedict.update({i: VARCHAR2(100)})
        return dtypedict

    def insertdata(self, df_data, dtypedict_status=True):
        """
        INPUT: df_data(要插回数据库的数据)
        """
        if dtypedict_status:
            dtypedict = self.mapping_df_types(df_data)
        else:
            dtypedict = None
        df_data.to_sql(self.out_table_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=100)

class HeightAndWeight():

    def __init__(self, db_name, in_table, out_table):
        """
        对同一患者存在的多条身高体重数据进行处理，刨除为0的数据，求均值
        """
        
        self.db_conn = db_connection(db_name)
        self.table_in = Table(in_table, self.db_conn.metadata, autoload = True)
        self.table_out = Table(out_table, self.db_conn.metadata, autoload = True)

    def get_vri(self, htype):

        s = select([distinct(self.table_in.c.visit_record_id)]).\
            where(self.table_in.c.type == htype)
        vri = self.db_conn.conn.execute(s).fetchall()
        result = [res for (res,) in vri]

        return result

    def getdata(self, htype, vri):

        s = select([self.table_in.c.visit_record_id,
                    self.table_in.c.type,
                    self.table_in.c.value]).\
                        where(self.table_in.c.type == htype).\
                            where(self.table_in.c.visit_record_id == vri)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['visit_record_id', 'type', 'value'])

        return result

    def hhupdate(self, htype):
        """
        分别针对体重和身高，sql中的更新字段名需要更换
        """
        vri = self.get_vri(htype)
        rootDir = os.path.split(os.path.realpath(__file__))[0]
        logpath = os.path.join(rootDir, 'log_1.txt')

        for v in vri:
            print("[Info] The data of {} are processing".format(v))
            data = self.getdata(htype, v)
            if not data.empty:
                values = [float(x) for x in data['value'].tolist()]
                if any(values):
                    values = list(filter(lambda x:x!=0, values))
                    values_mean = round(np.mean(values), 1)
                    sql = self.table_out.update().\
                        where(self.table_out.c.visit_record_id == data.iloc[0,0]).\
                            values(height=str(values_mean))
                    self.db_conn.conn.execute(sql)
                else:
                    sql = self.table_out.update().\
                        where(self.table_out.c.visit_record_id == data.iloc[0,0]).\
                            values(height='0')
                    self.db_conn.conn.execute(sql)

            else:
                print("[error] the {1} data of {2} is None".format(htype, v))
                f = open(logpath, 'a')
                f.write("[error] the {1} data of {2} is None".format(htype, v) + "\n")
                f.close()
                continue

class LabelProcess():

    # caution: 有默认值的参数，必须放在形参的最后
    def __init__(self, db_name, in_table, filepath_label=False):

        self.filepath = filepath_label
        self.db_conn = db_connection(db_name)
        self.in_table = Table(in_table, self.db_conn.metadata, autoload=True)

    def read_file(self):
        file = pd.read_excel(self.filepath, sheet_name='出院主要诊断频次分布', \
            names = ['诊断名称','唯一就诊数', '特殊情况', '感染性', '细菌', '病毒', '真菌', '寄生虫', '其他1', '非感染性', 'NIID', '自身免疫性', '自身炎症性', '肿瘤性', '血液系统恶性疾病', '实体恶性肿瘤', '良性肿瘤', '其他2'])
        df_label = pd.DataFrame(file, dtype=np.object)

        return df_label
    
    def label_process(self):
        label_data = self.read_file()

        # inplace character
        label_data.replace('✔', '1', inplace=True)
        label_data.replace('×', '0', inplace=True)

        label_data.loc[label_data['诊断名称'] == '感染性发热', '感染性'] = '1'
        label_data.loc[label_data['诊断名称'] == '感染性发热', '细菌':'寄生虫'] = '0'
        label_data['唯一就诊数'].astype(np.str)
        label_data=label_data.applymap((lambda x:"".join(x.split()) if type(x) is str else x))
        #insert into db
        FileUtils.df2db('fuo', 'LABEL', label_data)
    
    def label_ratio(self):

        s1 = select([self.in_table.c.诊断名称,
                    self.in_table.c.唯一就诊数,
                    self.in_table.c.感染性,
                    self.in_table.c.非感染性])
        result = self.db_conn.conn.execute(s1)
        df_res = pd.DataFrame(result, columns=['name', 'count', 'infec', 'noinfec'])
        df_res['count'] = df_res['count'].astype('int')
        # calculate the ratio of labeled data
        all_count = df_res.count().values[1]
        label_count = df_res[df_res['count'] > 9].count().values[1]
        ratio1 = round(label_count/all_count, 4)
        ratio2 = round(df_res[df_res['count'] > 9]['count'].sum()/df_res['count'].sum(), 4)
        print("总的标签量为：{}".format(all_count))
        print("频次10以上的已打标签量为：{}".format(label_count))
        print("频次10以上的标签量占比：{}".format(ratio1))
        print("频次10以上的已打标签覆盖就诊数据量比例为：{}".format(ratio2))

        # calculate infectious and non-infectious data ratio
        infec_label_count = df_res[df_res['infec'] == '1'].count().values[1]
        noinfec_label_count = df_res[df_res['noinfec'] == '1'].count().values[1]
        infec_data_count = df_res.loc[df_res['infec'] == '1', 'count'].sum()
        noinfec_data_count = df_res.loc[df_res['noinfec'] == '1', 'count'].sum()
        print("感染性标签量为：{}".format(infec_label_count))
        print("非感染性标签量为：{}".format(noinfec_label_count))
        print("感染性标签对应就诊量为：{}".format(infec_data_count))
        print("非感染性标签对应就诊量为：{}".format(noinfec_data_count))

class SymAndChiefcom():

    def __init__(self, db_name, db_name_sys, sym_table, chiefcom_table):

        self.db_conn = db_connection(db_name)
        self.db_conn_sys = db_connection(db_name_sys)
        self.column_query = Table('ALL_TAB_COLUMNS', self.db_conn_sys.metadata, autoload = True)
        self.sym_table = Table(sym_table, self.db_conn.metadata, autoload=True)
        self.chiefcom_table = Table(chiefcom_table, self.db_conn.metadata, autoload=True)

    def getcolumn(self, table_name):
        #get column name list
        s = select([self.column_query.c.column_name]).where(self.column_query.c.table_name == table_name)
        table_column = self.db_conn_sys.conn.execute(s).fetchall()
        column = [col for (col,) in table_column]
        column.reverse()
        return column

    def mapping_df_types(self, df):
        """
        INPUT: df(要插回数据库的DataFrame数据)
        OUTPUT: dtypedict(指定每个字段要插入到数据库的字段类型)
        """
        dtypedict = {}

        for i, j in zip(df.columns, df.dtypes):
            if str(i) == "VISIT_RECORD_ID_NEW_1":
                dtypedict.update({i: VARCHAR2(20)})
            if str(i) =="NAME":
                dtypedict.update({i: VARCHAR2(100)})
            if str(i) == "VALUE":
                dtypedict.update({i: VARCHAR2(50)})
        return dtypedict

    def insertdata(self, df_data, dtypedict_status=True):
        """
        INPUT: df_data(要插回数据库的数据)
        """
        #获取定义的字典
        if dtypedict_status:
            dtypedict = self.mapping_df_types(df_data)
        else:
            dtypedict = None
        #写入数据库
        df_data.to_sql('P0_NOTE_SYM', self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=1000)

    def symprocess(self):
        
        # preprocess the p0_note_zz table 
        s1 = select([self.sym_table.c.note_id,
                    self.sym_table.c.person_id,
                    self.sym_table.c.visit_record_id_new_1,
                    self.sym_table.c.name,
                    self.sym_table.c.value,
                    self.sym_table.c.duration,
                    self.sym_table.c.frequency])
        dfs = self.db_conn.conn.execute(s1).fetchall()
        dfs = pd.DataFrame(dfs, columns=self.getcolumn('P0_NOTE_ZZ'))

        print("[Message]: p0_note_zz表原始数据量为 {} ".format(dfs.shape))
        # drop duplicates
        if any(dfs.duplicated()):
            dfs = dfs.drop_duplicates(keep='first')
            print("[Message]: p0_note_zz表经 全表去重 数据量为 {} ".format(dfs.shape))
        # drop not happened symptom
        dfs = dfs[dfs['VALUE'] != '无']
        print("[Message]: p0_note_zz表经 去除否定症状 数据量为 {} ".format(dfs.shape))

        # process the value of duration
        # generalization
        dfs['DURATION']=dfs['DURATION'].str.replace('日','天')
        dfs['DURATION']=dfs['DURATION'].str.replace('半天','1天')
        dfs['DURATION']=dfs['DURATION'].str.replace('半月','15天')
        dfs['DURATION']=dfs['DURATION'].str.replace('半年','180天')
        dfs['DURATION']=dfs['DURATION'].str.replace('十','10')
        dfs['DURATION']=dfs['DURATION'].str.replace('九','9')
        dfs['DURATION']=dfs['DURATION'].str.replace('八','8')
        dfs['DURATION']=dfs['DURATION'].str.replace('七','7')
        dfs['DURATION']=dfs['DURATION'].str.replace('六','6')
        dfs['DURATION']=dfs['DURATION'].str.replace('五','5')
        dfs['DURATION']=dfs['DURATION'].str.replace('四','4')
        dfs['DURATION']=dfs['DURATION'].str.replace('三','3')
        dfs['DURATION']=dfs['DURATION'].str.replace('二','2')
        dfs['DURATION']=dfs['DURATION'].str.replace('两','2')
        dfs['DURATION']=dfs['DURATION'].str.replace('一','1')

        # extract the duration value and unit
        dfs['DURATION_VALUE'] = dfs['DURATION'].str.extract(r'(\d+).*(?=天|小时|周|月|年)')
        # CAUTION:str.extract(),every () matched means a value
        dfs['DURATION_UNIT'] = dfs['DURATION'].str.extract(r'(小时|天|周|月|年){1}')
        # update the value of DURATION_VALUE as the unit of day
        dfs.loc[dfs['DURATION_UNIT'] == '小时', 'DURATION_VALUE_DAY'] = '1'
        dfs.loc[dfs['DURATION_UNIT'] == '天', 'DURATION_VALUE_DAY'] = dfs['DURATION_VALUE']
        dfs.loc[dfs['DURATION_UNIT'] == '周', 'DURATION_VALUE_DAY'] = dfs['DURATION_VALUE'].astype(float)*7
        dfs.loc[dfs['DURATION_UNIT'] == '月', 'DURATION_VALUE_DAY'] = dfs['DURATION_VALUE'].astype(float)*30
        dfs.loc[dfs['DURATION_UNIT'] == '年', 'DURATION_VALUE_DAY'] = dfs['DURATION_VALUE'].astype(float)*356

        return dfs

    def chiefcomprocess(self):
        s1 = select([self.chiefcom_table.c.note_id,
                     self.chiefcom_table.c.person_id,
                     self.chiefcom_table.c.visit_record_id,
                     self.chiefcom_table.c.visit_record_id_new_1,
                     self.chiefcom_table.c.sign_name,
                     self.chiefcom_table.c.duration_value,
                     self.chiefcom_table.c.frequency])
        dfc = self.db_conn.conn.execute(s1).fetchall()
        dfc = pd.DataFrame(dfc, columns=self.getcolumn('P0_NOTE_ZS'))

        print("[Message]: p0_note_zs表原始数据量为 {} ".format(dfc.shape))
        # drop all duplicates
        if any(dfc.duplicated()):
            dfc = dfc.drop_duplicates(keep='first')
            print("[Message]: p0_note_zs表经 全表去重 数据量为 {} ".format(dfc.shape))

        return dfc

    def symconcat(self):
        
        # extract symptom data
        dfs = self.symprocess()
        dfs_1 = dfs[['PERSON_ID', 'VISIT_RECORD_ID_NEW_1', 'NAME', 'VALUE', 'FREQUENCY', 'DURATION_VALUE_DAY']]
        dfs_1.rename(columns = {'DURATION_VALUE_DAY': 'DURATION_VALUE'}, inplace = True)

        # extract chief complaint data
        dfc = self.chiefcomprocess()
        dfc_1 = dfc[['PERSON_ID', 'VISIT_RECORD_ID_NEW_1', 'SIGN_NAME', 'FREQUENCY', 'DURATION_VALUE']]
        dfc_1.rename(columns = {'SIGN_NAME': 'NAME'}, inplace = True)
        dfc_1['VALUE'] = '有'

        # concat the two tables
        df = pd.concat([dfs_1, dfc_1], ignore_index = True)
        print("[Message]: 症状与主诉表经合并之后 数据量为 {} ".format(df.shape))

        # drop duplicates after concat, keep the earlier one
        df.sort_values(by = ['VISIT_RECORD_ID_NEW_1', 'NAME', 'DURATION_VALUE'], na_position = 'last', inplace = True)
        df_nodup = df.drop_duplicates(subset=['VISIT_RECORD_ID_NEW_1','NAME'], keep='first')
        print("[Message]: 症状与主诉表合并且去重之后 数据量为 {} ".format(df_nodup.shape))

        # process the symptom data
        iterrow_count = 1
        sym = pd.DataFrame([], columns = ['VISIT_RECORD_ID_NEW_1', 'NAME', 'VALUE'])
        for index, row in df_nodup.iterrows():
            sym = sym.append({'VISIT_RECORD_ID_NEW_1': row['VISIT_RECORD_ID_NEW_1'], 'NAME': row['NAME'], 'VALUE': row['VALUE']}, ignore_index=True)
            sym = sym.append({'VISIT_RECORD_ID_NEW_1': row['VISIT_RECORD_ID_NEW_1'], 'NAME': row['NAME']+'-'+'频次', 'VALUE': row['FREQUENCY']}, ignore_index=True)
            sym = sym.append({'VISIT_RECORD_ID_NEW_1': row['VISIT_RECORD_ID_NEW_1'], 'NAME': row['NAME']+'-'+'时长', 'VALUE': row['DURATION_VALUE']}, ignore_index=True)
            print("[Message: {} data have been processed".format(iterrow_count))
            iterrow_count += 1

        sym = sym.where(sym.notnull(),'')
        sym = sym.astype(str, copy=True)
        self.insertdata(sym, dtypedict_status=False)

class FeatureAggre():

    def __init__(self, db_name, db_name_sys, zz_table, jws_table, grs_table, visit_table, mea_table):

        self.db_conn = db_connection(db_name)
        self.db_conn_sys = db_connection(db_name_sys)
        self.column_query = Table('ALL_TAB_COLUMNS', self.db_conn_sys.metadata, autoload = True)
        self.visit_table = Table(visit_table, self.db_conn.metadata, autoload=True)
        self.zz_table = Table(zz_table, self.db_conn.metadata, autoload=True)
        self.jws_table = Table(jws_table, self.db_conn.metadata, autoload=True)
        self.grs_table = Table(grs_table, self.db_conn.metadata, autoload=True)
        self.mea_table = Table(mea_table, self.db_conn.metadata, autoload=True)

    def getcolumn(self, table_name):
        #get column name list
        s = select([self.column_query.c.column_name]).where(self.column_query.c.table_name == table_name)
        table_column = self.db_conn_sys.conn.execute(s).fetchall()
        column = [col for (col,) in table_column]
        column.reverse()
        return column

    def zzprocess(self):
        s = select([self.zz_table.c.visit_record_id_new_1,
                    self.zz_table.c.name,
                    self.zz_table.c.value])
        result = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(result, columns = self.getcolumn('P0_NOTE_SYM'))
        print("[Message]: 主诉与症状数据合并、去重与转置之后数据量为 {}".format(result.shape))

        return result

    def jwsprocess(self):
        s = select([self.jws_table.c.note_id,
                    self.jws_table.c.person_id,
                    self.jws_table.c.visit_record_id,
                    self.jws_table.c.visit_record_id_new_1,
                    self.jws_table.c.name,
                    self.jws_table.c.value])
        result = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(result, columns=self.getcolumn('P0_NOTE_JWS'))
        print("[Message]: 既往史结构化原始 数据量为 {} ".format(result.shape))

        # drop duplicates in the range of all columns
        df_jws = result[['VISIT_RECORD_ID_NEW_1', 'NAME', 'VALUE']]
        if any(df_jws.duplicated()):
            df_jws.sort_values(by = ['VISIT_RECORD_ID_NEW_1', 'NAME', 'VALUE'], na_position = 'last', inplace = True)
            df_jws = df_jws.drop_duplicates(keep='last')
            print("[Message]: 既往史结构化数据经 全表去重 数据量为 {} ".format(df_jws.shape))

        # take specific jws data
        name_range = ['既往体质', '输血史', '外伤史', '手术史', '中毒史', '肾病史', '糖尿病史', '心脏病史', '肺结核史', '高血压史', '长期用药史', '病毒性肝炎史', '食物药物过敏史', '其他传染病史']
        df_jws = df_jws[df_jws['NAME'].isin(name_range)]
        print("[Message]: 既往史结构化数据经 病史筛选 数据量为 {} ".format(df_jws.shape))

         # there are still duplicates in the range of ['visit_record_id_new_1', 'name']
        if any(df_jws.duplicated(subset=['VISIT_RECORD_ID_NEW_1', 'NAME'])):
            df_jws = df_jws.drop_duplicates(subset=['VISIT_RECORD_ID_NEW_1', 'NAME'], keep='last')
            print("[Message]: 既往史结构化数据经 两个字段去重 数据量为 {} ".format(df_jws.shape))

        return df_jws

    def grsprocess(self):
        s = select([self.grs_table.c.note_id,
                    self.grs_table.c.person_id,
                    self.grs_table.c.visit_record_id,
                    self.grs_table.c.visit_record_id_new_1,
                    self.grs_table.c.name,
                    self.grs_table.c.value])
        result = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(result, columns=self.getcolumn('P0_NOTE_GRS'))
        print("[Message]: 个人史结构化原始 数据量为 {} ".format(result.shape))

        # drop duplicates in the range of all columns
        df_grs = result[['VISIT_RECORD_ID_NEW_1', 'NAME', 'VALUE']]
        if any(df_grs.duplicated()):
            df_grs.sort_values(by = ['VISIT_RECORD_ID_NEW_1', 'NAME', 'VALUE'], na_position = 'last', inplace = True)
            df_grs = df_grs.drop_duplicates(keep='last')
            print("[Message]: 个人史结构化数据经 全表去重 数据量为 {} ".format(df_grs.shape))

         # there are still duplicates in the range of ['visit_record_id_new_1', 'name']
        if any(df_grs.duplicated(subset=['VISIT_RECORD_ID_NEW_1', 'NAME'])):
            df_grs = df_grs.drop_duplicates(subset=['VISIT_RECORD_ID_NEW_1', 'NAME'], keep='last')
            print("[Message]: 个人史结构化数据经 两个字段去重 数据量为 {} ".format(df_grs.shape))

        return df_grs

    def visitprocess(self):
        s = select([self.visit_table.c.visit_record_id,
                    self.visit_table.c.visit_record_id_new,
                    self.visit_table.c.visit_record_id_new_1,
                    self.visit_table.c.person_id,
                    self.visit_table.c.visit_age,
                    self.visit_table.c.weight,
                    self.visit_table.c.height,
                    self.visit_table.c.los,
                    self.visit_table.c.visit_start_date,
                    self.visit_table.c.visit_end_date,
                    self.visit_table.c.flag,
                    self.visit_table.c.label,
                    self.visit_table.c.gender])
        result = self.db_conn.conn.execute(s).fetchall()
        # adjust the order of columns
        visit_columns = self.getcolumn('P0_VISIT_ZY')
        visit_columns_1 = visit_columns.pop(0)
        visit_columns.append(visit_columns_1)
        result = pd.DataFrame(result, columns=visit_columns)
        print("[Message]: 住院就诊原始 数据量为 {} ".format(result.shape))
        # drop duplicates in the range of all columns
        df_visit = result[['VISIT_RECORD_ID_NEW_1', 'VISIT_AGE', 'GENDER', 'WEIGHT', 'HEIGHT', 'LABEL']]
        if any(df_visit.duplicated(subset=['VISIT_RECORD_ID_NEW_1'])):
            df_visit.sort_values(by = ['VISIT_RECORD_ID_NEW_1', 'VISIT_AGE'], na_position = 'last', inplace = True)
            df_visit = df_visit.drop_duplicates(subset=['VISIT_RECORD_ID_NEW_1'], keep='last')
            print("[Message]: 住院就诊数据经 全表去重 数据量为 {} ".format(df_visit.shape))

        return df_visit

    def meaprocess(self):
        s = select([self.mea_table.c.visit_record_id_new_1,
                    self.mea_table.c.specimen,
                    self.mea_table.c.item_code,
                    self.mea_table.c.item_name,
                    self.mea_table.c.value])
        result = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(result, columns = ['VISIT_RECORD_ID_NEW_1', 'SPECIMEN', 'ITEM_CODE', 'ITEM_NAME', 'VALUE'])
        print("[Message]: 实验室化验原始 数据量为 {} ".format(result.shape))

        if any(result.duplicated()):
            result.sort_values(by = ['VISIT_RECORD_ID_NEW_1', 'ITEM_NAME'], na_position = 'last', inplace = True)
            result = result.drop_duplicates(keep='last')
            print("[Message]: 住院化验数据经 全表去重 数据量为 {} ".format(result.shape))
        # concat the specimen and item_name
        result['NAME'] = result.apply(lambda x: x['ITEM_NAME'] + '-' + x['SPECIMEN'], axis = 1)
        # delete part of data
        a = result.groupby(['VISIT_RECORD_ID_NEW_1', 'NAME']).count()>1
        b = a[a['VALUE'] == True].reset_index()
        c = pd.DataFrame(b['NAME'].value_counts())
        c.reset_index(inplace=True)
        for index, row in c.iterrows():
            df = result[result['NAME'] == row['index']]
            vc = df['ITEM_CODE'].value_counts().reset_index()
            if len(vc) == 1:
                continue
            elif len(vc) > 1 and vc.iloc[0, 1]/vc['ITEM_CODE'].sum() > 0.8:
                result.drop(index = df[df['ITEM_CODE'].isin(vc.iloc[1:, 0])].index.to_list(), inplace=True)
        print("[Message]: 经去除部分异常化验小项后 数据维度为 {}".format(result.shape))
        result = result[['VISIT_RECORD_ID_NEW_1', 'NAME', 'VALUE']]
        if any(result.duplicated(subset=['VISIT_RECORD_ID_NEW_1', 'NAME'])):
            result.sort_values(by = ['VISIT_RECORD_ID_NEW_1', 'NAME'], na_position = 'last', inplace = True)
            result = result.drop_duplicates(subset=['VISIT_RECORD_ID_NEW_1', 'NAME'], keep='last')
            print("[Message]: 住院化验数据经 双字段去重 数据维度为 {} ".format(result.shape))

        return result


    def feature_matrix(self):

        # get the inhospital visit data
        visit = self.visitprocess()
        # get the symptom data after processing
        zz = self.zzprocess()
        # get the grs data
        grs = self.grsprocess()
        # get the jws data
        jws = self.jwsprocess()
        # get the measurement data
        mea = self.meaprocess()

        # print the shape of all data
        print("[message]: the dimensions of visit are {}".format(visit.shape))
        print("[message]: the dimensions of symptom are {}".format(zz.shape))
        print("[message]: the dimensions of jws are {}".format(jws.shape))
        print("[message]: the dimensions of grs are {}".format(grs.shape))
        print("[message]: the dimensions of mea are {}".format(mea.shape))

        # concat the stack table zz, grs, jws, mea
        con = pd.concat([zz, grs, jws, mea], ignore_index = True)
        print("[message]: the dimensions of concated data are {}".format(con.shape))

        # pivot the table
        pi_table = con.pivot(index='VISIT_RECORD_ID_NEW_1', columns='NAME', values='VALUE')
        print("[message]: the dimensions of pivoted data are {}".format(pi_table.shape))

        # merge the all data
        visit = visit.set_index(['VISIT_RECORD_ID_NEW_1'])
        fea_mat = visit.merge(pi_table, left_index=True, right_index=True, how='left')
        print("[message]: the dimensions of feature matrix are {}".format(fea_mat.shape))

        # output the describe of the feature matrix
        base_dir = os.path.dirname(os.path.realpath('__file__'))
        excelpath = os.path.join(base_dir+"/output/", 'feature_matrix_describe.xlsx')
        des = fea_mat.describe(include='all')
        des.to_excel(excelpath)

        return fea_mat

    def drop_col(self, df, col_name, cutoff, type):
        
        # 分别对nan值占比过高和偏振分布的列进行删除
        if type == 'nan':
            cnt = df[col_name].count()
            if (float(cnt) / len(df)) < cutoff:
                df.drop(col_name, axis=1, inplace=True)
        if type == 'maldis':
            vc = pd.DataFrame(df[col_name].value_counts())
            vc.reset_index(inplace=True)
            if len(vc) == 1:
                df.drop(col_name, axis=1, inplace=True)
            elif len(vc) > 1 and (float(vc.iloc[0, 1]/vc[col_name].sum())) > cutoff:
                df.drop(col_name, axis=1, inplace=True)


    def feature_process(self):

        fm = self.feature_matrix()

        # drop the columns that the ratio of nan is very high
        col_list = fm.columns.values.tolist()
        for col in col_list:
            self.drop_col(fm, col, 0.5, type='nan')
        print("[message]: the dimensions of feature matrix are {}".format(fm.shape))

        # drop the columns that the ratio of max value_counts beyond some kind cutoff
        col_list_mvc = fm.columns.values.tolist()
        for col in col_list_mvc:
            self.drop_col(fm, col, 0.9, type='maldis')
        print("[message]: the dimensions of feature matrix are {}".format(fm.shape))       

        # output the new matrix to csv
        base_dir = os.path.dirname(os.path.realpath('__file__'))
        csvpath = os.path.join(base_dir+"/output/", 'p0_feature_l1.csv')
        fm.to_csv(csvpath, header=True, index=True)

        return fm

class LabelComplement():

    # 20210825 code for label process to complete the label from the information of column "diag_name" and to fill in the LABEL4VISIT_II
    def __init__(self, db_name, in_table_name, out_table_name):
        self.db_name = db_name
        self.db_conn = db_connection(db_name)
        self.in_table = Table(in_table_name, self.db_conn.metadata, autoload=True)
        self.out_table = out_table_name

        self.rootDir = os.path.split(os.path.realpath(__file__))[0]
        self.dictpath = os.path.join(self.rootDir, 'config/label_tag.txt')
    
    def read_dict(self):
        bacterial, viral, fungal, parasitic, otherinfec, aif, aid, others = {}, {}, {}, {}, {}, {}, {}, {}

        with codecs.open(self.dictpath, "r", encoding = "utf-8") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                if line.startswith('%') and line[2:] == '病毒性':
                    target_dict = viral
                elif line.startswith('%') and line[2:] == '细菌性':
                    target_dict = bacterial
                elif line.startswith('%') and line[2:] == '真菌性':
                    target_dict = fungal
                elif line.startswith('%') and line[2:] == '寄生虫性':
                    target_dict = parasitic
                elif line.startswith('%') and line[2:] == '其它感染':
                    target_dict = otherinfec
                elif line.startswith('%') and line[2:] == '自身炎症性':
                    target_dict = aif
                elif line.startswith('%') and line[2:] == '自身免疫性':
                    target_dict = aid
                elif line.startswith('%') and line[2:] == '其它':
                    target_dict = others
                else:
                    items = line.split(' ||| ')
                    assert len(items) == 2
                    key = items[0]
                    value = items[1].split()
                    target_dict[key] = value

        return bacterial, viral, fungal, parasitic, otherinfec, aif, aid, others

    def label_process(self):

        s1 = select([self.in_table.c.visit_record_id,
                     self.in_table.c.diag_name,
                     self.in_table.c.diag_append]).\
                          where(self.in_table.c.diag_type == '出院诊断').\
                          where(self.in_table.c.diag_index == '1').\
                          where(self.in_table.c.diag_append != None)
        result = self.db_conn.conn.execute(s1)
        df = pd.DataFrame(result, columns=['visit_record_id', 'diag_name', 'diag_append'])
        print("=====================================")
        print("The number of rows of dataframe is {}".format(len(df)))
        print("=====================================")
        # read dict file
        bacterial, viral, fungal, parasitic, otherinfec, aif, aid, others = self.read_dict()

        iter_count = 1
        df_res = pd.DataFrame(columns=['visit_record_id', 'diag_name', 'diag_append', 'label'])
        for index, rows in df.iterrows():
            # exclude the concurrent infection first
            if ('合并' in rows['diag_append'] or '混合' in rows['diag_append'] or '双重' in rows['diag_append']) and '感染' in rows['diag_append']:
                df_res = df_res.append({'visit_record_id':rows['visit_record_id'], 'diag_name': rows['diag_name'], \
                                            'diag_append':rows['diag_append'], 'label': '合并感染'}, ignore_index=True)
            else:
                label = {}
                for row in re.split(r'[，。,;；、 ]', rows['diag_append']):
                    if '待排' in row or '？' in row or '?' in row or '不能排除' in row or '除外' in row:
                        continue
                    for key, values in bacterial.items():
                        for value in values:
                            if value in row:
                                label["bacterial"] =  '1'
                                break
                    for key, values in viral.items():
                        for value in values:
                            if value in row:
                                label["viral"] =  '1'
                                break
                    for key, values in fungal.items():
                        for value in values:
                            if value in row:
                                label["fungal"] =  '1'
                                break
                    for key, values in parasitic.items():
                        for value in values:
                            if value in row:
                                label["parasitic"] =  '1'
                                break
                    for key, values in otherinfec.items():
                        for value in values:
                            if value in row:
                                label["otherinfec"] =  '1'
                                break
                    for key, values in aif.items():
                        for value in values:
                            if value in row:
                                label["aif"] =  '1'
                                break
                    for key, values in aid.items():
                        for value in values:
                            if value in row:
                                label["aid"] =  '1'
                                break
                    for key, values in others.items():
                        for value in values:
                            if value in row:
                                label["others"] =  '1'
                                break
                if len(label) == 1:
                    df_res = df_res.append({'visit_record_id':rows['visit_record_id'], 'diag_name': rows['diag_name'], \
                                            'diag_append':rows['diag_append'], 'label': list(label.keys())[0]}, ignore_index=True)
                elif len(label) > 1:
                    df_res = df_res.append({'visit_record_id':rows['visit_record_id'], 'diag_name': rows['diag_name'], \
                                            'diag_append':rows['diag_append'], 'label': '合并'}, ignore_index=True)
                else:
                    df_res = df_res.append({'visit_record_id':rows['visit_record_id'], 'diag_name': rows['diag_name'], \
                                            'diag_append':rows['diag_append'], 'label': ''}, ignore_index=True)
            
            if len(df_res) == 500 and iter_count < round(len(df)/500) + 1:
                FileUtils.df2db(self.db_name, self.out_table, df_res)
                print("=====================================")
                print("The number of rows inserted is 500*{}".format(iter_count))
                print("=====================================")
                df_res.drop(df_res.index, inplace=True)
                iter_count += 1
            if iter_count > round(len(df)/500):
                FileUtils.df2db(self.db_name, self.out_table, df_res)
                print("=====================================")
                print("The number of rows inserted is {}".format(500 * round(len(df)/500) + iter_count-round(len(df)/500)))
                print("=====================================")
                df_res.drop(df_res.index, inplace=True)
                iter_count += 1

class LabelCV():

    def __init__(self, db_name, label_table_name, drug_table_name, db_name_sys='fuo_sys'):

        self.db_conn = db_connection(db_name)
        self.db_conn_sys = db_connection(db_name_sys)
        self.label_table_name = label_table_name
        self.drug_table_name = drug_table_name
        self.label_table = Table(label_table_name, self.db_conn.metadata, autoload=True)
        self.drug_table = Table(drug_table_name, self.db_conn.metadata, autoload=True)
        self.column_query = Table('ALL_TAB_COLUMNS', self.db_conn_sys.metadata, autoload = True)

        self.rootDir = os.path.split(os.path.realpath(__file__))[0]
        self.logpath = os.path.join(self.rootDir, 'log/process_drug_label.txt')
        self.drug_dict = {'细菌':'A', '病毒':'B','真菌':'C','寄生虫':'D','血液恶性肿瘤':'E','实体恶性肿瘤':'F','化疗药':'G','自身免疫性':'H','自身炎症性':'I','细菌或梅毒螺旋体或衣原体':'J','不典型病原体，细菌梅毒支原体衣原体':'L'}

    def get_columns(self, table_name):

        #get column name list
        s = select([self.column_query.c.column_name]).where(self.column_query.c.table_name == table_name)
        table_column = self.db_conn_sys.conn.execute(s).fetchall()
        column = [col for (col,) in table_column]
        column.reverse()

        return column

    def get_vir_list(self):

        # get list of visit_record_id
        s1 = select([self.label_table.c.visit_record_id])
        vri = self.db_conn.conn.execute(s1).fetchall()
        result = [v for (v,) in vri]
        print("-----------------------------------------------------------------------")
        print("The number of visit_record_id of {0} is {1}".format(self.label_table_name.upper(), len(result)))
        print("-----------------------------------------------------------------------")

        return result

    def get_drug_data(self, vri):
        
        # get the data of DRUG table by the visit_record_id
        s2 = select([self.drug_table.c.drug_record_id,
                     self.drug_table.c.visit_record_id,
                     self.drug_table.c.ypmc,
                     self.drug_table.c.drug_label,
                     self.drug_table.c.drug_start_date,
                     self.drug_table.c.route]).\
                        where(self.drug_table.c.route != '退药').\
                            where(self.drug_table.c.visit_record_id == vri).\
                                where(self.drug_table.c.drug_label != None)
        res = self.db_conn.conn.execute(s2).fetchall()
        result = pd.DataFrame(res, columns = ['drug_record_id', 'visit_record_id', 'ypmc', 'drug_label', 'drug_start_date', 'route'])

        return result

    def get_label_data(self, vri):
        
        # get the data of LABEL4VISIT table by the visit_record_id
        s3 = select([self.label_table.c.visit_record_id,
                     self.label_table.c.source_i,
                     self.label_table.c.source_ii,
                     self.label_table.c.source_liu,
                     self.label_table.c.source_iii,
                     self.label_table.c.source_iii_flag,
                     self.label_table.c.source_union,
                     self.label_table.c.comment_tag]).\
                        where(self.label_table.c.visit_record_id == vri)
        res = self.db_conn.conn.execute(s3).fetchall()
        result = pd.DataFrame(res, columns = ['visit_record_id','source_i','source_ii','source_liu','source_iii', 'source_iii_flag', 'source_union', 'comment_tag'])

        return result

    def process_drug_label(self):

        count = 0
        vri = self.get_vir_list()

        for v in vri:

            print("-----------------------------------------------------------------------")
            print("The data of visit_record_id of {} is processing".format(v))

            count += 1
            drug_data = self.get_drug_data(v)

            if not drug_data.empty:
                drug_data.sort_values(by='drug_start_date', ascending=True, inplace=True)
                label_list = [x for x in drug_data['drug_label'].tolist()]
                for i in range(len(label_list)):
                    label_list[i] = self.drug_dict[label_list[i]]
                
                # turn the list into char and update into column "SOURCE_UNION"
                label_char = "".join(label_list)
                sql = self.label_table.update().\
                    where(self.label_table.c.visit_record_id == v).\
                        values(source_iii_tag=label_char)
                self.db_conn.conn.execute(sql)

                # fill the column source_iii and source_iii_flag
                count_dict = Counter(label_list)

                if set(label_list).issubset(['A', 'B', 'C', 'D', 'J', 'L']):
                    # update the parasitic infectious
                    if 'D' in label_list:
                        sql = self.label_table.update().\
                            where(self.label_table.c.visit_record_id == v).\
                                values(source_iii='1-4', source_iii_flag='type-i')
                        self.db_conn.conn.execute(sql)

                    # update the other infectious
                    if 'J' in label_list:
                        if len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-5', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                        elif max(count_dict, key=count_dict.get) == 'J' or label_list[-1] == 'J':
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-5', source_iii_flag='type-ii')
                            self.db_conn.conn.execute(sql)
                    
                    # update the other infectious
                    if 'L' in label_list:
                        if len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-5', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                        elif max(count_dict, key=count_dict.get) == 'L' or label_list[-1] == 'L':
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-5', source_iii_flag='type-ii')
                            self.db_conn.conn.execute(sql)

                    # update the fungal infectious
                    if 'C' in label_list:
                        if len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-3', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                        elif max(count_dict, key=count_dict.get) == 'C' or label_list[-1] == 'C':
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-3', source_iii_flag='type-ii')
                            self.db_conn.conn.execute(sql)
                    
                    # update the viral infectious
                    if 'B' in label_list:
                        if len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-2', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                        elif max(count_dict, key=count_dict.get) == 'B' or label_list[-1] == 'B':
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-2', source_iii_flag='type-ii')
                            self.db_conn.conn.execute(sql)

                    # update the bacterial infectious
                    if 'A' in label_list:
                        if len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-1', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                        elif max(count_dict, key=count_dict.get) == 'A' or label_list[-1] == 'A':
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1-1', source_iii_flag='type-ii')
                            self.db_conn.conn.execute(sql)

                    # update the rest of infectious disease, but undifferentiated        
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='1', source_iii_flag='type-i')
                    self.db_conn.conn.execute(sql)

                    print("The total number of visit_record_id being processed is {}".format(count))
                    print("-----------------------------------------------------------------------")
                    
                else:
                    # update the NIID type
                    if set(label_list).issubset(['A', 'B', 'C', 'D', 'J', 'L', 'H', 'I']):
                        # update the bacterial infectious
                        if 'I' in label_list:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='2-1-2', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)

                        if 'H' in label_list and len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='2-1-1', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                    # update the neoplastic type
                    else:
                        if 'E' in label_list and len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='2-2-1', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                        if 'F' in label_list and len(set(label_list)) == 1:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='2-2-2', source_iii_flag='type-i')
                            self.db_conn.conn.execute(sql)
                        if 'E' in label_list or 'F' in label_list or 'G' in label_list:
                            sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    where(self.label_table.c.source_iii == None).\
                                        values(source_iii='2-2', source_iii_flag='type-ii')
                            self.db_conn.conn.execute(sql)

                    print("The total number of visit_record_id being processed is {}".format(count))
                    print("-----------------------------------------------------------------------")

            else:
                # flag the visit_record_id whose drug data is None
                sql = self.label_table.update().\
                    where(self.label_table.c.visit_record_id == v).\
                        values(source_iii='无药')
                self.db_conn.conn.execute(sql)
                
                f = open(self.logpath, 'a')
                f.write("[error] the {0} data of {1} is None".format(self.drug_table_name, v)+ "\n")
                f.close()
                continue
    
    def union_label(self):
        
        count = 0
        vri = self.get_vir_list()

        for v in vri:

            print("-----------------------------------------------------------------------")
            print("The label of visit_record_id of {} is processing".format(v))

            count += 1
            label_data = self.get_label_data(v)

            if label_data.iloc[0].at['comment_tag']=='fuo':
                if label_data.iloc[0].at['source_ii'] != None:
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_ii'])
                    self.db_conn.conn.execute(sql)
                elif label_data.iloc[0].at['source_iii'] != None and label_data.iloc[0].at['source_iii'] != '无药':
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_iii'])
                    self.db_conn.conn.execute(sql)
                print("The total number of visit_record_id being processed is {}".format(count))
                print("-----------------------------------------------------------------------")

            elif label_data.iloc[0].at['comment_tag'] == 'notsure_liu':
                if  label_data.iloc[0].at['source_liu'] != None:
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_liu'])
                    self.db_conn.conn.execute(sql)
                elif label_data.iloc[0].at['source_ii'] != None:
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_ii'])
                    self.db_conn.conn.execute(sql)
                elif label_data.iloc[0].at['source_i'] == None and label_data.iloc[0].at['source_iii'] != '无药':
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_iii'])
                    self.db_conn.conn.execute(sql)
                elif label_data.iloc[0].at['source_i'] != None:
                    if label_data.iloc[0].at['source_iii'] == '无药' or label_data.iloc[0].at['source_iii'] == None:
                        sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_i'])
                        self.db_conn.conn.execute(sql)
                    elif label_data.iloc[0].at['source_i'] == '1':
                        sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_iii'])
                        self.db_conn.conn.execute(sql)
                    elif label_data.iloc[0].at['source_i'] == '1-1':
                        sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_iii'])
                        self.db_conn.conn.execute(sql)
                    else:
                        sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_i'])
                        self.db_conn.conn.execute(sql)
                print("The total number of visit_record_id being processed is {}".format(count))
                print("-----------------------------------------------------------------------")
            # deal with the data without "comment_tag"
            else:
                if  label_data.iloc[0].at['source_liu'] != None:
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_liu'])
                    self.db_conn.conn.execute(sql)

                elif label_data.iloc[0].at['source_ii'] != None:
                    sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_ii'])
                    self.db_conn.conn.execute(sql)
                elif label_data.iloc[0].at['source_i'] == None and label_data.iloc[0].at['source_iii'] != '无药':
                    sql = self.label_table.update().\
                                    where(self.label_table.c.visit_record_id == v).\
                                        values(source_union=label_data.iloc[0].at['source_iii'])
                    self.db_conn.conn.execute(sql)
                elif label_data.iloc[0].at['source_i'] != None:
                    if label_data.iloc[0].at['source_iii'] == '无药' or label_data['source_iii'] is None:
                        sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_i'])
                        self.db_conn.conn.execute(sql)
                    elif label_data.iloc[0].at['source_i'] == '1':
                        sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_iii'])
                        self.db_conn.conn.execute(sql)
                    else:
                        sql = self.label_table.update().\
                                where(self.label_table.c.visit_record_id == v).\
                                    values(source_union=label_data.iloc[0].at['source_i'])
                        self.db_conn.conn.execute(sql)
                print("The total number of visit_record_id being processed is {}".format(count))
                print("-----------------------------------------------------------------------")

    def union_label_leak_filling(self):
        
        # leak filling the label of source_union='1'
        s1 = select([self.label_table.c.visit_record_id]).where(self.label_table.c.source_union=='1')
        vri = self.db_conn.conn.execute(s1).fetchall()
        result = [v for (v,) in vri]

        for i in result:
            s2 = select([self.label_table.c.visit_record_id,
                        self.label_table.c.source_i,
                        self.label_table.c.source_ii,
                        self.label_table.c.source_liu,
                        self.label_table.c.source_iii,
                        self.label_table.c.source_iii_flag,
                        self.label_table.c.source_union,
                        self.label_table.c.comment_tag]).\
                        where(self.label_table.c.visit_record_id == i)
            res = self.db_conn.conn.execute(s2).fetchall()
            df = pd.DataFrame(res, columns = ['visit_record_id','source_i','source_ii','source_liu','source_iii', 'source_iii_flag', 'source_union', 'comment_tag'])

            if df.iloc[0].at['source_ii'] != None:
                sql = self.label_table.update().\
                    where(self.label_table.c.visit_record_id == i).\
                        values(source_union=df.iloc[0].at['source_ii'])
                self.db_conn.conn.execute(sql)
            elif df.iloc[0].at['source_i'] != None and df.iloc[0].at['source_i'] != '1' and df.iloc[0].at['source_i'] != '2':
                sql = self.label_table.update().\
                    where(self.label_table.c.visit_record_id == i).\
                        values(source_union=df.iloc[0].at['source_i'])
                self.db_conn.conn.execute(sql)
            elif df.iloc[0].at['source_iii'] != None and df.iloc[0].at['source_iii'] != '无药' and df.iloc[0].at['source_i'] != '2-2':
                sql = self.label_table.update().\
                    where(self.label_table.c.visit_record_id == i).\
                        values(source_union=df.iloc[0].at['source_iii'])
                self.db_conn.conn.execute(sql)

        # leak filling the label of source_union='2-2'
        s3 = select([self.label_table.c.visit_record_id]).where(self.label_table.c.source_union=='2-2')
        vri = self.db_conn.conn.execute(s3).fetchall()
        result = [v for (v,) in vri]

        for i in result:
            s4 = select([self.label_table.c.visit_record_id,
                        self.label_table.c.source_i,
                        self.label_table.c.source_ii,
                        self.label_table.c.source_liu,
                        self.label_table.c.source_iii,
                        self.label_table.c.source_iii_flag,
                        self.label_table.c.source_union,
                        self.label_table.c.comment_tag]).\
                        where(self.label_table.c.visit_record_id == i)
            res = self.db_conn.conn.execute(s4).fetchall()
            df = pd.DataFrame(res, columns = ['visit_record_id','source_i','source_ii','source_liu','source_iii', 'source_iii_flag', 'source_union', 'comment_tag'])
            
            if df.iloc[0].at['source_i'] == '2-2-2':
                sql = self.label_table.update().\
                    where(self.label_table.c.visit_record_id == i).\
                        values(source_union=df.iloc[0].at['source_i'])
                self.db_conn.conn.execute(sql)

class MeasurementProcess():
    """
    split the categorical variables and numeric variables, and seperately deal with the abnormal value,
    and then insert the categorical data back into the table "P1_MEASUREMENT_CAT"
    """
    def __init__(self, db_name, in_table_name, out_table_name, out_table_name_2, db_name_sys='fuo_sys'):

        self.db_conn = db_connection(db_name)
        self.db_conn_sys = db_connection(db_name_sys)
        self.in_table_name = in_table_name
        self.out_table_name = out_table_name
        self.out_table_name_2 = out_table_name_2
        self.in_table = Table(in_table_name, self.db_conn.metadata, autoload=True)
        self.out_table = Table(out_table_name, self.db_conn.metadata, autoload=True)
        self.out_table_2 = Table(out_table_name_2, self.db_conn.metadata, autoload=True)
        self.column_query = Table('ALL_TAB_COLUMNS', self.db_conn_sys.metadata, autoload = True)

        self.rootDir = os.path.split(os.path.realpath(__file__))[0]
        self.logpath = os.path.join(self.rootDir, 'log/process_measurement.txt')

    def get_columns(self, table_name):

        #get column name list
        s = select([self.column_query.c.column_name]).where(self.column_query.c.table_name == table_name)
        table_column = self.db_conn_sys.conn.execute(s).fetchall()
        column = [col for (col,) in table_column]
        column.reverse()

        return column

    def get_data(self):

        s = select([self.in_table.c.lab_id,
                    self.in_table.c.person_id,
                    self.in_table.c.visit_record_id_1,
                    self.in_table.c.visit_record_id_new_1,
                    self.in_table.c.specimen,
                    self.in_table.c.group_measurement_name,
                    self.in_table.c.item_code,
                    self.in_table.c.item_name,
                    self.in_table.c.item_alais,
                    self.in_table.c.value,
                    self.in_table.c.unit,
                    self.in_table.c.range_upper,
                    self.in_table.c.range_lower])
        re = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(re, columns = ['lab_id','person_id','visit_record_id_1','visit_record_id_new_1','specimen','group_measurement_name','item_code','item_name','item_alais','value','unit','range_upper','range_lower'])

        return result

    def mapping_df_types(self, df):
        """
        INPUT: df(要插回数据库的DataFrame数据)
        OUTPUT: dtypedict(指定每个字段要插入到数据库的字段类型)
        """
        dtypedict = {}

        for i, j in zip(df.columns, df.dtypes):
            if str(i) == "LAB_ID":
                dtypedict.update({i: VARCHAR2(50)})
            if str(i) =="PERSON_ID":
                dtypedict.update({i: VARCHAR2(54)})
            if str(i) == "VISIT_RECORD_ID_1":
                dtypedict.update({i: VARCHAR2(53)})
            if str(i) == "VISIT_RECORD_ID_NEW_1":
                dtypedict.update({i: VARCHAR2(53)})
            if str(i) == "SPECIMEN":
                dtypedict.update({i: VARCHAR2(100)})
            if str(i) == "GROUP_MEASUREMENT_NAME":
                dtypedict.update({i: VARCHAR2(2000)})
            if str(i) == "ITEM_CODE":
                dtypedict.update({i: VARCHAR2(22)})
            if str(i) == "ITEM_NAME":
                dtypedict.update({i: VARCHAR2(100)})
            if str(i) == "ITEM_ALAIS":
                dtypedict.update({i: VARCHAR2(100)})
            if str(i) == "VALUE":
                dtypedict.update({i: VARCHAR2(200)})
            if str(i) == "UNIT":
                dtypedict.update({i: VARCHAR2(20)})
            if str(i) == "RANGE_UPPER":
                dtypedict.update({i: VARCHAR2(20)})
            if str(i) == "RANGE_LOWER":
                dtypedict.update({i: VARCHAR2(20)})
        return dtypedict

    def insertdata_cat(self, df_data, dtypedict_status=True):
        """
        INPUT: df_data(要插回数据库的数据)
        """
        #获取定义的字典
        if dtypedict_status:
            dtypedict = self.mapping_df_types(df_data)
        else:
            dtypedict = None
        #写入数据库
        df_data.to_sql(self.out_table_name_2, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)

    def insertdata_val(self, df_data, dtypedict_status=True):
        """
        INPUT: df_data(要插回数据库的数据)
        """
        #获取定义的字典
        if dtypedict_status:
            dtypedict = self.mapping_df_types(df_data)
        else:
            dtypedict = None
        #写入数据库
        df_data.to_sql(self.out_table_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)

    def process_measurement(self):

        df_mea = self.get_data()
        print("-----------------------------------------------------------------------")
        print("The original shape of measurement data is {}".format(df_mea.shape))
        print("-----------------------------------------------------------------------")

        # split the data
        df_urine = df_mea.loc[df_mea['specimen']=='尿液']
        df_shit = df_mea.loc[df_mea['specimen']=='大便']
        df_blood = df_mea.loc[df_mea['specimen'].isin(['血液', '血浆', '血清'])]
        df_blood_cat = df_mea.loc[df_mea['item_name'].isin(['乙肝表面抗原', '梅毒螺旋体抗体', '丙肝抗体IgG', 'ABO血型', 'Rh(D)血型'])]
        df_blood_val = df_blood[~ df_blood['item_name'].isin(df_blood_cat['item_name'])]
        print("-----------------------------------------------------------------------")
        print("The shapes of measurement data of urine, shit, blood_cat, blood_val are respectively {0}, {1}, {2}, {3}".format(df_urine.shape, df_shit.shape, df_blood_cat.shape, df_blood_val.shape))
        print("-----------------------------------------------------------------------")

        # process the df_blood_val test data
        df_blood_val['value'] = df_blood_val['value'].replace('[^\d^\.]+', '', regex=True)
        df_blood_val['value'] = df_blood_val['value'].replace('\.{2,100}', '.', regex=True)
        df_blood_val.loc[df_blood_val['value'] == '.', ['value']] = None

        # process the df_blood_cat test data
        df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']].replace('(阳性(>|<)*[0-9]+(.)*[0-9]*)', '1', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']].replace('(阴性(>|<)*[0-9]+(.)*[0-9]*)', '0', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']].replace('(>250.00)', '1', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']].replace('弱阳性', '1', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='乙肝表面抗原', ['value']].replace('(重度溶血|临界|建议复检)', '', regex=True)
        df_blood_cat.loc[(df_blood_cat['item_name']=='乙肝表面抗原') & (df_blood_cat['value'] == '阳性'), ['value']] = '1'
        df_blood_cat.loc[(df_blood_cat['item_name']=='乙肝表面抗原') & (df_blood_cat['value'] == '阴性'), ['value']] = '0'

        df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']].replace('阴性(>|<)*[0-9]+(.)*[0-9]*', '阴性', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']].replace('阳性(>|<)*[0-9]+(.)*[0-9]*', '阳性', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']].replace('弱阳性', '阳性', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='梅毒螺旋体抗体', ['value']].replace('(临界值)', '', regex=True)
        
        df_blood_cat.loc[df_blood_cat['item_name']=='丙肝抗体IgG', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='丙肝抗体IgG', ['value']].replace('(临界|复检)', '', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='丙肝抗体IgG', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='丙肝抗体IgG', ['value']].replace('弱阳性', '阳性', regex=True)

        df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']].replace('[^A-Za-z0-9]*O+[^A-Za-z0-9]*', 'O', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']].replace('[^A-Za-z0-9]*B+[^A-Za-z0-9]*', 'B', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']].replace('[^A-Za-z0-9]*A+[^A-Za-z0-9]*', 'A', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']].replace('[^A-Za-z0-9]*(AB)+[^A-Za-z0-9]*', 'AB', regex=True)
        df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='ABO血型', ['value']].replace('(建议血液中心|自凝)', '', regex=True)
        
        df_blood_cat.loc[df_blood_cat['item_name']=='Rh(D)血型', ['value']] = df_blood_cat.loc[df_blood_cat['item_name']=='Rh(D)血型', ['value']].replace('(自凝)', '', regex=True)
        
        # process the df_urine test data
        df_urine.loc[df_urine['value'] == '.', ['value']] = None
        df_urine.loc[df_urine['item_name']=='酮体', ['value']] = df_urine.loc[df_urine['item_name']=='酮体', ['value']].replace('(neg\(-\)|阴性)', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='酮体', ['value']] = df_urine.loc[df_urine['item_name']=='酮体', ['value']].replace('(\+\+\+\+\(OVER\)|\+\+\+\+\(15\))', '4+', regex=True)
        df_urine.loc[df_urine['item_name']=='酮体', ['value']] = df_urine.loc[df_urine['item_name']=='酮体', ['value']].replace('(\+\+\+\(8\)|\+\+\+\(7.4\)|\+\+\+\(10\)|7.8\(\+\+\+\))', '3+', regex=True)
        df_urine.loc[df_urine['item_name']=='酮体', ['value']] = df_urine.loc[df_urine['item_name']=='酮体', ['value']].replace('(\+\+\(4\)|\+\+\(2.8\)|\+\+\(6\)|3.9\(\+\+\)|\+\+)', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='酮体', ['value']] = df_urine.loc[df_urine['item_name']=='酮体', ['value']].replace('(\+\(1\)|\+\(2\)|\+\(0.93\)|1.5\(\+\))', '1+', regex=True)        
        df_urine.loc[df_urine['item_name']=='酮体', ['value']] = df_urine.loc[df_urine['item_name']=='酮体', ['value']].replace('(0.5\(\+\-\)|\±\(\)|5\(\+\-\))', '±', regex=True)
        df_urine.loc[(df_urine['item_name']=='酮体') & (df_urine['value'] == '+'), ['value']] = '1+'
        
        df_urine.loc[df_urine['item_name']=='隐血', ['value']] = df_urine.loc[df_urine['item_name']=='隐血', ['value']].replace('(\±\(0.3\)|10\(\+\-\)|\+\-)', '±', regex=True)
        df_urine.loc[df_urine['item_name']=='隐血', ['value']] = df_urine.loc[df_urine['item_name']=='隐血', ['value']].replace('neg\(\-\)', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='隐血', ['value']] = df_urine.loc[df_urine['item_name']=='隐血', ['value']].replace('(\+\+\+\+\(7.5\)|\+\+\+\(7.5\)|\+\+\+\(10.0\)|\+\+\+\(OVER\)|200\(\+\+\+\)|250\(\+\+\+\))', '3+', regex=True)
        df_urine.loc[df_urine['item_name']=='隐血', ['value']] = df_urine.loc[df_urine['item_name']=='隐血', ['value']].replace('(\+\+\(2.0\)|\+\+\(1.5\)|\+\+\(5.0\)|80\(\+\+\)|50\(\+\+\)|\+\+|75)', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='隐血', ['value']] = df_urine.loc[df_urine['item_name']=='隐血', ['value']].replace('(\+\(0.6\)|\+\(1.0\)|25\(\+\)|10\(\+\))', '1+', regex=True)
        df_urine.loc[(df_urine['item_name']=='隐血') & (df_urine['value'] == '+'), ['value']] = '1+'
        
        df_urine.loc[df_urine['item_name']=='蛋白质', ['value']] = df_urine.loc[df_urine['item_name']=='蛋白质', ['value']].replace('neg\(\-\)', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='蛋白质', ['value']] = df_urine.loc[df_urine['item_name']=='蛋白质', ['value']].replace('(\±\(0.2\)|\±\(0.1\)|\±\(0.15\)|0.1\(\+\-\)|\+\-\(0.3\))', '±', regex=True)
        df_urine.loc[df_urine['item_name']=='蛋白质', ['value']] = df_urine.loc[df_urine['item_name']=='蛋白质', ['value']].replace('(\+\+\+\+\(OVER\)|\+\+\+\+\(10\)|20\(\+\+\+\+\)|\+\+\+\+\(10.0\))', '4+', regex=True)
        df_urine.loc[df_urine['item_name']=='蛋白质', ['value']] = df_urine.loc[df_urine['item_name']=='蛋白质', ['value']].replace('(\+\+\+\(3.0\)|3.0\(\+\+\+\)|\+\+\+\(6.0\))', '3+', regex=True)
        df_urine.loc[df_urine['item_name']=='蛋白质', ['value']] = df_urine.loc[df_urine['item_name']=='蛋白质', ['value']].replace('(\+\+\(1.0\)|\+\+\(2.0\)|1.0\(\+\+\)|\+\+|100\(\+\+\))', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='蛋白质', ['value']] = df_urine.loc[df_urine['item_name']=='蛋白质', ['value']].replace('(\+\(0.3\)|\+\(0.5\)|\+\(0.7\)|0.3\(\+\)|30\(\+\))', '1+', regex=True)      
        
        df_urine.loc[df_urine['item_name']=='亚硝酸盐', ['value']] = df_urine.loc[df_urine['item_name']=='亚硝酸盐', ['value']].replace('neg\(\-\)', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='亚硝酸盐', ['value']] = df_urine.loc[df_urine['item_name']=='亚硝酸盐', ['value']].replace('(\+\+)', '2+', regex=True)
        df_urine.loc[(df_urine['item_name']=='亚硝酸盐') & (df_urine['value'] == '+'), ['value']] = '1+'
                
        df_urine.loc[df_urine['item_name']=='尿胆原', ['value']] = df_urine.loc[df_urine['item_name']=='尿胆原', ['value']].replace('(正常|neg\(\-\)|norm\(\+\-\))', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='尿胆原', ['value']] = df_urine.loc[df_urine['item_name']=='尿胆原', ['value']].replace('(\+\+\+\+\(OVER\)|\+\+\+\+\(202\))', '4+', regex=True)
        df_urine.loc[df_urine['item_name']=='尿胆原', ['value']] = df_urine.loc[df_urine['item_name']=='尿胆原', ['value']].replace('(\+\+\+\(140\)|\+\+\+\(200\)|\+\+\+\(135\)|130\(\+\+\+\))', '3+', regex=True)
        df_urine.loc[df_urine['item_name']=='尿胆原', ['value']] = df_urine.loc[df_urine['item_name']=='尿胆原', ['value']].replace('(\+\+\(70\)|\+\+\(100\)|\+\+\(68\)|66\(\+\+\))', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='尿胆原', ['value']] = df_urine.loc[df_urine['item_name']=='尿胆原', ['value']].replace('(\+\(34\)|\+\(2.0\)|\+\(50\)|33\(\+\))', '1+', regex=True)
        
        df_urine.loc[df_urine['item_name']=='胆红素', ['value']] = df_urine.loc[df_urine['item_name']=='胆红素', ['value']].replace('neg\(\-\)', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='胆红素', ['value']] = df_urine.loc[df_urine['item_name']=='胆红素', ['value']].replace('(\+\+\+\(34\)|\+\+\(34\)|\+\+\(50\)|\+\+\(70\)|\+\+\+\(100\)|\+\+\+\(170\)|\+\+\+\(140\)|\+\+\+\+\(OVER\)|\+\+\+\+|\+\+\+)', '3+', regex=True)
        df_urine.loc[df_urine['item_name']=='胆红素', ['value']] = df_urine.loc[df_urine['item_name']=='胆红素', ['value']].replace('(\+\+\(17\)|\+\(17\)|\+\+)', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='胆红素', ['value']] = df_urine.loc[df_urine['item_name']=='胆红素', ['value']].replace('(\+\(8.5\)|\+\(8.6\))', '1+', regex=True)
        df_urine.loc[(df_urine['item_name']=='胆红素') & (df_urine['value'] == '+'), ['value']] = '1+'
        
        df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']] = df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']].replace('(neg\(\-\)|正常|\±\(1.7\))', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']] = df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']].replace('(\±\(2.8\)|\+\(3.9\)|2.8\(\+\-\)|5.5\(\+\-\))', '±', regex=True)
        df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']] = df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']].replace('(\+\+\+\(28\)|\+\+\+\+\(56\)|28\(\+\+\+\)|28\(\+\+\)|55\(\+\+\+\)|56\(\+\+\+\+\))', '3+', regex=True)
        df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']] = df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']].replace('(\+\+\+\+\(OVER\)|\+\+\+\+\(111\)|\+\+\+\+)', '4+', regex=True)
        df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']] = df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']].replace('(\+\+\(14\)|14\(\+\)|\+\+\+\(17\))', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']] = df_urine.loc[df_urine['item_name']=='葡萄糖', ['value']].replace('(\+\(5.6\)|\+\+\(8.3\)|\+\+\(11\)|11\(\+\+\)|5.6\(\+\)|100\(\+\-\))', '1+', regex=True)
        
        df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']] = df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']].replace('(neg\(\-\)|阴性|正常)', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']] = df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']].replace('(15\(\+\-\)|10\(\+\-\)|\±\(0.15\)|\+\-)', '±', regex=True)
        df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']] = df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']].replace('(500\(\+\+\+\)|\+\+\+\(500\)|500|\+\+\+)', '3+', regex=True)
        df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']] = df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']].replace('(\+\+\(75\)|250|125\(\+\+\)|75\(\+\+\)|\+\+|70\(\+\)|75)', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']] = df_urine.loc[df_urine['item_name']=='白细胞酯酶', ['value']].replace('(\+\(25\)|25\(\+\)|\+\（25\）|25)', '1+', regex=True)
        df_urine.loc[(df_urine['item_name']=='白细胞酯酶') & (df_urine['value'] == '+'), ['value']] = '1+'
        
        df_urine.loc[df_urine['item_name']=='比重', ['value']] = df_urine.loc[df_urine['item_name']=='比重', ['value']].replace('(UNDER|OVER|LAMPL\.)', '', regex=True)
        df_urine.loc[df_urine['item_name']=='比重', ['value']] = df_urine.loc[df_urine['item_name']=='比重', ['value']].replace('(<1.03)', '1.03', regex=True)

        df_urine.loc[df_urine['item_name']=='浊度', ['value']] = df_urine.loc[df_urine['item_name']=='浊度', ['value']].replace('(\-\(清\)|清)', '-', regex=True)
        df_urine.loc[df_urine['item_name']=='浊度', ['value']] = df_urine.loc[df_urine['item_name']=='浊度', ['value']].replace('(\+\+\(混浊\)|\+\+|混浊)', '2+', regex=True)
        df_urine.loc[df_urine['item_name']=='浊度', ['value']] = df_urine.loc[df_urine['item_name']=='浊度', ['value']].replace('(微浊|\+\(微浊\))', '1+', regex=True)
        df_urine.loc[(df_urine['item_name']=='浊度') & (df_urine['value'] == '+'), ['value']] = '1+'

        df_urine.loc[df_urine['item_name']=='颜色', ['value']] = df_urine.loc[df_urine['item_name']=='颜色', ['value']].replace('(亮棕色|棕色|深棕色|暗棕色)', '棕色', regex=True)
        df_urine.loc[df_urine['item_name']=='颜色', ['value']] = df_urine.loc[df_urine['item_name']=='颜色', ['value']].replace('(橙色|暗橙色|亮橙色)', '橙色', regex=True)
        df_urine.loc[df_urine['item_name']=='颜色', ['value']] = df_urine.loc[df_urine['item_name']=='颜色', ['value']].replace('(红色|亮红色|暗红色)', '红色', regex=True)
        df_urine.loc[df_urine['item_name']=='颜色', ['value']] = df_urine.loc[df_urine['item_name']=='颜色', ['value']].replace('(黄褐色)', '黄色', regex=True)
        df_urine.loc[df_urine['item_name']=='颜色', ['value']] = df_urine.loc[df_urine['item_name']=='颜色', ['value']].replace('(绿色|乳白色)', '其他', regex=True)

        df_urine.loc[df_urine['item_name']=='电导率', ['value']] = df_urine.loc[df_urine['item_name']=='电导率', ['value']].replace('(-----\.--)', '', regex=True)
        df_urine.loc[df_urine['item_name']=='类酵母菌', ['value']] = df_urine.loc[df_urine['item_name']=='类酵母菌', ['value']].replace('[^\d^\.]+', '', regex=True)
        df_urine.loc[df_urine['item_name']=='病理管型', ['value']] = df_urine.loc[df_urine['item_name']=='病理管型', ['value']].replace('[^\d^\.]+', '', regex=True)
        df_urine.loc[df_urine['item_name']=='结晶', ['value']] = df_urine.loc[df_urine['item_name']=='结晶', ['value']].replace('[^\d^\.]+', '', regex=True)
        df_urine.loc[df_urine['item_name']=='管型', ['value']] = df_urine.loc[df_urine['item_name']=='管型', ['value']].replace('(-----\.--)', '', regex=True)

        df_urine.loc[df_urine['item_name']=='细菌', ['value']] = df_urine.loc[df_urine['item_name']=='细菌', ['value']].replace('(\*\*\*\*\*\.\*\*|-----\.--)', '', regex=True)
        df_urine.loc[df_urine['item_name']=='细菌', ['value']] = df_urine.loc[df_urine['item_name']=='细菌', ['value']].replace('[^\d^\.]+', '', regex=True)

        df_urine.loc[df_urine['item_name']=='上皮细胞', ['value']] = df_urine.loc[df_urine['item_name']=='上皮细胞', ['value']].replace('(-----\.--|([0-9](-)[0-9]+))', '0', regex=True)
        df_urine.loc[(df_urine['item_name']=='上皮细胞') & (df_urine['value'] == '+'), ['value']] = '3.6'
        df_urine.loc[(df_urine['item_name']=='上皮细胞') & (df_urine['value'] == '++'), ['value']] = '5.4'
        df_urine.loc[(df_urine['item_name']=='上皮细胞') & (df_urine['value'] == '+++'), ['value']] = '9'
        df_urine.loc[df_urine['item_name']=='上皮细胞', ['value']] = df_urine.loc[df_urine['item_name']=='上皮细胞', ['value']].replace('[^\d^\.]+', '', regex=True)

        df_urine.loc[df_urine['item_name']=='白细胞', ['value']] = df_urine.loc[df_urine['item_name']=='白细胞', ['value']].replace('(-----\.--|([0-9](-)[0-9]+))', '0', regex=True)
        df_urine.loc[(df_urine['item_name']=='白细胞') & (df_urine['value'] == '+'), ['value']] = '2.88'
        df_urine.loc[(df_urine['item_name']=='白细胞') & (df_urine['value'] == '++'), ['value']] = '9.18'
        df_urine.loc[(df_urine['item_name']=='白细胞') & (df_urine['value'] == '+++'), ['value']] = '27.18'
        df_urine.loc[(df_urine['item_name']=='白细胞') & (df_urine['value'] == '++++'), ['value']] = '63'
        df_urine.loc[df_urine['item_name']=='白细胞', ['value']] = df_urine.loc[df_urine['item_name']=='白细胞', ['value']].replace('[^\d^\.]+', '', regex=True)
        
        df_urine.loc[df_urine['item_name']=='红细胞', ['value']] = df_urine.loc[df_urine['item_name']=='红细胞', ['value']].replace('(-----\.--|([0-9](-)[0-9]+))', '0', regex=True)
        df_urine.loc[(df_urine['item_name']=='红细胞') & (df_urine['value'] == '-'), ['value']] = '0'
        df_urine.loc[(df_urine['item_name']=='红细胞') & (df_urine['value'] == '+'), ['value']] = '3.78'
        df_urine.loc[(df_urine['item_name']=='红细胞') & (df_urine['value'] == '++'), ['value']] = '12.78'
        df_urine.loc[(df_urine['item_name']=='红细胞') & (df_urine['value'] == '+++'), ['value']] = '36.18'
        df_urine.loc[(df_urine['item_name']=='红细胞') & (df_urine['value'] == '++++'), ['value']] = '72'
        df_urine.loc[df_urine['item_name']=='红细胞', ['value']] = df_urine.loc[df_urine['item_name']=='红细胞', ['value']].replace('[^\d^\.]+', '', regex=True)
        
        # process the df_shit test data
        df_shit.loc[df_shit['item_name']=='白细胞', ['value']] = df_shit.loc[df_shit['item_name']=='白细胞', ['value']].replace('(阴性|未查见|未见到|φ)', '-', regex=True)
        df_shit.loc[df_shit['item_name']=='白细胞', ['value']] = df_shit.loc[df_shit['item_name']=='白细胞', ['value']].replace('(0-1|0-3|2-3|1-3|0-2|3-4|4-6|少量/HP|1-3/HP|少量|2-5|1-5|0-2/HP|1.5|0-1/HP|1-2|3-6|2-4|3-5|±/HP)', '±', regex=True)
        df_shit.loc[df_shit['item_name']=='白细胞', ['value']] = df_shit.loc[df_shit['item_name']=='白细胞', ['value']].replace('(\+\+\+\+|\+\+\+|\+\+)', '2+', regex=True)
        df_shit.loc[df_shit['item_name']=='白细胞', ['value']] = df_shit.loc[df_shit['item_name']=='白细胞', ['value']].replace('(5-6|5-7|\+/HP|5-8|6-8)', '1+', regex=True)
        df_shit.loc[(df_shit['item_name']=='白细胞') & (df_shit['value'] == '+'), ['value']] = '1+'
        
        df_shit.loc[(df_shit['item_name']=='红细胞') & (df_shit['value'] == '0'), ['value']] = '-'
        df_shit.loc[df_shit['item_name']=='红细胞', ['value']] = df_shit.loc[df_shit['item_name']=='红细胞', ['value']].replace('(阴性|未查见|未见到|φ|未找到)', '-', regex=True)
        df_shit.loc[df_shit['item_name']=='红细胞', ['value']] = df_shit.loc[df_shit['item_name']=='红细胞', ['value']].replace('(\+\+\+\+|\+\+\+|\+\+|10-15)', '2+', regex=True)
        df_shit.loc[df_shit['item_name']=='红细胞', ['value']] = df_shit.loc[df_shit['item_name']=='红细胞', ['value']].replace('(0-1|0-2|1-2|1-3|0-3|2-3|2-4|1-4|0-4|3-5|3-4|2-5|4-5|0-6|2-6|3\.3|少量|4-6|3-6)', '±', regex=True)
        df_shit.loc[df_shit['item_name']=='红细胞', ['value']] = df_shit.loc[df_shit['item_name']=='红细胞', ['value']].replace('(6-8|5-6|2-7|2-8|4-8)', '1+', regex=True)
        df_shit.loc[(df_shit['item_name']=='红细胞') & (df_shit['value'] == '+'), ['value']] = '1+'
        
        df_shit.loc[df_shit['item_name']=='脓细胞', ['value']] = df_shit.loc[df_shit['item_name']=='脓细胞', ['value']].replace('(阴性|未查见|未见到|φ|未找到)', '-', regex=True)
        df_shit.loc[df_shit['item_name']=='脓细胞', ['value']] = df_shit.loc[df_shit['item_name']=='脓细胞', ['value']].replace('(0-1|0-2|0-3|1-3|1-2|2-4|少量)', '±', regex=True)
        df_shit.loc[df_shit['item_name']=='脓细胞', ['value']] = df_shit.loc[df_shit['item_name']=='脓细胞', ['value']].replace('(\+\+\+\+|\+\+\+|\+\+)', '2+', regex=True)
        df_shit.loc[df_shit['item_name']=='脓细胞', ['value']] = df_shit.loc[df_shit['item_name']=='脓细胞', ['value']].replace('(4-6|3-5)', '1+', regex=True)
        df_shit.loc[(df_shit['item_name']=='脓细胞') & (df_shit['value'] == '+'), ['value']] = '1+'
        
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(粘液便|黏液稀便|软便带粘液|黏液便)', '粘液', regex=True)
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(软便|软有血丝)', '软', regex=True)
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(烂便带血丝|烂便|烂带血丝)', '烂', regex=True)
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(糊状便带血丝|糊状便|糊便|糊粘稠)', '糊', regex=True)
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(稀汁样便|稀便)', '稀', regex=True)     
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(硬便)', '硬', regex=True)
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(水样便|稀水样|水样稀)', '水样', regex=True)
        df_shit.loc[df_shit['item_name']=='性状', ['value']] = df_shit.loc[df_shit['item_name']=='性状', ['value']].replace('(\-|肛拭子|脓血便|\+\+|无|已干)', '', regex=True)

        df_shit.loc[df_shit['item_name']=='不消化食物', ['value']] = df_shit.loc[df_shit['item_name']=='不消化食物', ['value']].replace('(\+\+|\+\-|少量|\+)', '阳性', regex=True)
        df_shit.loc[df_shit['item_name']=='不消化食物', ['value']] = df_shit.loc[df_shit['item_name']=='不消化食物', ['value']].replace('(阴性|未查见|未见到|φ|--|-)', '阴性', regex=True)
        
        df_shit.loc[df_shit['item_name']=='隐血试验', ['value']] = df_shit.loc[df_shit['item_name']=='隐血试验', ['value']].replace('(\+\-|0-1|量少)', '弱阳性', regex=True)
        df_shit.loc[df_shit['item_name']=='隐血试验', ['value']] = df_shit.loc[df_shit['item_name']=='隐血试验', ['value']].replace('(\+\+\+\+)', '强阳性', regex=True)
        df_shit.loc[df_shit['item_name']=='隐血试验', ['value']] = df_shit.loc[df_shit['item_name']=='隐血试验', ['value']].replace('(\+\+\+|阳性\+|\+\+|\+)', '阳性', regex=True)
        df_shit.loc[(df_shit['item_name']=='隐血试验') & (df_shit['value'] == '-'), ['value']] = '阴性'
        df_shit.loc[(df_shit['item_name']=='隐血试验') & (df_shit['value'] == '阴'), ['value']] = '阴性'
        df_shit.loc[(df_shit['item_name']=='隐血试验') & (df_shit['value'] == '弱阳'), ['value']] = '弱阳性'
        df_shit.loc[df_shit['item_name']=='隐血试验', ['value']] = df_shit.loc[df_shit['item_name']=='隐血试验', ['value']].replace('(失效|\?\?\?)', '', regex=True)

        df_shit.loc[df_shit['item_name']=='颜色（大便）', ['value']] = df_shit.loc[df_shit['item_name']=='颜色（大便）', ['value']].replace('(淡红色|樱红色|砖红色|黄色偏红|黄红色|红棕色|红棕)', '红色', regex=True)
        df_shit.loc[df_shit['item_name']=='颜色（大便）', ['value']] = df_shit.loc[df_shit['item_name']=='颜色（大便）', ['value']].replace('(灰白色|灰白|灰褐色|灰)', '白色', regex=True)
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '黄'), ['value']] = '黄色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '黄褐'), ['value']] = '黄褐色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '咖啡'), ['value']] = '黑褐色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '深褐色'), ['value']] = '黑褐色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '陶土色'), ['value']] = '黑褐色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '土黄色'), ['value']] = '黄色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '褐绿色'), ['value']] = '绿色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '黄绿'), ['value']] = '黄绿色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '鲜血便'), ['value']] = '红色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '墨绿色'), ['value']] = '绿色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '绿褐色'), ['value']] = '绿色'
        df_shit.loc[(df_shit['item_name']=='颜色（大便）') & (df_shit['value'] == '淡黄色'), ['value']] = '黄色'

        # df_out = pd.concat([df_shit, df_urine, df_blood_cat, df_blood_val])
        df_cat_out = pd.concat([df_shit, df_urine, df_blood_cat])
        print("-----------------------------------------------------------------------")
        print("The shapes of measurement data after processing are {}".format(df_cat_out.shape))
        print("-----------------------------------------------------------------------")

        self.insertdata_cat(df_cat_out)
        print("-----------------------------------------------------------------------")
        print("The categorical data have been inserted!")
        print("-----------------------------------------------------------------------")
        self.insertdata_val(df_blood_val)

class SymAndChiefcomV2():

    def __init__(self, db_name, db_name_sys, sym_table, chiefcom_table, out_table_name):

        self.db_conn = db_connection(db_name)
        self.db_conn_sys = db_connection(db_name_sys)
        self.out_table_name = out_table_name
        self.column_query = Table('ALL_TAB_COLUMNS', self.db_conn_sys.metadata, autoload = True)
        self.sym_table = Table(sym_table, self.db_conn.metadata, autoload=True)
        self.chiefcom_table = Table(chiefcom_table, self.db_conn.metadata, autoload=True)
        self.out_table = Table(out_table_name, self.db_conn.metadata, autoload=True)
        self.target_sym_dict = ['发热','咳嗽','乏力','畏寒','咳痰','头痛','胸闷','呼吸困难','疼痛','最高体温']

    def getcolumn(self, table_name):
        #get column name list
        s = select([self.column_query.c.column_name]).where(self.column_query.c.table_name == table_name)
        table_column = self.db_conn_sys.conn.execute(s).fetchall()
        column = [col for (col,) in table_column]
        column.reverse()
        return column

    def get_sym_data(self):

        # preprocess the p1_note_zz table 
        s1 = select([self.sym_table.c.sign_id,
                     self.sym_table.c.note_id,
                     self.sym_table.c.person_id,
                     self.sym_table.c.visit_record_id,
                     self.sym_table.c.location,
                     self.sym_table.c.name,
                     self.sym_table.c.value,
                     self.sym_table.c.duration,
                     self.sym_table.c.frequency,
                     self.sym_table.c.sign_name_general,
                     self.sym_table.c.location_general])
        dfs = self.db_conn.conn.execute(s1).fetchall()
        result = pd.DataFrame(dfs, columns=['sign_id','note_id','person_id','visit_record_id','location','name','value','duration','frequency','sign_name_general','location_general'])
        return result

    def get_chief_data(self):

        # preprocess the p1_note_zs table 
        s1 = select([self.chiefcom_table.c.sign_id,
                     self.chiefcom_table.c.note_id,
                     self.chiefcom_table.c.person_id,
                     self.chiefcom_table.c.visit_record_id,
                     self.chiefcom_table.c.location,
                     self.chiefcom_table.c.sign_name,
                     self.chiefcom_table.c.duration,
                     self.chiefcom_table.c.frequency,
                     self.chiefcom_table.c.sign_name_general,
                     self.chiefcom_table.c.location_general])
        dfs = self.db_conn.conn.execute(s1).fetchall()
        result = pd.DataFrame(dfs, columns=['sign_id','note_id','person_id','visit_record_id','location','name','duration','frequency','sign_name_general','location_general'])
        return result

    def mapping_df_types(self, df):
        """
        INPUT: df(要插回数据库的DataFrame数据)
        OUTPUT: dtypedict(指定每个字段要插入到数据库的字段类型)
        """
        dtypedict = {}

        for i, j in zip(df.columns, df.dtypes):
            if str(i) == "VISIT_RECORD_ID_NEW_1":
                dtypedict.update({i: VARCHAR2(20)})
            if str(i) =="NAME":
                dtypedict.update({i: VARCHAR2(100)})
            if str(i) == "VALUE":
                dtypedict.update({i: VARCHAR2(50)})
        return dtypedict

    def insertdata(self, df_data, dtypedict_status=True):
        """
        INPUT: df_data(要插回数据库的数据)
        """
        #获取定义的字典
        if dtypedict_status:
            dtypedict = self.mapping_df_types(df_data)
        else:
            dtypedict = None
        #写入数据库
        df_data.to_sql(self.out_table_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)

    def symprocess(self):
        
        data_sym = self.get_sym_data()

        print("-----------------------------------------------------------------------")
        print("The original shape of P1_NOTE_ZZ data is {}".format(data_sym.shape))
        print("-----------------------------------------------------------------------")
        # extract the target data
        data_sym_tar = data_sym.loc[data_sym['sign_name_general'].isin(self.target_sym_dict),]
        data_sym_tar.loc[data_sym_tar['sign_name_general'] != '发热', ['duration']] = ''
        data_sym_tar.loc[data_sym_tar['sign_name_general'] != '发热', ['frequency']] = ''

        # drop duplicates
        if any(data_sym_tar.duplicated(subset=['visit_record_id','location_general','sign_name_general','value', 'duration', 'frequency'])):
            data_sym_tar = data_sym_tar.drop_duplicates(subset=['visit_record_id','location_general','sign_name_general','value', 'duration', 'frequency'], keep='first')
            print("-----------------------------------------------------------------------")
            print("The duplicated and filtered shape of P1_NOTE_ZZ data is {}".format(data_sym_tar.shape))
            print("-----------------------------------------------------------------------")

        # process the value of duration
        # generalization
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('日','天')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('半天','1天')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('半月','15天')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('半年','180天')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('十','10')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('九','9')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('八','8')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('七','7')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('六','6')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('五','5')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('四','4')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('三','3')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('二','2')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('两','2')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('一','1')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('数','1')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('+','')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('９','9')
        data_sym_tar['duration']=data_sym_tar['duration'].str.replace('１','1')

        data_sym_tar['frequency']=data_sym_tar['frequency'].str.replace('间歇','间断')
        data_sym_tar['frequency']=data_sym_tar['frequency'].str.replace('长期','持续')
        data_sym_tar['frequency']=data_sym_tar['frequency'].str.replace('阵发性','间断')
        data_sym_tar['frequency']=data_sym_tar['frequency'].str.replace('偶有','间断')

        data_sym_tar.loc[data_sym_tar['sign_name_general']=='最高体温', ['value']] = data_sym_tar.loc[data_sym_tar['sign_name_general']=='最高体温', ['value']].replace('[^\d^\.]+', '', regex=True)
        # extract the duration value and unit
        data_sym_tar['duration_value'] = data_sym_tar['duration'].str.extract(r'(\d+).*(?=天|小时|周|月|年)')
        # CAUTION:str.extract(),every () matched means a value
        data_sym_tar['duration_unit'] = data_sym_tar['duration'].str.extract(r'(小时|天|周|月|年){1}')
        # update the value of DURATION_VALUE as the unit of day
        data_sym_tar.loc[data_sym_tar['duration_unit'] == '小时', 'duration_value_day'] = '1'
        data_sym_tar.loc[data_sym_tar['duration_unit'] == '天', 'duration_value_day'] = data_sym_tar['duration_value']
        data_sym_tar.loc[data_sym_tar['duration_unit'] == '周', 'duration_value_day'] = data_sym_tar['duration_value'].astype(float)*7
        data_sym_tar.loc[data_sym_tar['duration_unit'] == '月', 'duration_value_day'] = data_sym_tar['duration_value'].astype(float)*30
        data_sym_tar.loc[data_sym_tar['duration_unit'] == '年', 'duration_value_day'] = data_sym_tar['duration_value'].astype(float)*356

        print("-----------------------------------------------------------------------")
        print("The  shape of P1_NOTE_ZZ data after processing is {}".format(data_sym_tar.shape))
        print("-----------------------------------------------------------------------")

        return data_sym_tar

    def chiefcomprocess(self):

        data_chief = self.get_chief_data()

        print("-----------------------------------------------------------------------")
        print("The original shape of P1_NOTE_ZS data is {}".format(data_chief.shape))
        print("-----------------------------------------------------------------------")
        # extract the target data
        data_chief_tar = data_chief.loc[data_chief['sign_name_general'].isin(self.target_sym_dict),]
        data_chief_tar.loc[data_chief_tar['sign_name_general'] != '发热', ['duration']] = ''
        data_chief_tar.loc[data_chief_tar['sign_name_general'] != '发热', ['frequency']] = ''

        # drop duplicates
        if any(data_chief_tar.duplicated(subset=['visit_record_id','location_general','sign_name_general', 'duration', 'frequency'])):
            data_chief_tar = data_chief_tar.drop_duplicates(subset=['visit_record_id','location_general','sign_name_general','duration', 'frequency'], keep='first')
            print("-----------------------------------------------------------------------")
            print("The duplicated and filtered shape of P1_NOTE_ZS data is {}".format(data_chief_tar.shape))
            print("-----------------------------------------------------------------------")

        # process the value of duration
        # generalization
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('日','天')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('半天','1天')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('半月','15天')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('半年','180天')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('十','10')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('九','9')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('八','8')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('七','7')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('六','6')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('五','5')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('四','4')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('三','3')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('二','2')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('两','2')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('一','1')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('数','1')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('+','')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('９','9')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('１','1')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('８','8')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('４','4')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('１０','10')
        data_chief_tar['duration']=data_chief_tar['duration'].str.replace('２０','10')

        data_chief_tar['frequency']=data_chief_tar['frequency'].str.replace('间歇','间断')
        data_chief_tar['frequency']=data_chief_tar['frequency'].str.replace('长期','持续')
        data_chief_tar['frequency']=data_chief_tar['frequency'].str.replace('阵发性','间断')
        data_chief_tar['frequency']=data_chief_tar['frequency'].str.replace('偶有','间断')

        # extract the duration value and unit
        data_chief_tar['duration_value'] = data_chief_tar['duration'].str.extract(r'(\d+).*(?=天|小时|周|月|年)')
        # CAUTION:str.extract(),every () matched means a value
        data_chief_tar['duration_unit'] = data_chief_tar['duration'].str.extract(r'(小时|天|周|月|年){1}')
        # update the value of DURATION_VALUE as the unit of day
        data_chief_tar.loc[data_chief_tar['duration_unit'] == '小时', 'duration_value_day'] = '1'
        data_chief_tar.loc[data_chief_tar['duration_unit'] == '天', 'duration_value_day'] = data_chief_tar['duration_value']
        data_chief_tar.loc[data_chief_tar['duration_unit'] == '周', 'duration_value_day'] = data_chief_tar['duration_value'].astype(float)*7
        data_chief_tar.loc[data_chief_tar['duration_unit'] == '月', 'duration_value_day'] = data_chief_tar['duration_value'].astype(float)*30
        data_chief_tar.loc[data_chief_tar['duration_unit'] == '年', 'duration_value_day'] = data_chief_tar['duration_value'].astype(float)*356

        print("-----------------------------------------------------------------------")
        print("The  shape of P1_NOTE_ZS data after processing is {}".format(data_chief_tar.shape))
        print("-----------------------------------------------------------------------")
        
        return data_chief_tar

    def concat2one(self):
        
        # extract symptom data
        data_sym = self.symprocess()

        # extract chief complaint data
        data_chief = self.chiefcomprocess()
        data_chief.rename(columns = {'sign_name': 'name'}, inplace = True)
        data_chief['value'] = '有'

        # concat the two tables
        data = pd.concat([data_sym, data_chief], ignore_index = True)
        print("-----------------------------------------------------------------------")
        print("The  shape of P1_NOTE_ZZ and P1_NOTE_ZS data after concat is {}".format(data.shape))
        print("-----------------------------------------------------------------------")

        # drop duplicates after concat, keep the earlier one
        data.sort_values(by = ['visit_record_id', 'sign_name_general', 'location_general', 'value', 'duration', 'frequency'], na_position = 'last', inplace = True, ascending = True)
        data_nodup = data.drop_duplicates(subset=['visit_record_id', 'sign_name_general', 'location_general'], keep='first')
        print("-----------------------------------------------------------------------")
        print("The  shape of P1_NOTE_ZZ and P1_NOTE_ZS data after duplicated is {}".format(data_nodup.shape))
        print("-----------------------------------------------------------------------")

        # process the symptom data
        iterrow_count = 1
        df = pd.DataFrame([], columns = ['sign_id', 'visit_record_id', 'name', 'value'])
        for index, row in data_nodup.iterrows():
            if row['sign_name_general'] == '疼痛':
                if row['value'] == '无':
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general'], 'value': row['value']}, ignore_index=True)
                else:
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general'], 'value': row['location_general']}, ignore_index=True)
            elif row['sign_name_general'] == '咳痰':
                if row['value'] == '无':
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general'], 'value': row['value']}, ignore_index=True)
                else:
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general'], 'value': row['name']}, ignore_index=True)
            elif row['sign_name_general'] == '发热':
                if row['value'] == '无':
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general'], 'value': row['value']}, ignore_index=True)
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general']+'-'+'频次', 'value': row['frequency']}, ignore_index=True)
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general']+'-'+'时长', 'value': row['duration_value_day']}, ignore_index=True)
                else:
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general'], 'value': row['name']}, ignore_index=True)
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general']+'-'+'频次', 'value': row['frequency']}, ignore_index=True)
                    df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general']+'-'+'时长', 'value': row['duration_value_day']}, ignore_index=True)
            else:
                df = df.append({'sign_id': row['sign_id'],'visit_record_id': row['visit_record_id'], 'name': row['sign_name_general'], 'value': row['value']}, ignore_index=True)
            print("[Message: {} data have been processed".format(iterrow_count))
            iterrow_count += 1

        df = df.where(df.notnull(),'')
        df = df.astype(str, copy=True)
        print("-----------------------------------------------------------------------")
        print("The  shape of  last data is {}".format(df.shape))
        print("-----------------------------------------------------------------------")

        self.insertdata(df, dtypedict_status=False)

class FeatureAggreV2():

    def __init__(self, db_name, db_name_sys, table_person, table_visit, table_zz, table_jws, table_grs, table_mea_val, table_mea_cat, table_sign, table_sign_out_48h, table_sign_out_stat, table_feature_cat, table_feature_val):

        self.db_conn = db_connection(db_name)
        self.db_conn_sys = db_connection(db_name_sys)
        self.column_query = Table('ALL_TAB_COLUMNS', self.db_conn_sys.metadata, autoload = True)
        self.table_person_name = table_person
        self.table_visit_name = table_visit
        self.table_zz_name = table_zz
        self.table_jws_name = table_jws
        self.table_grs_name = table_grs
        self.table_mea_val_name = table_mea_val
        self.table_mea_cat_name = table_mea_cat
        self.table_sign_name = table_sign
        self.table_sign_out_48h_name = table_sign_out_48h
        self.table_sign_out_stat_name = table_sign_out_stat
        self.table_feature_cat_name = table_feature_cat
        self.table_feature_val_name = table_feature_val
        self.table_person = Table(table_person, self.db_conn.metadata, autoload=True)
        self.table_visit = Table(table_visit, self.db_conn.metadata, autoload=True)
        self.table_zz = Table(table_zz, self.db_conn.metadata, autoload=True)
        self.table_jws = Table(table_jws, self.db_conn.metadata, autoload=True)
        self.table_grs = Table(table_grs, self.db_conn.metadata, autoload=True)
        self.table_mea_val = Table(table_mea_val, self.db_conn.metadata, autoload=True)
        self.table_mea_cat = Table(table_mea_cat, self.db_conn.metadata, autoload=True)
        self.table_sign = Table(table_sign, self.db_conn.metadata, autoload=True)
        self.table_sign_out_48h = Table(table_sign_out_48h, self.db_conn.metadata, autoload=True)
        self.table_sign_out_stat = Table(table_sign_out_stat, self.db_conn.metadata, autoload=True)
        self.table_feature_cat = Table(table_feature_cat, self.db_conn.metadata, autoload=True)
        self.table_feature_val = Table(table_feature_val, self.db_conn.metadata, autoload=True)


        self.rootDir = os.path.split(os.path.realpath(__file__))[0]
        self.logpath = os.path.join(self.rootDir, 'log/FeatureAggreV2.txt')
        self.datapath_val = os.path.join(self.rootDir, 'data/feature_matrix_val.txt')
        self.datapath_cat = os.path.join(self.rootDir, 'data/feature_matrix_cat.txt')

    def getcolumn(self, table_name):
        #get column name list
        s = select([self.column_query.c.column_name]).where(self.column_query.c.table_name == table_name)
        table_column = self.db_conn_sys.conn.execute(s).fetchall()
        print(table_column)
        column = [col for (col,) in table_column]
        column.reverse()
        return column
    
    def getvrilist(self, table_name):

        if table_name.lower() == 'signs':
            s1 = select([distinct(self.table_sign.c.visit_record_id_new_1)])
            result = self.db_conn.conn.execute(s1).fetchall()
            vris = [vri for (vri,) in result]
        elif table_name.lower() == 'p1_sign_48h':
            s1 = select([distinct(self.table_sign_out_48h.c.visit_record_id_new_1)])
            result = self.db_conn.conn.execute(s1).fetchall()
            vris = [vri for (vri,) in result]

        return vris

    def mapping_df_types(self, df, table_name):
        """
        INPUT: df(要插回数据库的DataFrame数据)
        OUTPUT: dtypedict(指定每个字段要插入到数据库的字段类型)
        """
        #定义空字典
        dtypedict = {}
        if table_name.lower() == 'p1_sign_stat':
            for i, j in zip(df.columns, df.dtypes):
                if str(i) =="visit_record_id_new_1":
                    dtypedict.update({i: VARCHAR2(50)})
                if str(i) == "type":
                    dtypedict.update({i: VARCHAR2(20)})
                if str(i) == "min_value":
                    dtypedict.update({i: VARCHAR2(20)})
                if str(i) == "max_value":
                    dtypedict.update({i: VARCHAR2(20)})
                if str(i) == "avg_value":
                    dtypedict.update({i: VARCHAR2(20)})
                if str(i) == "std_value":
                    dtypedict.update({i: VARCHAR2(20)})
                if str(i) == "median_value":
                    dtypedict.update({i: VARCHAR2(20)})
        
        elif table_name.lower() == 'p1_sign_48h':
            for i, j in zip(df.columns, df.dtypes):
                if str(i) == "sign_id":
                    dtypedict.update({i: VARCHAR2(100)})
                if str(i) == "visit_record_id":
                    dtypedict.update({i: VARCHAR2(50)})
                if str(i) =="visit_record_id_new_1":
                    dtypedict.update({i: VARCHAR2(50)})
                if str(i) == "person_id":
                    dtypedict.update({i: VARCHAR2(100)})
                if str(i) == "mea_time":
                    dtypedict.update({i: VARCHAR2(100)})
                if str(i) == "type":
                    dtypedict.update({i: VARCHAR2(20)})
                if str(i) == "value":
                    dtypedict.update({i: VARCHAR2(100)})
                if str(i) == "unit":
                    dtypedict.update({i: VARCHAR2(20)})

        elif table_name.lower() == 'p2_feature_val':
            for i, j in zip(df.columns, df.dtypes):
                if str(i) == "visit_record_id_new_1":
                    dtypedict.update({i: VARCHAR2(50)})
                if str(i) == "name":
                    dtypedict.update({i: VARCHAR2(500)})
                if str(i) =="value":
                    dtypedict.update({i: VARCHAR2(500)})

        elif table_name.lower() == 'p2_feature_cat':
            for i, j in zip(df.columns, df.dtypes):
                if str(i) == "visit_record_id_new_1":
                    dtypedict.update({i: VARCHAR2(50)})
                if str(i) == "name":
                    dtypedict.update({i: VARCHAR2(500)})
                if str(i) =="value":
                    dtypedict.update({i: VARCHAR2(500)})
            
            return dtypedict
    
    def data_insert(self, df_data, table_out_name, dtypedict_status=True):
        """
        INPUT: df_data(要插回数据库的数据)
        """
        #获取定义的字典
        if dtypedict_status:
            dtypedict = self.mapping_df_types(df_data, table_out_name)
        else:
            dtypedict = None
        #写入数据库
        if table_out_name.lower() == 'p1_sign_48h':
            df_data.to_sql(self.table_sign_out_48h_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=1)
        
        elif table_out_name.lower() == 'p1_sign_stat':
            df_data.to_sql(self.table_sign_out_stat_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)
        elif table_out_name.lower() == 'p1_feature_val':
            df_data.to_sql(self.table_feature_val_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)
        elif table_out_name.lower() == 'p2_feature_cat':
            df_data.to_sql(self.table_feature_cat_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)
        elif table_out_name.lower() == 'p2_feature_val':
            df_data.to_sql(self.table_feature_val_name, self.db_conn.engine, index=False, if_exists='append', dtype=dtypedict, chunksize=10)

    def process_visit(self):

        # read the visit data
        data_visit = pd.read_sql_table(self.table_visit_name.lower(), self.db_conn.conn)

        try:
            f = open(self.logpath, 'a')
            f.write("[info] The shape of original data_visit is {}".format(data_visit.shape) + "\n")
            f.write("[info] The number of unique  person_id is {}".format(len(data_visit['person_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id is {}".format(len(data_visit['visit_record_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id_new_1 is {}".format(len(data_visit['visit_record_id_new_1'].unique())) + "\n")     
            f.close()
        except IOError:
            f.close()

        # drop duplicated data
        if any(data_visit.duplicated(subset=['visit_record_id_new_1'])):
            data_visit.sort_values(by = ['visit_record_id_new_1', 'visit_start_date'], na_position = 'last', inplace = True, ascending = True)
            data_visit.drop_duplicates(subset=['visit_record_id_new_1'], keep='first', inplace=True)
            
        # take the several columns
        data_visit = data_visit[['visit_record_id', 'visit_record_id_new_1', 'person_id', 'visit_age', 'weight', 'height', 'visit_start_date']]

        # take out the visit year, month, day from the column visit_start_date
        data_visit['visit_year'] = data_visit['visit_start_date'].dt.year
        data_visit['visit_month'] = data_visit['visit_start_date'].dt.month
        data_visit['visit_day'] = data_visit['visit_start_date'].dt.day
        data_visit.drop('visit_start_date', axis=1, inplace=True)

        # merge with table_person to get the gender data
        data_person = pd.read_sql_table(self.table_person_name.lower(), self.db_conn.conn)
        data_person = data_person[['person_id', 'gender']]
        data_person.drop_duplicates(subset=['person_id', 'gender'], keep='first', inplace=True)

        result = data_visit.merge(data_person, how='left', on='person_id')
        # pd.set_option('display.max_columns', None)

        try:
            f = open(self.logpath, 'a')
            f.write("[info] The shape of data_visit after duplication is {}".format(result.shape) + "\n")
            f.write("[info] The number of unique  person_id is {}".format(len(result['person_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id is {}".format(len(result['visit_record_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id_new_1 is {}".format(len(result['visit_record_id_new_1'].unique())) + "\n")
            f.write("-------------------------------------------------------------------------" + "\n")
            f.close()
        except IOError:
            f.close()
        
        return result

    def process_jws(self):
        
        data_jws = pd.read_sql_table(self.table_jws_name.lower(), self.db_conn.conn)
        data_visit = self.process_visit()

        try:
            f = open(self.logpath, 'a')
            f.write("[info] The shape of original data_jws is {}".format(data_jws.shape) + "\n")
            f.write("[info] The number of unique  person_id is {}".format(len(data_jws['person_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id is {}".format(len(data_jws['visit_record_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id_new_1 is {}".format(len(data_jws['visit_record_id_new_1'].unique())) + "\n")     
            f.close()
        except IOError:
            f.close()

        # take specific jws data
        name_included = ['既往体质', '输血史', '外伤史', '糖尿病史', '肾病史', '肾病史', '心脏病史', '肺结核史', '高血压史', '长期用药史', '手术史']
        data_jws = data_jws[data_jws['name'].isin(name_included)]

        # # take the data in visit_zy
        # data_jws = data_jws[data_jws['visit_record_id'].isin(data_visit['visit_record_id'])]

        # drop duplicates in the range of all columns
        if any(data_jws.duplicated(subset=['visit_record_id_new_1', 'name'])):
            data_jws.sort_values(by = ['visit_record_id_new_1', 'name', 'time'], na_position = 'last', inplace = True, ascending=True)
            data_jws.drop_duplicates(subset=['visit_record_id_new_1', 'name'], keep='first', inplace=True)

        result = data_jws[['visit_record_id', 'visit_record_id_new_1', 'person_id', 'name', 'value']]

        # generalize the value
        result.loc[result['value']=='否认', ['value']] = '无'
        result.loc[result['value']=='患', ['value']] = '无'
        result.loc[(result['name']=='既往体质') & (result['value']=='良'), ['value']] = '良好'
        result.loc[(result['name']=='既往体质') & (result['value']=='偏差'), ['value']] = '较差'
        result.loc[(result['name']=='既往体质') & (result['value']=='好'), ['value']] = '良好'
        result.loc[(result['name']=='既往体质') & (result['value']=='欠佳'), ['value']] = '较差'
        result.loc[(result['name']=='既往体质') & (result['value']=='差'), ['value']] = '较差'
        result.loc[(result['name']=='既往体质') & (result['value']=='可'), ['value']] = '一般'
        result.loc[(result['name']=='既往体质') & (result['value']=='尚可'), ['value']] = '一般'
        result.loc[(result['name']=='既往体质') & (result['value']=='体健'), ['value']] = '良好'
        try:
            f = open(self.logpath, 'a')
            f.write("[info] The shape of data_jws after processing is {}".format(result.shape) + "\n")
            f.write("[info] The number of unique  person_id is {}".format(len(result['person_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id is {}".format(len(result['visit_record_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id_new_1 is {}".format(len(result['visit_record_id_new_1'].unique())) + "\n")
            f.write("-------------------------------------------------------------------------" + "\n")
            f.close()
        except IOError:
            f.close()

        return result

    def process_grs(self):

        data_grs = pd.read_sql_table(self.table_grs_name.lower(), self.db_conn.conn)
        data_visit = self.process_visit()
        try:
            f = open(self.logpath, 'a')
            f.write("[info] The shape of original data_grs is {}".format(data_grs.shape) + "\n")
            f.write("[info] The number of unique  person_id is {}".format(len(data_grs['person_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id is {}".format(len(data_grs['visit_record_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id_new_1 is {}".format(len(data_grs['visit_record_id_new_1'].unique())) + "\n")     
            f.close()
        except IOError:
            f.close()

        # take the data in visit_zy
        # data_grs = data_grs[data_grs['visit_record_id'].isin(data_visit['visit_record_id'])]

        # drop duplicates in the range of all columns
        if any(data_grs.duplicated(subset=['visit_record_id_new_1', 'name'])):
            data_grs.sort_values(by = ['visit_record_id_new_1', 'name', 'time'], na_position = 'last', inplace = True, ascending=True)
            data_grs.drop_duplicates(subset=['visit_record_id_new_1', 'name'], keep='first', inplace=True)

        result = data_grs[['visit_record_id', 'visit_record_id_new_1', 'person_id', 'name', 'value']]

        try:
            f = open(self.logpath, 'a')
            f.write("[info] The shape of data_grs after processing is {}".format(result.shape) + "\n")
            f.write("[info] The number of unique  person_id is {}".format(len(result['person_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id is {}".format(len(result['visit_record_id'].unique())) + "\n")
            f.write("[info] The number of unique  visit_record_id_new_1 is {}".format(len(result['visit_record_id_new_1'].unique())) + "\n")
            f.write("-------------------------------------------------------------------------" + "\n")
            f.close()
        except IOError:
            f.close()

        return result

    def process_signs(self):

        # data_sign = pd.read_sql_table(self.table_sign_name.lower(), self.db_conn.conn)
        data_visit = self.process_visit()
        # # take specific jws data
        type_included = ['体温', '呼吸', '血压', '脉搏', 'SPO2']

        data_index = self.getvrilist()

        for i in data_index:
            s = select([self.table_sign.c.sign_id,
                        self.table_sign.c.visit_record_id,
                        self.table_sign.c.visit_record_id_new_1,
                        self.table_sign.c.person_id,
                        self.table_sign.c.mea_time,
                        self.table_sign.c.type,
                        self.table_sign.c.value,
                        self.table_sign.c.unit]).where(self.table_sign.c.visit_record_id_new_1 == i)
        
            result = self.db_conn.conn.execute(s).fetchall()
            data_sign = pd.DataFrame(result, columns=['sign_id', 'visit_record_id','visit_record_id_new_1','person_id','mea_time','type', 'value', 'unit'])

            data_sign = data_sign[data_sign['type'].isin(type_included)]

            #  # take the data in visit_zy
            data_sign = data_sign[data_sign['visit_record_id'].isin(data_visit['visit_record_id'])]
            if any(data_sign.duplicated(subset=['visit_record_id_new_1', 'type', 'mea_time', 'value'])):
                data_sign.sort_values(by = ['visit_record_id_new_1', 'type', 'mea_time'], na_position = 'last', inplace = True, ascending=True)
                data_sign.drop_duplicates(subset=['visit_record_id_new_1', 'type', 'mea_time', 'value'], keep='first', inplace=True)

            # process the unusual data
            data_sign['value'] = data_sign['value'].replace('[^\d^\.]+', '', regex=True)
            data_sign['value'] = data_sign['value'].replace('\.{2,100}', '.', regex=True)
            data_sign.loc[data_sign['value'] == '.', ['value']] = None

            print("[info]: data of {0} was processing".format(i))

            time_tag = data_sign['mea_time'].min() + np.timedelta64(48, 'h')

            data = data_sign.loc[data_sign['mea_time'] <= time_tag,]
            data['mea_time'] = data['mea_time'].astype(str)
            

            self.data_insert(data, self.table_sign_out_48h_name, dtypedict_status=True )

    def process_sign_stat(self):

        data_index = self.getvrilist(self.table_sign_out_48h_name)
        type_dict = ['体温', '呼吸', '血压', '脉搏', 'SPO2']
        
        count = 1
        data_out = pd.DataFrame([], columns=['visit_record_id_new_1','type','min_value','max_value','avg_value', 'std_value','median_value'])
        
        for j in type_dict:
            for i in data_index:
                s = select([self.table_sign_out_48h.c.visit_record_id_new_1,
                            self.table_sign_out_48h.c.type,
                            self.table_sign_out_48h.c.value]).where(self.table_sign_out_48h.c.type == j).where(self.table_sign_out_48h.c.visit_record_id_new_1 == i)
                result = self.db_conn.conn.execute(s).fetchall()
                data = pd.DataFrame(result, columns=['visit_record_id_new_1', 'type', 'value'])
            
                data_out = data_out.append({'visit_record_id_new_1': i, 'type': j, 'min_value': str(round(data['value'].astype(float).min(),2)), 'max_value': str(round(data['value'].astype(float).max(),2)), 'avg_value': str(round(data['value'].astype(float).mean(),2)), 'std_value': str(round(data['value'].astype(float).std(),2)), 'median_value': str(round(data['value'].astype(float).median(),2))}, ignore_index=True)
                print("[info]: The number of processed data is {} ".format(count))
                count += 1

        self.data_insert(data_out, self.table_sign_out_stat_name, dtypedict_status=True)
            
    def feature_matrix(self):
        
        # get the visit data
        data_visit = self.process_visit()
        data_visit = data_visit[['visit_record_id_new_1','gender','visit_age', 'weight', 'height', 'visit_year', 'visit_month', 'visit_day']]  # number (gender)
        data_visit = pd.melt(data_visit, id_vars=['visit_record_id_new_1'], value_vars=['gender','visit_age','weight','height', 'visit_year', 'visit_month', 'visit_day'], var_name='name', value_name='value')
        data_visit_cat = data_visit.loc[data_visit['name'].isin(['gender', 'visit_year', 'visit_month', 'visit_day']),]   # category
        data_visit_val = data_visit.loc[~data_visit['name'].isin(['gender', 'visit_year', 'visit_month', 'visit_day']),]   # number
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_visit_cat is {}".format(data_visit_cat.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_visit_cat['visit_record_id_new_1'].unique())))
        print("[info] The shape of data_visit_val is {}".format(data_visit_val.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_visit_cat['visit_record_id_new_1'].unique())))
        print("-----------------------------------------------------------------------")

        # get the jws data
        data_jws = self.process_jws()
        data_jws = data_jws[['visit_record_id_new_1','name','value']]   # category
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_jws is {}".format(data_jws.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_jws['visit_record_id_new_1'].unique())))
        print("[info] The unique number of name is {}".format(len(data_jws['name'].unique())))
        print("-----------------------------------------------------------------------")

        # get the grs data
        data_grs = self.process_grs()
        data_grs = data_grs[['visit_record_id_new_1','name','value']]   # category
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_grs is {}".format(data_grs.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_grs['visit_record_id_new_1'].unique())))
        print("[info] The unique number of name is {}".format(len(data_grs['name'].unique())))
        print("-----------------------------------------------------------------------")
        
        # get the measurement data
        data_mea_cat = pd.read_sql_table(self.table_mea_cat_name, self.db_conn.conn)
        data_mea_val = pd.read_sql_table(self.table_mea_val_name, self.db_conn.conn)
        # drop duplicates from measurement
        if any(data_mea_cat.duplicated(subset=['visit_record_id_new_1', 'specimen', 'item_name'])):
            data_mea_cat.sort_values(by = ['visit_record_id_new_1', 'specimen', 'item_name'], na_position = 'last', ascending=True, inplace = True)
            data_mea_cat = data_mea_cat.drop_duplicates(subset=['visit_record_id_new_1','specimen','item_name'], keep='first')
    
        if any(data_mea_val.duplicated(subset=['visit_record_id_new_1', 'specimen', 'item_name'])):
            data_mea_val.sort_values(by = ['visit_record_id_new_1', 'specimen', 'item_name'], na_position = 'last', ascending=True, inplace = True)
            data_mea_val = data_mea_val.drop_duplicates(subset=['visit_record_id_new_1','specimen','item_name'], keep='first')

        data_mea_cat['name'] = data_mea_cat.apply(lambda x: x['specimen'] + '-' + x['item_name'], axis = 1)
        data_mea_val['name'] = data_mea_val.apply(lambda x: x['specimen'] + '-' + x['item_name'], axis = 1)
        data_mea_val = data_mea_val[['visit_record_id_new_1','name','value']]     # number
        data_mea_cat = data_mea_cat[['visit_record_id_new_1','name','value']]     # category
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_mea_val is {}".format(data_mea_val.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_mea_val['visit_record_id_new_1'].unique())))
        print("[info] The unique number of name is {}".format(len(data_mea_val['name'].unique())))
        print("[info] The shape of data_mea_cat is {}".format(data_mea_cat.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_mea_cat['visit_record_id_new_1'].unique())))
        print("[info] The unique number of name is {}".format(len(data_mea_cat['name'].unique())))
        print("-----------------------------------------------------------------------")   

        # get the zz data
        data_zz = pd.read_sql_table(self.table_zz_name, self.db_conn.conn)
        data_zz = data_zz[['visit_record_id_new_1','name','value']]     # category (发热-时长，最高体温)
        if any(data_zz.duplicated(subset=['visit_record_id_new_1', 'name'])):
            data_zz.sort_values(by = ['visit_record_id_new_1', 'name', 'value'], na_position = 'last', ascending=True, inplace = True)
            data_zz = data_zz.drop_duplicates(subset=['visit_record_id_new_1','name'], keep='first')
        data_zz_val = data_zz.loc[(data_zz['name'] == '发热-时长') | (data_zz['name'] == '最高体温')]
        data_zz_cat = data_zz.loc[(data_zz['name'] != '发热-时长') & (data_zz['name'] != '最高体温')]
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_zz_val is {}".format(data_zz_val.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_zz_val['visit_record_id_new_1'].unique())))
        print("[info] The unique number of name is {}".format(len(data_zz_val['name'].unique())))
        print("[info] The shape of data_zz_cat is {}".format(data_zz_cat.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_zz_cat['visit_record_id_new_1'].unique())))
        print("[info] The unique number of name is {}".format(len(data_zz_cat['name'].unique())))
        print("-----------------------------------------------------------------------")

        # get the sign data
        data_sign_stat = pd.read_sql_table(self.table_sign_out_stat_name, self.db_conn.conn)
        data_sign_stat = pd.melt(data_sign_stat, id_vars=['visit_record_id_new_1', 'type'], value_vars=['min_value', 'max_value', 'avg_value', 'std_value', 'median_value'], var_name='value_name', value_name='value')
        data_sign_stat['name'] = data_sign_stat.apply(lambda x: x['type'] + '-' + x['value_name'], axis = 1)
        data_sign_stat = data_sign_stat[['visit_record_id_new_1','name','value']]  # number
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_sign_stat is {}".format(data_sign_stat.shape))
        print("[info] The unique number of visit_record_id_new_1 is {}".format(len(data_sign_stat['visit_record_id_new_1'].unique())))
        print("[info] The unique number of name is {}".format(len(data_sign_stat['name'].unique())))
        print("-----------------------------------------------------------------------")

        # concat the data
        data_cat = pd.concat([data_visit_cat, data_jws, data_grs, data_mea_cat, data_zz_cat], ignore_index=True)
        data_val = pd.concat([data_visit_val, data_mea_val, data_zz_val, data_sign_stat], ignore_index=True)
        data_val['value'] = data_val['value'].astype(str)
        data_val = data_val.where(data_val.notnull(), '')
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_cat is {}".format(data_cat.shape))
        print("[info] The shape of data_val is {}".format(data_val.shape))
        # print(data_val.dtypes)
        print("-----------------------------------------------------------------------")
        # pivot the table
        # data_cat = data_cat.pivot(index='visit_record_id_new_1', columns='name', values='value').reset_index()
        # data_val = data_val.pivot(index='visit_record_id_new_1', columns='name', values='value').reset_index()

        # self.data_insert(data_cat, self.table_feature_cat_name)
        print("-----------------------------------------------------------------------")
        print("[info] The data_cat have been inserted into oracle ")
        print("-----------------------------------------------------------------------")
        # self.data_insert(data_val, self.table_feature_val_name)
        print("-----------------------------------------------------------------------")
        print("[info] The data_val have been inserted into oracle ")
        print("-----------------------------------------------------------------------")
        with open('/mnt/data/wzx/jupyter_notebook/HC4FUO/data/p2_feature.pickle', 'wb') as f:
            pickle.dump((data_cat, data_val), f, -1)
        print("-----------------------------------------------------------------------")
        print("[info] The data_val and data_cat have been inserted into pickle file")
        print("-----------------------------------------------------------------------")
    
    def drop_col(self, df, col_name, cutoff, type):
        
        # 分别对nan值占比过高和偏振分布的列进行删除
        if type == 'nan':
            cnt = df[col_name].count()
            if (float(cnt) / len(df)) < cutoff:
                df.drop(col_name, axis=1, inplace=True)
                print("[info] category: the nan ratio of column {} are {} and have been droped".format(col_name, float(cnt) / len(df)))
        if type == 'maldis':
            vc = pd.DataFrame(df[col_name].value_counts())
            vc.reset_index(inplace=True)
            if len(vc) == 1:
                df.drop(col_name, axis=1, inplace=True)
            elif len(vc) > 1 and (float(vc.iloc[0, 1]/vc[col_name].sum())) > cutoff:
                df.drop(col_name, axis=1, inplace=True)
                print("[info] number: the max_count ratio of column {} are {} and have been droped".format(col_name, float(vc.iloc[0, 1]/vc[col_name].sum())))

    def feature_process(self):

        # read the table 
        # data_cat = pd.read_sql_table(self.table_feature_cat_name, self.db_conn.conn)
        # data_val = pd.read_sql_table(self.table_feature_val_name, self.db_conn.conn)
        f = open('/mnt/data/wzx/jupyter_notebook/HC4FUO/data/p2_feature.pickle', 'rb')
        (data_cat, data_val) = pickle.load(f)
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_cat is {}".format(data_cat.shape))
        print("[info] The unique number of name is {}".format(len(data_cat['name'].unique())))
        print("[info] The shape of data_val is {}".format(data_val.shape))
        print("[info] The unique number of name is {}".format(len(data_val['name'].unique())))
        print("-----------------------------------------------------------------------")
        
        # pivot the table
        data_cat = data_cat.pivot(index='visit_record_id_new_1', columns='name', values='value').reset_index()
        data_val = data_val.pivot(index='visit_record_id_new_1', columns='name', values='value').reset_index()

        data_cat = data_cat.fillna(value= np.nan)
        data_val = data_val.fillna(value= np.nan)
        print("-----------------------------------------------------------------------")
        print("[info] The shape of data_cat is {}".format(data_cat.shape))
        print("[info] The shape of data_val is {}".format(data_val.shape))
        print("-----------------------------------------------------------------------")

        # drop the columns that the ratio of nan is very high
        # col_list = data_cat.columns.values.tolist()
        # for col in col_list:
        #     self.drop_col(data_cat, col, 0.5, type='nan')
        # print("[message]: the dimensions of feature matrix of data_cat are {}".format(data_cat.shape))
        # # drop the columns that the ratio of max value_counts beyond some kind cutoff
        # col_list_mvc = data_cat.columns.values.tolist()
        # for col in col_list_mvc:
        #     self.drop_col(data_cat, col, 0.9, type='maldis')
        # print("[message]: the dimensions of feature matrix of data_cat are {}".format(data_cat.shape))       


        # drop the columns that the ratio of nan is very high
        # col_list = data_val.columns.values.tolist()
        # for col in col_list:
        #     self.drop_col(data_val, col, 0.5, type='nan')
        # print("[message]: the dimensions of feature matrix of data_val are {}".format(data_val.shape))
        # # drop the columns that the ratio of max value_counts beyond some kind cutoff
        # col_list_mvc = data_val.columns.values.tolist()
        # for col in col_list_mvc:
        #     self.drop_col(data_val, col, 0.9, type='maldis')
        # print("[message]: the dimensions of feature matrix of data_val are {}".format(data_val.shape)) 

        # output the new matrix to csv
        base_dir = os.path.dirname(os.path.realpath('__file__'))
        csvpath_cat = os.path.join(base_dir+"/data/", 'p2_feature_cat.csv')
        csvpath_val = os.path.join(base_dir+"/data/", 'p2_feature_val.csv')
        data_val.to_csv(csvpath_val, header=True, index=True)
        data_cat.to_csv(csvpath_cat, header=True, index=True)

class LabelComplementV2():
    """
    In:
        db_name: the SID of database

    """
    def __init__(self, db_name):
        self.db_name = db_name
        self.db_conn = db_connection(db_name)
        self.WORKING_PATH = '.'
        self.data = self.getData()
        self.diagnosis = self.getExplicitDiagnosis()
        self.bacterial, self.viral, self.fungal, self.parasitic, self.otherinfec, self.hm, self.sm, self.bt, self.aid, self.afd, self.other_noninfec = self.read_dict()

    def getData(self):

        sql = '''
        select visit_record_id, 
               diag_name, 
               diag_append, 
               flag_v2, 
               source_liu, 
               source_wang,
               source_iii_tag
        from label4visit
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        data = pd.DataFrame(results, columns=['visit_record_id','diag_name','diag_append','flag_v2','source_liu','source_wang', 'source_iii_tag'])
        print("[Info]: The number of visit_record_id is {}".format(data.shape[0]))

        return data

    def getExplicitDiagnosis(self):

        diagnosis = {}

        sql = '''
        select distinct 诊断名称 from label0818 where flag='1-1'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_bacterial = [x.lower() for (x,) in results]
        diagnosis['1-1'] = diag_bacterial

        sql = '''
        select distinct 诊断名称 from label0818 where flag='1-2'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_viral = [x.lower() for (x,) in results]
        diagnosis['1-2'] = diag_viral

        sql = '''
        select distinct 诊断名称 from label0818 where flag='1-3'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_fungal = [x.lower() for (x,) in results]
        diagnosis['1-3'] = diag_fungal

        sql = '''
        select distinct 诊断名称 from label0818 where flag='1-4'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_parasitic = [x.lower() for (x,) in results]
        diagnosis['1-4'] = diag_parasitic

        sql = '''
        select distinct 诊断名称 from label0818 where flag='1-5'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_otherinfec = [x.lower() for (x,) in results]
        diagnosis['1-5'] = diag_otherinfec

        sql = '''
        select distinct 诊断名称 from label0818 where flag='2-1-1'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_aid = [x.lower() for (x,) in results]
        diagnosis['2-1-1'] = diag_aid

        sql = '''
        select distinct 诊断名称 from label0818 where flag='2-1-2'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_afd = [x.lower() for (x,) in results]
        diagnosis['2-1-2'] = diag_afd

        sql = '''
        select distinct 诊断名称 from label0818 where flag='2-2-1'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_hm = [x.lower() for (x,) in results]
        diagnosis['2-2-1'] = diag_hm

        sql = '''
        select distinct 诊断名称 from label0818 where flag='2-2-2'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_sm = [x.lower() for (x,) in results]
        diagnosis['2-2-2'] = diag_sm

        sql = '''
        select distinct 诊断名称 from label0818 where flag='2-2-3'
        '''
        results = self.db_conn.conn.execute(sql).fetchall()
        diag_bt = [x.lower() for (x,) in results]
        diagnosis['2-2-3'] = diag_bt

        return diagnosis

    def read_dict(self):
        bacterial, viral, fungal, parasitic, otherinfec, hm, sm, bt, aid, afd, other_noninfec = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        with codecs.open(os.path.join(self.WORKING_PATH, 'config', 'label_tag.txt'), "r", encoding = "utf-8") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                if line.startswith('%') and line[2:] == '病毒性':
                    target_dict = viral
                elif line.startswith('%') and line[2:] == '细菌性':
                    target_dict = bacterial
                elif line.startswith('%') and line[2:] == '真菌性':
                    target_dict = fungal
                elif line.startswith('%') and line[2:] == '寄生虫性':
                    target_dict = parasitic
                elif line.startswith('%') and line[2:] == '其它感染':
                    target_dict = otherinfec
                elif line.startswith('%') and line[2:] == '自身炎症性':
                    target_dict = afd
                elif line.startswith('%') and line[2:] == '自身免疫性':
                    target_dict = aid
                elif line.startswith('%') and line[2:] == '血液系统恶性疾病':
                    target_dict = hm
                elif line.startswith('%') and line[2:] == '良性肿瘤':
                    target_dict = bt
                elif line.startswith('%') and line[2:] == '实体恶性肿瘤':
                    target_dict = sm
                elif line.startswith('%') and line[2:] == '其它非感染性':
                    target_dict = other_noninfec
                else:
                    items = line.split(' ||| ')
                    assert len(items) == 2
                    key = items[0]
                    value = items[1].split()
                    target_dict[key] = value

        return bacterial, viral, fungal, parasitic, otherinfec, hm, sm, bt, aid, afd, other_noninfec

    def lastChance(self, row):
        sql = '''
        select diag_name, 
               diag_append 
        from condition 
        where diag_type='出院诊断' and visit_record_id='{}'
        '''.format(row['visit_record_id'])
        results = self.db_conn.conn.execute(sql).fetchall()
        if results:
            diag, diag_append = [], []
            for r in results:
                diag.append(r[0])
                diag_append.append(r[1])
            diag_append = list(filter(None, diag_append))
            label_diag, label_diagappend = {}, {}
            if len(diag_append) != 0:
                for da in diag_append:
                    for t in re.split(r'[，。,;；、 ]', da):
                        if '待排' in t or '待查' in t or '？' in t or '?' in t or '排除' in t or '除外' in t :
                            continue
                        for key, values in self.bacterial.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["1-1"] =  1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.viral.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["1-2"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.fungal.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["1-3"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.parasitic.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["1-4"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.otherinfec.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["1-5"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.aid.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["2-1-1"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.afd.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["2-1-2"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.hm.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["2-2-1"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.sm.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["2-2-2"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.bt.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["2-2-3"] = 1
                                    break
                            else:
                                continue
                            break
                        for key, values in self.other_noninfec.items():
                            for value in values:
                                if value.lower() in t.lower():
                                    label_diagappend["2-3"] = 1
                                    break
                            else:
                                continue
                            break
            if len(diag) != 0:
                for da in diag:
                    for key, values in self.diagnosis.items():
                        if da.lower() in values:
                            label_diag[key] = 1

            label = [label_diag, label_diagappend]
            label_final = {}
            for item in label:
                for key, value in item.items():
                    label_final.setdefault(key, []).append(value)

            if len(label_final) == 1:
                sql = '''
                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                '''.format(list(label_final.keys())[0], '4', row['visit_record_id'])
                self.db_conn.conn.execute(sql)

            elif len(label_final) > 1:
                sql = '''
                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                '''.format('未知','4', row['visit_record_id'])
                self.db_conn.conn.execute(sql)
            else:
                if row['source_iii_tag'] is not None:
                    # complete the label with source_iii_tag
                    source_iii = list(row['source_iii_tag'])
                    count_dict = Counter(source_iii)

                    if set(source_iii).issubset(['A','B','C','D','J','L']):
                        if len(set(source_iii)) == 1 and source_iii[0] == 'A':
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('1-1','5', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        elif (len(set(source_iii)) == 1 and source_iii[0] == 'B') or max(count_dict, key=count_dict.get) == 'B':
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('1-2','5', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        elif (len(set(source_iii)) == 1 and source_iii[0] == 'C') or max(count_dict, key=count_dict.get) == 'C':
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('1-3','5', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        elif (len(set(source_iii)) == 1 and source_iii[0] == 'D') or max(count_dict, key=count_dict.get) == 'D':
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('1-4','5', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        elif (len(set(source_iii)) == 1 and source_iii[0] in ['J', 'L']) or max(count_dict, key=count_dict.get) in ['J', 'L']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('1-5','5', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        else:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('未知','5', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                    else:
                        sql = '''
                        update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                        '''.format('未知','5', row['visit_record_id'])
                        self.db_conn.conn.execute(sql)
                else:
                    sql = '''
                    update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                    '''.format('未知','5', row['visit_record_id'])
                    self.db_conn.conn.execute(sql)
        else:
            sql = '''
            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
            '''.format('无出院诊断','5', row['visit_record_id'])
            self.db_conn.conn.execute(sql)

    def processing(self):
        '''
        The meaning of flag_py_tag:
        0: the label checked by doc liu
        1: the label checked by wang
        2: the label ensured manually one by one on diagnosis
        3: the label come from diag_append column
        4: the label come from other diagnosis and diag_append
        5: the label come from drug
        '''
        count = 0
        for index, row in self.data.iterrows():
            print("[Info] the {}-th visit label is processing".format(count))
            count += 1
            if row['diag_name'] is not None:
                if row['source_liu'] is not None:
                    sql = '''
                    update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                    '''.format(row['source_liu'], '0', row['visit_record_id'])
                    self.db_conn.conn.execute(sql)

                elif row['source_wang'] is not None and row['source_wang'] != '2-3':
                    sql = '''
                    update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                    '''.format(row['source_wang'], '1', row['visit_record_id'])
                    self.db_conn.conn.execute(sql)

                elif row['flag_v2'] in ['1-1', '1-2', '1-3', '1-4', '1-5', '2-1-1', '2-1-2', '2-2-1', '2-2-2', '2-2-3']:
                    sql = '''
                    update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                    '''.format(row['flag_v2'], '2', row['visit_record_id'])
                    self.db_conn.conn.execute(sql)

                    if row['diag_append'] is not None:
                        label_append = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.hm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.sm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.bt.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-3"] = 1
                                        break
                                else:
                                    continue
                                break
                        if len(label_append) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_append.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_append) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        else:
                            continue
                    else:
                        continue
                # the infectious visit but without explicit type
                elif row['flag_v2'] == '1':

                    # deal the visit with diag_append
                    if row['diag_append'] is not None:
                        label_append = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.hm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.sm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.bt.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-3"] = 1
                                        break
                                else:
                                    continue
                                break
                        if len(label_append) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_append.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_append) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        
                        # deal with the row with diag_append but extract nothing
                        else:
                            if row['diag_name'] == '感染性发热':
                                if row['source_iii_tag'] is not None:
                                    # complete the label with source_iii_tag
                                    source_iii = list(row['source_iii_tag'])
                                    count_dict = Counter(source_iii)

                                    if set(source_iii).issubset(['A','B','C','D','J','L']):
                                        if len(set(source_iii)) == 1 and source_iii[0] == 'A':
                                            sql = '''
                                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                                '''.format('1-1','5', row['visit_record_id'])
                                            self.db_conn.conn.execute(sql)
                                        elif (len(set(source_iii)) == 1 and source_iii[0] == 'B') or max(count_dict, key=count_dict.get) == 'B':
                                            sql = '''
                                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                                '''.format('1-2','5', row['visit_record_id'])
                                            self.db_conn.conn.execute(sql)
                                        elif (len(set(source_iii)) == 1 and source_iii[0] == 'C') or max(count_dict, key=count_dict.get) == 'C':
                                            sql = '''
                                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                                '''.format('1-3','5', row['visit_record_id'])
                                            self.db_conn.conn.execute(sql)
                                        elif (len(set(source_iii)) == 1 and source_iii[0] == 'D') or max(count_dict, key=count_dict.get) == 'D':
                                            sql = '''
                                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                                '''.format('1-4','5', row['visit_record_id'])
                                            self.db_conn.conn.execute(sql)
                                        elif (len(set(source_iii)) == 1 and source_iii[0] in ['J', 'L']) or max(count_dict, key=count_dict.get) in ['J', 'L']:
                                            sql = '''
                                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                                '''.format('1-5','5', row['visit_record_id'])
                                            self.db_conn.conn.execute(sql)
                                        else:
                                            sql = '''
                                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                                '''.format('未知','5', row['visit_record_id'])
                                            self.db_conn.conn.execute(sql)
                                    else:
                                        sql = '''
                                        update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                        '''.format('未知','5', row['visit_record_id'])
                                        self.db_conn.conn.execute(sql)
                                else:
                                    sql = '''
                                    update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                    '''.format('未知','5', row['visit_record_id'])
                                    self.db_conn.conn.execute(sql)
                            else:
                                self.lastChance(row)

                    # deal the visit without diag_append
                    else:
                        if row['diag_name'] == '感染性发热':
                            if row['source_iii_tag'] is not None:
                                # complete the label with source_iii_tag
                                source_iii = list(row['source_iii_tag'])
                                count_dict = Counter(source_iii)

                                if set(source_iii).issubset(['A','B','C','D','J','L']):
                                    if len(set(source_iii)) == 1 and source_iii[0] == 'A':
                                        sql = '''
                                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                            '''.format('1-1','5', row['visit_record_id'])
                                        self.db_conn.conn.execute(sql)
                                    elif (len(set(source_iii)) == 1 and source_iii[0] == 'B') or max(count_dict, key=count_dict.get) == 'B':
                                        sql = '''
                                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                            '''.format('1-2','5', row['visit_record_id'])
                                        self.db_conn.conn.execute(sql)
                                    elif (len(set(source_iii)) == 1 and source_iii[0] == 'C') or max(count_dict, key=count_dict.get) == 'C':
                                        sql = '''
                                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                            '''.format('1-3','5', row['visit_record_id'])
                                        self.db_conn.conn.execute(sql)
                                    elif (len(set(source_iii)) == 1 and source_iii[0] == 'D') or max(count_dict, key=count_dict.get) == 'D':
                                        sql = '''
                                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                            '''.format('1-4','5', row['visit_record_id'])
                                        self.db_conn.conn.execute(sql)
                                    elif (len(set(source_iii)) == 1 and source_iii[0] in ['J', 'L']) or max(count_dict, key=count_dict.get) in ['J', 'L']:
                                        sql = '''
                                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                            '''.format('1-5','5', row['visit_record_id'])
                                        self.db_conn.conn.execute(sql)
                                    else:
                                        sql = '''
                                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                            '''.format('未知','5', row['visit_record_id'])
                                        self.db_conn.conn.execute(sql)
                                else:
                                    sql = '''
                                    update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                    '''.format('未知','5', row['visit_record_id'])
                                    self.db_conn.conn.execute(sql)
                            else:
                                sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('未知','5', row['visit_record_id'])
                                self.db_conn.conn.execute(sql)
                        else:
                            self.lastChance(row)

                elif row['flag_v2'] == '1+2':

                    # deal the visit with diag_append
                    if row['diag_append'] is not None:
                        
                        if row['diag_name'] == '甲状腺炎' and '亚急性' in row['diag_append']:
                            sql = '''
                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                            '''.format('1-2', '3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                            continue

                        label_append = {}
                        label_append_diag = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.hm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.sm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.bt.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-3"] = 1
                                        break
                                else:
                                    continue
                                break

                            for key, values in self.diagnosis.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append_diag[key] = 1
                                        break
                        
                        label = [label_append, label_append_diag]
                        label_final = {}
                        for item in label:
                            for key, value in item.items():
                                label_final.setdefault(key, []).append(value)

                        if len(label_final) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_final.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_final) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        
                        # deal with the row with diag_append but extract nothing
                        else:
                            self.lastChance(row)

                    # deal the visit without diag_append
                    else:
                        self.lastChance(row)

                elif row['flag_v2'] == '1-0':
                    # deal the visit with diag_append
                    if row['diag_append'] is not None:
                        label_append = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                        if len(label_append) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_append.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_append) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        
                        # deal with the row with diag_append but extract nothing
                        else:  
                            self.lastChance(row)

                    # deal the visit without diag_append
                    else:
                        self.lastChance(row)

                elif row['flag_v2'] == '2-2':
                    # deal the visit with diag_append
                    if row['diag_append'] is not None:  
                        label_append = {}
                        label_append_diag = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.hm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.sm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.bt.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.diagnosis.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append_diag[key] = 1
                                        break
                        
                        label = [label_append, label_append_diag]
                        label_final = {}
                        for item in label:
                            for key, value in item.items():
                                label_final.setdefault(key, []).append(value)

                        if len(label_final) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_final.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_final) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        
                        # deal with the row with diag_append but extract nothing
                        else:
                            self.lastChance(row)

                    # deal the visit without diag_append
                    else:
                        self.lastChance(row)

                elif row['flag_v2'] == '2-3':
                    # deal the visit with diag_append
                    if row['diag_append'] is not None:
                        label_append = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.hm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.sm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.bt.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-3"] = 1
                                        break
                                else:
                                    continue
                                break
                        if len(label_append) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_append.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_append) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        
                        # deal with the row with diag_append but extract nothing
                        else:  
                            sql = '''
                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                            '''.format('未知','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                    # deal the visit without diag_append
                    else:
                        sql = '''
                        update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                        '''.format('未知','3', row['visit_record_id'])
                        self.db_conn.conn.execute(sql)

                elif row['flag_v2'] == '术后':
                    sql = '''
                    update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                    '''.format('术后', '2', row['visit_record_id'])
                    self.db_conn.conn.execute(sql)

                elif row['flag_v2'] == '剔除':
                    # deal the visit with diag_append
                    if row['diag_append'] is not None:
                        label_append = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.hm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.sm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.bt.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-3"] = 1
                                        break
                                else:
                                    continue
                                break
                        if len(label_append) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_append.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_append) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        
                        # deal with the row with diag_append but extract nothing
                        else:  
                            sql = '''
                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                            '''.format('发热待查','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                    # deal the visit without diag_append
                    else:
                        sql = '''
                        update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                        '''.format('发热待查','3', row['visit_record_id'])
                        self.db_conn.conn.execute(sql)
                
                else:
                    # deal the visit with diag_append
                    if row['diag_append'] is not None:
                        label_append = {}
                        for t in re.split(r'[，。,;；、 ]', row['diag_append']):
                            if '待排' in t or '待查' in t or '排除' in t or '除外' in t or '?' in t or '？' in t:
                                continue
                            for key, values in self.bacterial.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.viral.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.fungal.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-3"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.parasitic.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-4"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.otherinfec.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["1-5"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.aid.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.afd.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-1-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.hm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-1"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.sm.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-2"] = 1
                                        break
                                else:
                                    continue
                                break
                            for key, values in self.bt.items():
                                for value in values:
                                    if value.lower() in t.lower():
                                        label_append["2-2-3"] = 1
                                        break
                                else:
                                    continue
                                break
                        if len(label_append) == 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format(list(label_append.keys())[0],'3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
        
                        elif len(label_append) > 1:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                        elif '合并感染' in row['diag_append'] or '混合感染' in row['diag_append'] or '复合感染' in row['diag_append']:
                            sql = '''
                                update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                                '''.format('合并感染','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)
                        
                        # deal with the row with diag_append but extract nothing
                        else:  
                            sql = '''
                            update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                            '''.format('未知','3', row['visit_record_id'])
                            self.db_conn.conn.execute(sql)

                    # deal the visit without diag_append
                    else:
                        sql = '''
                        update label_v2 set flag_py='{0}', flag_py_tag='{1}' where visit_record_id='{2}'
                        '''.format('未知','3', row['visit_record_id'])
                        self.db_conn.conn.execute(sql)
            
            # the diag_name is None
            else:
                self.lastChance(row)

    def postProcessing(self):

        # to merge the label
        # sql = '''
        # update label_v2 set flag_py='未知', flag_py_tag='0' where flag_py='0-0'
        # '''
        # self.db_conn.conn.execute(sql)

        # sql = '''
        # update label_v2 set flag_py='未知', flag_py_tag='0' where flag_py='0-1'
        # '''
        # self.db_conn.conn.execute(sql)

        # sql = '''
        # update label_v2 set flag_py='合并感染', 
        #                     flag_py_tag='0' 
        # where flag_py in ('1-1+2','1-1+2+3','1-1+3','1-2+3')
        # '''
        # self.db_conn.conn.execute(sql)

        # sql = '''
        # update label_v2 set flag_py='怀孕', 
        #                     flag_py_tag='0' 
        # where diag_name like '%孕%' or diag_append like '%孕%'
        # '''
        # self.db_conn.conn.execute(sql)

        # sql = '''
        # update label_v2 set flag_py='发热待查', 
        #                     flag_py_tag='0' 
        # where flag_py='剔除'
        # '''
        # self.db_conn.conn.execute(sql)
        
        # process the patients with multi-visits
        # take the label of last visit_record_id in one visit_record_id_new_1
        # sql = '''
        # select distinct(visit_record_id_new_1) from label_v2
        # '''
        # results = self.db_conn.conn.execute(sql).fetchall()
        # vri_new = [x for (x,) in results]

        # for vri in vri_new:
        #     sql = '''
        #     select visit_record_id from label_v2 where visit_record_id_new_1 = '{}' order by visit_start_date
        #     '''.format(vri)
        #     results = self.db_conn.conn.execute(sql).fetchall()
            
        #     sql = '''
        #     update label_v2 set flag_visit='1' where visit_record_id='{}'
        #     '''.format(results[-1][0])
        #     self.db_conn.conn.execute(sql)

        # update the flag_py label into label1, label2, label3, label4, label5
        # infectious and non-infectious
        sql = '''
        update label_v2 set label1='0' where flag_py in ('1-1', '1-2','1-3','1-4','1-5')
        '''
        self.db_conn.conn.execute(sql)
        sql = '''
        update label_v2 set label1='1' where flag_py in ('2-1-1', '2-1-2','2-2-1','2-2-2')
        '''
        self.db_conn.conn.execute(sql)

        # bacterial, viral, fungal, parasitic, others
        sql = '''
        update label_v2 set label2='0' where flag_py='1-1'
        '''
        self.db_conn.conn.execute(sql)
        sql = '''
        update label_v2 set label2='1' where flag_py='1-2'
        '''
        self.db_conn.conn.execute(sql)
        sql = '''
        update label_v2 set label2='2' where flag_py in ('1-3', '1-4', '1-5')
        '''
        self.db_conn.conn.execute(sql)
        # sql = '''
        # update label_v2 set label2='3' where flag_py='1-4'
        # '''
        # self.db_conn.conn.execute(sql)
        # sql = '''
        # update label_v2 set label2='4' where flag_py='1-5'
        # '''
        # self.db_conn.conn.execute(sql)

        # NIID and Neo-
        sql = '''
        update label_v2 set label3='0' where flag_py in ('2-1-1','2-1-2')
        '''
        self.db_conn.conn.execute(sql)
        sql = '''
        update label_v2 set label3='1' where flag_py in ('2-2-1','2-2-2')
        '''
        self.db_conn.conn.execute(sql)

        # afd and aid
        sql = '''
        update label_v2 set label4='0' where flag_py='2-1-1'
        '''
        self.db_conn.conn.execute(sql)
        sql = '''
        update label_v2 set label4='1' where flag_py='2-1-2'
        '''
        self.db_conn.conn.execute(sql)

        # hm, sm and bt
        sql = '''
        update label_v2 set label5='0' where flag_py='2-2-1'
        '''
        self.db_conn.conn.execute(sql)
        sql = '''
        update label_v2 set label5='1' where flag_py='2-2-2'
        '''
        self.db_conn.conn.execute(sql)
        # sql = '''
        # update label_v2 set label5='2' where flag_py='2-2-3'
        # '''
        # self.db_conn.conn.execute(sql)

        self.db_conn.conn.close()

def _feature_csv_process():
    """
    主要是对空格进行去除，部分数值型特征内的异常文字进行nan，大于小于号去除等异常字符的处理，删除部分暂时无法使用的字段
    """

    csv_file = '/mnt/data/wzx/jupyter-notebook/Regex Test/output/p0_feature_l1.csv'
    csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
    fm_csv = pd.DataFrame(csv_data)

    # delete the space between strings
    fm=fm_csv.applymap((lambda x:"".join(x.split()) if type(x) is str else x))

    # inplace some kind of str
    fm.replace({'28楼填表重抽':'', '2-8楼填表复检':'', 'HIV感染待确定':'', 'HIV抗体待复查':'', '标本需重抽':'', '标本重抽':'', '重抽复检':'', '标本重抽复检':'', '感染待确定':'', '送检':'', '填表送确认':'', '填表送确证':'', '复检':'', '填表重抽':'', '填表重抽复检':'', '已确证':'', '0.50已复查':'0.50', 'HIV感染待确证':'', 'HIV抗体待复检':'', '感染待确证':'', '送疾控确证':''}, inplace=True)
    fm.replace({'。':'有', '否认':'无', '好':'良好', '尚可':'良好', '体健':'良好', '干扰':'', '，':'有', '偶':'有', '偶尔':'有', 'LAMPL.':'', '黄疸':'', '可':'良好', '良':'良好'}, inplace=True)
    fm.replace({'软':'软便', '糊':'糊状便', '糊便':'糊状便', '糊粘稠':'糊状便', '糊状便带血':'糊状便', '烂':'烂便', '稀':'稀便', '烂便带血丝':'烂便', '黏液便':'黏液稀便', '软便带粘液':'软便', '水样':'水样便', '水样稀':'水样便', '已干':'硬便', '硬':'硬便', '粘液':'粘液便', '糊状便带血丝':'', '黏液稀便':'粘液便', '粘液':'粘液便'}, inplace=True)
    fm.replace({'建议血液中心':'', 'UNDER':'', '临界值':'', 'A':'A型', 'AB':'AB型', 'B':'B型', 'O':'O型', '正定AB型':'AB型', '正定A型':'A型', '正定B型':'B型', '正定O型':'O型'}, inplace=True)
    fm.replace({'溶血干扰':'', '黄疸干扰':'', '冷凝集': '', '凝集':'', 'RBC双峰':'', 'OVER':'', '脂浊干扰':'', '脂浊':'', '88.7已复':'88.7', '严重溶血':'', '溶血':'', '干扰复查':'', 'PLT聚集':''}, inplace=True)
    fm.replace(regex=r'。+', value='', inplace=True)
    fm.replace(regex=r'>+', value='', inplace=True)
    fm.replace(regex=r'<+', value='', inplace=True)
    fm.replace(regex=r'\+', value='', inplace=True)
    fm.replace(regex=r'-+', value='', inplace=True)
    fm.replace(regex=r'\*+', value='', inplace=True)
    fm.replace(regex=r'\(手工\)', value='', inplace=True)
    fm.replace(regex=r'阴性\d+(.)*\d+', value='阴性', inplace=True)
    fm.replace(regex=r'A型\s*\S+', value='A型', inplace=True)
    fm.replace(regex=r'B型\s*\S+', value='B型', inplace=True)
    fm.replace(regex=r'O型\s*\S+', value='O型', inplace=True)
    fm.replace(regex=r'(\d+)已复查', value=r'\1', inplace=True)
    fm.replace(regex=r'阴性\d+', value='阴性', inplace=True)
    fm.replace(regex=r'阳性\d+(.)*\d+', value='阳性', inplace=True)
    fm.replace(regex=r'(\d+)(溶血|干扰)', value=r'\1', inplace=True)
    fm.replace(regex=r'(\d+)聚集', value=r'\1', inplace=True)
    fm.replace({'120s':'120', '40.0()':'40.0', '150()':'150', '.':'', '0.0.':'', '未找到':''}, inplace=True)
    fm.replace({'血小板聚集':'', '血凝':'', '聚集':'','PLT聚集':'', '59（部分大PLT）':'59','58(已复)':'58', '56(已复)':'56', '45,大PLT':'45', '42,轻度聚集':'42', '41（已复）':'41', '40(已复)':'40', '34(见大PLT)':'34', 'REM':'', 'RBC自凝':'', 'LEV':'', '〉140':'140', 'ERROR':'', '64,聚集':'64', '184.70YIFUCHA': '184.70', '30手工':'30', '2.80已复':''}, inplace=True)

    # the columns will be droped in the training
    del_col_list = ['不消化食物-大便', '亚硝酸盐-尿液', '尿胆原-尿液', '浊度-尿液', '白细胞-大便', '白细胞酯酶-尿液', '红细胞-大便', '隐血试验-大便', '胆红素-尿液', '蛋白质-尿液', '脓细胞-大便', '葡萄糖-尿液', '酮体-尿液', '隐血-尿液', '中毒史', '其他传染病史', '冶游史', '外伤史', '婚姻状况', '学历', '手术史', '毒性及放射性物质接触史', '疫区居留史', '病毒性肝炎史', '职业', '肺结核史', '输血史', '长期用药史', '食物药物过敏史', '颜色-尿液', '颜色（大便）-大便']
    fm.drop(del_col_list, axis=1, inplace=True)
    print("[message]: the dimensions of csv data after cleaning are {}".format(fm.shape))

    # output the new matrix to csv
    base_dir = os.path.dirname(os.path.realpath('__file__'))
    csvpath = os.path.join(base_dir+"/Regex Test/output/", 'p0_feature_l2.csv')
    fm.to_csv(csvpath, header=True, index=True)

    return fm

def _feature_preprocessing():

    csv_file = '/mnt/data/wzx/jupyter-notebook/Regex Test/output/p0_feature_l2.csv'
    csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
    fm = pd.DataFrame(csv_data)
    fm.drop(['VISIT_RECORD_ID_NEW_1','Unnamed: 0'], axis=1, inplace=True)

    # split the feature and label
    fm_y = fm[['LABEL']]
    col = fm.columns.values.tolist()
    col.remove('LABEL') 
    fm_x = fm[col].copy()
    print("[Message]: the dimensions of x is {}".format(fm_x.shape))
    print("[Message]: the dimensions of y is {}".format(fm_y.shape))

def _position_dictfile_pre():

    posi_sign = FileUtils.read_txt_file("/mnt/data/wzx/jupyter_notebook/Regex_Test/config/position_dict.txt")
    posi_dict = list(set(posi_sign))
    posi_dict.sort()
    FileUtils.write_txt_file(posi_dict, "/mnt/data/wzx/jupyter_notebook/Regex_Test/config/position_dict.txt")



if __name__ == '__main__':

    # 基于导出的csv，进行某些列的删除，字段更替
    # _feature_csv_process()

    # 对主诉内症状的位置字典进行去重与排序
    # _position_dictfile_pre()

    # 对各个特征进行分布描述和异常值检测，进行处理，构建患者特征矩阵
    fa = FeatureAggre('fuo', 'fuo_sys', 'P0_NOTE_JWS', 'P0_NOTE_GRS', 'P0_VISIT_ZY', 'P0_MEASUREMENT')
    fa.feature_matrix()