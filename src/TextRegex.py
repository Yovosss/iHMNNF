#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@author: wang zhixiao
@time  : 2020-12-16 11:36
@detail: 对患者主诉、既往史、婚姻史、现病史、家族史与首次病程记录切分后的子文本类型进行结构化信息提取
"""

import re
import os
import cx_Oracle
import codecs
import pandas as pd

from utils.utils import FileUtils
from config.orcl import db_connection
from sqlalchemy import Table, select, text, distinct, types


class TextRegex():

    def __init__(self, db_name, out_table):

        self.db_conn = db_connection(db_name)
        self.out_table_name = out_table
        self.global_sign_dictfile = 'global_sign_dict.txt'
        self.locally_sign_dict = 'locally_sign_dict.txt'
        self.pos_dict = 'position_dict.txt'
        self.dis_dictfile = 'diseasedict.txt'

        # read the dict file
        self.glob_sign = FileUtils.read_txt_file(os.path.dirname(__file__)+"/config/"+self.global_sign_dictfile)
        self.loca_sign = FileUtils.read_txt_file(os.path.dirname(__file__)+"/config/"+self.locally_sign_dict)
        self.posi_sign = FileUtils.read_txt_file(os.path.dirname(__file__)+"/config/"+self.pos_dict)
        self.disease = FileUtils.read_txt_file(os.path.dirname(__file__)+"/config/"+self.dis_dictfile)
        self.disease.sort(key = lambda i:len(i),reverse=True)

    def chiefcomregex(self, text_data):
        """
        对主诉文本进行结构化提取：症状名称-持续时间-频次
        param: text_data:文本数据，dictfile:症状字典文件
        output: structuredict

        """
        structuredict = {}
        record = text_data['record'].values[0]

        if record:

            record = re.sub(r"\\r\\n", r"\r\n", record)   #回车和换行，统一替换为\r或\n
            record = re.sub(r"\s{0,1}[:|：]\s{0,1}", ":", record)   #将英文与中文冒号，及其带空格的部分，全部替换为英文:
            record = re.sub(r"\s+", "", record)  #去除空格
            record = re.sub(r"\s*[""“”]+\s*", "", record)  #去除中英文引号
            record = re.sub(r"(?<!\d)[\.]+(?!\d)", "。", record)  #除了1.5样式表示数字的.之外，所有的.将被更换为。        
            record_tmp = re.split(r"[,|，|。|；|;|\r\n]+", record)
            count = 1

            for i in record_tmp:
                # match the global sign     
                for j in self.glob_sign:
                    regex_rule_glob = "(?P<{0}>({1})+)".format("glob_sign_name", j)
                    searchobj_glob = re.search(r"{}".format(regex_rule_glob), i)
                    if searchobj_glob:
                        # match the fever related sign, duration and frequency 
                        if searchobj_glob.group("glob_sign_name") in ["发热", "低热", "低度热", "高热", "高度热", "中低热", "中低度热", "发烧", "低烧", "高烧", "体温上升", "体温增高", "体温升高", "体温偏高", "体温反复增高", "体温反复升高", "体温反复偏高"]:
                            structuredict.setdefault("sign_{}".format(count), {})
                            structuredict["sign_{}".format(count)]["symptom"] = searchobj_glob.group("glob_sign_name")
                            i = i[:searchobj_glob.start()] + i[searchobj_glob.end():]

                            searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                            if searchobj_time:
                                structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                            else:
                                structuredict["sign_{}".format(count)]["duration"] = ""

                            searchobj_fre = re.search(r"(反复|间断|间歇|持续|阵发性|偶有|长期)", i)
                            if searchobj_fre:
                                structuredict["sign_{}".format(count)]["frequency"] = searchobj_fre.group()
                            else:
                                structuredict["sign_{}".format(count)]["frequency"] = ""
                            
                            structuredict["sign_{}".format(count)]["location"] = ""
                            structuredict["sign_{}".format(count)]["sign_type"] = "global"

                            count += 1
                        else:
                            structuredict.setdefault("sign_{}".format(count), {})
                            structuredict["sign_{}".format(count)]["symptom"] = searchobj_glob.group("glob_sign_name")
                            i = i[:searchobj_glob.start()] + i[searchobj_glob.end():]

                            searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                            if searchobj_time:
                                structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                            else:
                                structuredict["sign_{}".format(count)]["duration"] = ""

                            structuredict["sign_{}".format(count)]["frequency"] = ""
                            structuredict["sign_{}".format(count)]["location"] = ""
                            structuredict["sign_{}".format(count)]["sign_type"] = "global"
                            
                            count += 1
                    else:
                        continue

                # match the locally sign
                # to store the  matched position in the first loop
                matched_posi = []
                # to mark the order of matched local sign 
                loca_sign_tag = 1
                for m in self.loca_sign:
                    regex_rule_loca = "(?P<{0}>({1}){{1}})".format("loca_sign_name", m)
                    searchobj_loca = re.search(r"{}".format(regex_rule_loca), i)
                    if searchobj_loca:
                        # match the position dict
                        posi_tag = 1

                        if loca_sign_tag == 1:
                            loca_tag = 1
                            for n in self.posi_sign:

                                regex_rule_posi = "(?P<{0}>({1})+)".format("posi_name", n)
                                searchobj_posi = re.search(r"{}".format(regex_rule_posi), i)
                                if searchobj_posi:
                                
                                    structuredict.setdefault("sign_{}".format(count), {})
                                    structuredict["sign_{}".format(count)]["symptom"] = searchobj_loca.group("loca_sign_name")
                                    if loca_tag == 1:
                                        i = i[:searchobj_loca.start()] + i[searchobj_loca.end():]
                                    structuredict["sign_{}".format(count)]["location"] = searchobj_posi.group("posi_name")
                                    i = i[:searchobj_posi.start()] + i[searchobj_posi.end():]

                                    searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                    if searchobj_time:
                                        structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                    else:
                                        structuredict["sign_{}".format(count)]["duration"] = ""

                                    structuredict["sign_{}".format(count)]["frequency"] = ""
                                    structuredict["sign_{}".format(count)]["sign_type"] = "locally"

                                    matched_posi.append(searchobj_posi.group("posi_name"))
                                    
                                    loca_tag += 1
                                    count += 1

                                elif not searchobj_posi:
                                    posi_tag += 1

                                    if posi_tag == len(self.posi_sign) + 1:
                                        structuredict.setdefault("sign_{}".format(count), {})
                                        structuredict["sign_{}".format(count)]["symptom"] = searchobj_loca.group("loca_sign_name")
                                        structuredict["sign_{}".format(count)]["location"] = ""
                                        structuredict["sign_{}".format(count)]["frequency"] = ""

                                        searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                        if searchobj_time:
                                            structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                        else:
                                            structuredict["sign_{}".format(count)]["duration"] = ""

                                        structuredict["sign_{}".format(count)]["sign_type"] = "locally"

                                        count += 1
                                    else:
                                        continue
                            loca_sign_tag += 1
                        else:
                            for k in matched_posi:
                                structuredict.setdefault("sign_{}".format(count), {})
                                structuredict["sign_{}".format(count)]["symptom"] = searchobj_loca.group("loca_sign_name")
                                structuredict["sign_{}".format(count)]["location"] = k

                                # print(i)
                                searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                if searchobj_time:
                                    structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                else:
                                    structuredict["sign_{}".format(count)]["duration"] = ""

                                structuredict["sign_{}".format(count)]["frequency"] = ""
                                structuredict["sign_{}".format(count)]["sign_type"] = "locally"

                                count += 1                  
                    else:
                        continue

        else:
            structuredict = {}

        return structuredict

    def personalhisregex(self, text_data):
        """
        对个人史文本进行结构化提取：个人史-值
        param: text_data:文本数据
        output: structuredict

        """
        structuredict = {}
        record = text_data['record'].values[0]
        if record:

            record = re.sub(r"\\r\\n", r"\r\n", record)   #回车和换行，统一替换为\r或\n
            record = record.strip()    #前后去空格
            # 冒号统一
            record = re.sub(r"\s{0,1}[:|：]\s{0,1}", ":", record)   #将英文与中文冒号，及其带空格的部分，全部替换为英文:
            record = re.sub(r"\s+", "", record)  #去除空格
            record = re.sub(r"\s*[""“”]+\s*", "", record)  #去除中英文引号
            record = re.sub(r"(?<!\d)[\.]+(?!\d)", "。", record)  #除了1.5样式表示数字的.之外，所有的.将被更换为。

            #生成正则
            career_pattern = re.compile(r"(?P<career>(政府工作人员|商业服务|军人|微商|木工|厨师|辅警|农民|学生|退休|工人|事业编制|药师|药剂师|文员|职员|待业|个体|经商|会计|干部|自由职业|公务员|职工|务农|警察|教师|老师|医生|医师|护士|医务人员|医务工作者|司机|保安|工程师|程序员|财务|园林管理人员|门卫|无业|家庭妇女|主妇|设计|务工|清洁工|模特|服务|淘宝业主|家务|电商|销售|行政管理|画家|离休|餐饮|商人|木匠|服务行业|口腔医师|出纳|文职|搬运工|业务员|离退|律师))")
            searchobj_career = career_pattern.search(record)
            if searchobj_career:
                if searchobj_career.group("career") in ['木工', '农民', '工人', '务农', '务工', '清洁工', '木匠', '搬运工']:
                    structuredict["职业"] = "体力工作者"
                elif searchobj_career.group("career") in ['个体', '厨师', '司机', '保安', '园林管理人员', '门卫', '服务', '餐饮', '服务行业']:
                    structuredict["职业"] = "服务业者"
                elif searchobj_career.group("career") in ['药师', '药剂师', '医生', '医师', '护士', '医务人员', '医务工作者', '口腔医师']:
                    structuredict["职业"] = "医务工作者"
                elif searchobj_career.group("career") in ['政府工作人员', '军人', '辅警', '事业编制', '干部', '公务员', '警察']:
                    structuredict["职业"] = "政府工作者"
                elif searchobj_career.group("career") in ['学生', '教师', '老师']:
                    structuredict["职业"] = "教育工作者"
                elif searchobj_career.group("career") in ['退休', '离休', '离退']:
                    structuredict["职业"] = "离退休人员"
                elif searchobj_career.group("career") in ['设计', '模特', '画家', '自由职业']:
                    structuredict["职业"] = "文化事业者"
                elif searchobj_career.group("career") in ['微商', '商业服务', '经商', '淘宝业主', '电商', '商人']:
                    structuredict["职业"] = "经商职业者"
                elif searchobj_career.group("career") in ['文员', '职员', '会计', '职工', '工程师', '程序员', '财务', '销售', '行政管理', '出纳', '文职', '业务员', '律师']:
                    structuredict["职业"] = "白领工作者"
                elif searchobj_career.group("career") in ['待业', '无业', '家庭妇女', '主妇', '家务']:
                    structuredict["职业"] = "无业者"
            else:
                structuredict["职业"] = ""

            degree_pattern = re.compile(r"(?P<degree>(未读学历|未上过学学历|未上学|未接受过教育学历|未接受教育学历|未受教育学历|农民学历|无学历|文盲|识字学历|小学|高小学历|大小学历|一般学历|初中|初一|初二|初三|初小|中专|高中|高一|高二|高三|中学|职高|告知|高职|大专|专科|本科|大一|大二|大三|大四|师范学历|大学|硕士|研究生|博士|不详学历|学历不详|不清学历|其他学历|未知学历|未提供学历))")
            searchobj_degree = degree_pattern.search(record)
            # searchobj_degree = re.search(r"(?P<degree>((?<=，|。)(\S){1,5}(?=(文化|肄业|以上|及以上)?学历))|(学历不详)|(?<=学历：)(\S){1,5}(?=，| |。))", record)
            if searchobj_degree:
                if searchobj_degree.group("degree") in ['未读学历', '未上过学学历', '未上学', '未接受过教育学历', '未接受教育学历', '未受教育学历', '农民学历', '无学历', '文盲']:
                    structuredict["学历"] = "文盲"
                elif searchobj_degree.group("degree") in ['识字学历', '小学', '高小学历', '大小学历', '一般学历']:
                    structuredict["学历"] = "小学"
                elif searchobj_degree.group("degree") in ['初中', '初一', '初二', '初三', '初小', '中专']:
                    structuredict["学历"] = "初中"
                elif searchobj_degree.group("degree") in ['高中', '高一', '高二', '高三', '中学', '职高', '告知', '高职', '大专', '专科']:
                    structuredict["学历"] = "高中"
                elif searchobj_degree.group("degree") in ['本科', '大一', '大二', '大三', '大四', '师范学历', '大学']:
                    structuredict["学历"] = "大学"
                elif searchobj_degree.group("degree") in ['硕士', '研究生', '博士']:
                    structuredict["学历"] = "研究生"
                elif searchobj_degree.group("degree") in ['不详学历', '学历不详', '不清学历', '其他学历', '未知学历', '未提供学历']:
                    structuredict["学历"] = "不详"
            else:
               structuredict["学历"] = ""

            marital_pattern = re.compile(r"(.*)(?P<hunyinzhuangtai>((婚姻|家庭)(和睦|关系和睦|离婚))|离异|丧偶|未婚|不和睦|(婚姻关系：和睦)|丈夫已去世|婚姻和|离婚|丈夫已故|已离婚|配偶已逝|妻子已故|尚未结婚|妻子已逝|配偶已故|配偶去世)")
            searchobj_marry = marital_pattern.search(record)
            if searchobj_marry:
                if searchobj_marry.group("hunyinzhuangtai") in ['婚姻和睦', '家庭关系和睦', '家庭和睦', '婚姻关系和睦', '婚姻关系：和睦', '婚姻和']:
                    structuredict["婚姻状况"] = '婚姻和睦'
                elif searchobj_marry.group("hunyinzhuangtai") in ['不和睦', '离婚', '已离婚', '离异']:
                    structuredict["婚姻状况"] = '婚姻不和睦'
                elif searchobj_marry.group("hunyinzhuangtai") in ['丧偶', '丈夫已去世', '已离婚', '丈夫已故', '配偶已逝', '妻子已故', '妻子已逝', '配偶已故', '配偶去世']:
                    structuredict["婚姻状况"] = '丧偶'
                elif searchobj_marry.group("hunyinzhuangtai") in ['未婚', '尚未结婚']:
                    structuredict["婚姻状况"] = '未婚'
            else:
                structuredict["婚姻状况"] =""

            other_pattern = re.compile(r"(?P<yiqujuliushi>(有|无|否认){1}(?=疫区居留史|长期疫地居住史|疫地居住史|疫水(、)?疫源地接触史))(.*)(?P<yeyoushi>(有|无|否认){1}(?=冶游史))(.*)(?P<yinjiuxiguan>(有|无|否认|偶有|偶尔|。|，|偶){1}(?=饮酒习惯|饮酒|嗜酒|喝酒))(.*)(?P<xiyanshi>(有|无|否认|偶有|。|，|偶尔|偶){1}(?=吸烟习惯|吸烟|吸烟史|抽烟))(.*)(?P<duwufangshexingwuzhi>(有|无|否认){1}(?=毒物及放射性物质接触史|放射(\S)*接触史))")
            searchobj_other = other_pattern.search(record)
            if searchobj_other:
                if searchobj_other.group("yiqujuliushi"):
                    if searchobj_other.group("yiqujuliushi") in ['无', '否认']:
                        structuredict["疫区居留史"] = "无"
                    elif searchobj_other.group("yiqujuliushi") == "有":
                        structuredict["疫区居留史"] = "有"
                    else:
                        structuredict["疫区居留史"] = ""

                if searchobj_other.group("yeyoushi"):
                    if searchobj_other.group("yeyoushi") in ['无', '否认']:
                        structuredict["冶游史"] = "无"
                    elif searchobj_other.group("yeyoushi") == '有':
                        structuredict["冶游史"] = searchobj_other.group("yeyoushi")
                    else:
                        structuredict["冶游史"] = ""

                if searchobj_other.group("yinjiuxiguan"):
                    if searchobj_other.group("yinjiuxiguan") in ['有', '偶尔', '偶', '。', '，']:
                        structuredict["饮酒习惯"] = "有"
                    elif searchobj_other.group("yinjiuxiguan") in ['无', '否认']:
                        structuredict["饮酒习惯"] = "无"
                    else:
                        structuredict["饮酒习惯"] = ""
                
                if searchobj_other.group("xiyanshi"):
                    if searchobj_other.group("xiyanshi") in ['无', '否认']:
                        structuredict["吸烟习惯"] ="无"
                    elif searchobj_other.group("xiyanshi") in ['有', '，', '。', '偶尔', '偶']:
                        structuredict["吸烟习惯"] ="有"
                    else:
                        structuredict["吸烟习惯"] =""

                if searchobj_other.group("duwufangshexingwuzhi"):
                    if searchobj_other.group("duwufangshexingwuzhi") in ['无', '否认']:
                        structuredict["毒性及放射性物质接触史"] = "无"
                    elif searchobj_other.group("duwufangshexingwuzhi") == "有":
                        structuredict["毒性及放射性物质接触史"] = "有"
                    else:
                        structuredict["毒性及放射性物质接触史"] = ""
        else:
            structuredict = {}

        return structuredict

    def previoushisregex(self, text_data):
        """
        对既往史文本进行结构化提取：既往史-值
        param: text_data:文本数据
        output: structuredict

        """
        structuredict = {}
        record = text_data['record'].values[0]
        if record:

            record = re.sub(r"\\r\\n", r"\r\n", record)   #回车和换行，统一替换为\r或\n
            # record = record.strip()    #前后去空格
            record = re.sub(r"\s{0,1}[:|：]\s{0,1}", ":", record)   #将英文与中文冒号，及其带空格的部分，全部替换为英文:
            record = re.sub(r"\s{0,1}[;|；]\s{0,1}", ";", record)   
            record = re.sub(r"\s+", "", record)  #去除空格
            record = re.sub(r"\s{0,1}[“|”]\s{0,1}", "", record )

            #患者既往体质提取
            if re.search(r"(患者)*(既往|过去|平素)(体质|健康状况|身体状况|身体)?(良好|良|一般|较差|体健|尚可|可|好|差|欠佳|偏差)", record):
                searchobj_health = re.search(r"(患者)*(既往|过去|平素)(体质|健康状况|身体状况|身体)?(?P<status>(良好|良|一般|较差|体健|尚可|可|好|差|欠佳|偏差))", record)
                structuredict["既往体质"] = searchobj_health.group("status")
                record = record[:searchobj_health.start()] + record[searchobj_health.end():]
            else:
                structuredict["既往体质"] = ""

            r_lines = [x.strip() for x in re.split(r"[;|。|，|,|\r\n]", record) if x]

            for r in r_lines:
                searchobj = re.search(r"(史|病)", r)            
                if searchobj:
                    searchobj_m = re.search(r"(?P<val>(有|无|否认|没有|没|有过|有诊断|否认有|患))(?P<name>\S+(、)\S+(病史|史))", r)
                    if searchobj_m:
                        items = [i.strip() for i in re.split(r"、", searchobj_m.group("name")) if i]
                        for item in items:
                            structuredict.setdefault(item, searchobj_m.group("val"))
                        r = r[:searchobj_m.start()] + r[searchobj_m.end():]
                        # print(r)
                        if re.search(r"(?P<val>(有|无|否认|有过|有诊断|否认有|患))(?P<name>\S+?(病史|史))", r):
                            searchobj_mm = re.search(r"(?P<val>(有|无|否认|有过|有诊断|否认有|患))(?P<name>\S+?(病史|史))", r)
                            structuredict.setdefault(searchobj_mm.group("name"), searchobj_mm.group("val"))
                    else:
                        searchobj_n = re.search(r"(?P<val>(有|无|否认|有过|有诊断|否认有|患))(?P<name>\S+?(病史|病|史))", r)
                        while searchobj_n:
                            structuredict.setdefault(searchobj_n.group("name"), searchobj_n.group("val"))
                            r = r[searchobj_n.end():]
                            searchobj_n = re.search(r"(?P<val>(有|无|否认|有过|有诊断|否认有|患))(?P<name>\S+?(病史|病|史))", r)
                else:
                    for d in self.disease:
                        regex_rule_d = "(?P<{0}>({1})+)".format("name", d)
                        searchobj_d = re.search(r"{}".format(regex_rule_d), r)
                        if searchobj_d:
                            structuredict.setdefault(searchobj_d.group('name'), "有")
                            r = r[:searchobj_d.start()] + r[searchobj_d.end():]
                        else:
                            continue
        else:
            structuredict = {}

        return structuredict

    def marriagehisregex(self, text_data):
        """
        对“婚育史”文本进行结构化提取，婚育史-值
        param: text_data:文本数据
        output: structuredict
        """
        structuredict = {}

        record = text_data['record'].values[0]
        if record:
            """
            由于与刘医生沟通，“婚育史”对于发热鉴别意义不大，所以暂不处理
            """
        else:
            structuredict = {}

        return structuredict

    def familyhisregex(self, text_data):
        """
        对"家族史"文本进行结构化提取，家族史-值
        param: text_data:文本数据
        output: structuredict
        """
        structuredict = {}

        record = text_data['record'].values[0]
        if record:
            """
            由于与刘医生沟通，“家族史”对于发热鉴别意义不大，所以暂不处理
            """
        else:
            structuredict = {}

        return structuredict

    def firstrecordzsregex(self, text_data):
        """
        对"首次病程记录_ZS"文本进行结构化提取，年龄-值，性别-值
        param: text_data:文本数据
        output: structuredict
        """
        structuredict = {}

        record = text_data['record'].values[0]
        if record:
            record = re.sub(r"\\r\\n", r"\r\n", record)   #回车和换行，统一替换为\r或\n
            record = record.strip()    #前后去空格
            # 冒号统一
            record = re.sub(r"\s{0,1}[:|：]\s{0,1}", ":", record)   #将英文与中文冒号，及其带空格的部分，全部替换为英文:
            record = re.sub(r"\s{0,1}[;|；]\s{0,1}", ";", record)   
            record = re.sub(r"\s+", "", record)  #去除空格

            # 生成正则
            gender_pattern = re.compile(r"(?P<gender>(?<=，)(男|女)(?=(性|，)))")
            searchobj_gender = gender_pattern.search(record)
            if searchobj_gender:
                structuredict["性别"] = searchobj_gender.group("gender")
            else:
                structuredict["性别"] = ""
            searchobj_age = re.search(r"(?P<age>(?<=(男|女|性)(，))(\d){1,3}(?=\s?(岁)?\S*(，因)))", record)
            if searchobj_age:
                structuredict["年龄"] = searchobj_age.group("age")
            else:
                structuredict["年龄"] = ""
        else:
            structuredict = {}

        return structuredict

    def firstrecordzzregex(self, text_data):
        """
        对症状文本进行结构化提取：症状名称-持续时间-频次
        param: text_data:文本数据，dictfile:症状字典文件
        output: structuredict

        """
        structuredict = {}
        record = text_data['record'].values[0]

        if record:
            record = re.sub(r"\\r\\n", r"\r\n", record)   #回车和换行，统一替换为\r或\n
            record = re.sub(r"\s+", "", record)  #去除空格
            record = re.sub(r"\s{0,1}[:|：]\s{0,1}", ":", record)   #将英文与中文冒号，及其带空格的部分，全部替换为英文:
            record = re.sub(r"\s*[""“”]+\s*", "", record)  #去除中英文引号
            record = re.sub(r"(?<!\d)[\.]+(?!\d)", "。", record)  #除了1.5样式表示数字的.之外，所有的.将被更换为。
            record = re.sub(r"(?<=\d)[。]+(?=\d)", ".", record)  #将前后是数字的。更换为.
            record_tmp = re.split(r"[,|，|。|；|\r\n]+", record)

            count = 1
            for i in record_tmp:
                if re.search(r"(没|未见|否认|未诉|无(?!((明显)?诱因|力)))+", i):
                    for j in self.glob_sign:
                        regex_rule_glob = "(?P<{0}>({1})+)".format("glob_sign_name", j)
                        searchobj_glob = re.search(r"{}".format(regex_rule_glob), i)
                        if searchobj_glob:
                            structuredict.setdefault("sign_{}".format(count), {})
                            structuredict["sign_{}".format(count)]["name"] = searchobj_glob.group('glob_sign_name')
                            structuredict["sign_{}".format(count)]["item_type"] = "global"
                            structuredict["sign_{}".format(count)]["value"] = "无"
                            structuredict["sign_{}".format(count)]["location"] = ""
                            structuredict["sign_{}".format(count)]["duration"] = ""
                            structuredict["sign_{}".format(count)]["frequency"] = ""
                            i = i[:searchobj_glob.start()] + i[searchobj_glob.end():]

                            count += 1
                        else:
                            continue

                    # match the locally sign
                    # to store the  matched position in the first loop
                    matched_posi = []
                    # to mark the order of matched local sign 
                    loca_sign_tag = 1
                    for m in self.loca_sign:
                        regex_rule_loca = "(?P<{0}>({1}){{1}})".format("loca_sign_name", m)
                        searchobj_loca = re.search(r"{}".format(regex_rule_loca), i)
                        if searchobj_loca:
                            # match the position dict
                            posi_tag = 1
                            if loca_sign_tag == 1:
                                loca_tag = 1
                                for n in self.posi_sign:
                                    regex_rule_posi = "(?P<{0}>({1})+)".format("posi_name", n)
                                    searchobj_posi = re.search(r"{}".format(regex_rule_posi), i)
                                    if searchobj_posi:
                                        structuredict.setdefault("sign_{}".format(count), {})
                                        structuredict["sign_{}".format(count)]["name"] = searchobj_loca.group("loca_sign_name")
                                        if loca_tag == 1:
                                            i = i[:searchobj_loca.start()] + i[searchobj_loca.end():]
                                        structuredict["sign_{}".format(count)]["location"] = searchobj_posi.group("posi_name")
                                        i = i[:searchobj_posi.start()] + i[searchobj_posi.end():]

                                        structuredict["sign_{}".format(count)]["value"] = "无"
                                        structuredict["sign_{}".format(count)]["duration"] = ""
                                        structuredict["sign_{}".format(count)]["frequency"] = ""
                                        structuredict["sign_{}".format(count)]["item_type"] = "locally"

                                        matched_posi.append(searchobj_posi.group("posi_name"))
                                    
                                        loca_tag += 1
                                        count += 1

                                    elif not searchobj_posi:
                                        posi_tag += 1

                                        if posi_tag == len(self.posi_sign) + 1:
                                            structuredict.setdefault("sign_{}".format(count), {})
                                            structuredict["sign_{}".format(count)]["name"] = searchobj_loca.group("loca_sign_name")
                                            structuredict["sign_{}".format(count)]["location"] = ""
                                            structuredict["sign_{}".format(count)]["value"] = "无"
                                            structuredict["sign_{}".format(count)]["frequency"] = ""
                                            structuredict["sign_{}".format(count)]["duration"] = ""
                                            structuredict["sign_{}".format(count)]["item_type"] = "locally"

                                            count += 1
                                        else:
                                            continue
                                loca_sign_tag += 1
                            else:
                                for k in matched_posi:
                                    structuredict.setdefault("sign_{}".format(count), {})
                                    structuredict["sign_{}".format(count)]["name"] = searchobj_loca.group("loca_sign_name")
                                    structuredict["sign_{}".format(count)]["location"] = k
                                    structuredict["sign_{}".format(count)]["duration"] = ""
                                    structuredict["sign_{}".format(count)]["value"] = "无"
                                    structuredict["sign_{}".format(count)]["frequency"] = ""
                                    structuredict["sign_{}".format(count)]["item_type"] = "locally"

                                    count += 1                  
                        else:
                            continue
       
                elif re.search(r"(体温)?(最高|高峰|峰值|热峰|Tmax)+(体温)?\S*(3|4){1}\d{1}(.)?\d*(℃|摄氏度|度|°)?", i):
                    regex_rule_t = "(体温)?(最高|高峰|峰值|热峰|Tmax)+(体温)?\S*(?P<tem>(3|4){1}\d{1}(.)?\d*)(℃|摄氏度|度|°)?"
                    searchobj_t = re.search(r"{}".format(regex_rule_t), i)
                    if searchobj_t:
                        structuredict.setdefault("sign_{}".format(count), {})
                        structuredict["sign_{}".format(count)]["name"] = "最高体温"
                        structuredict["sign_{}".format(count)]["value"] = searchobj_t.group("tem")
                        structuredict["sign_{}".format(count)]["location"] = ""
                        structuredict["sign_{}".format(count)]["duration"] = ""
                        structuredict["sign_{}".format(count)]["frequency"] = ""
                        structuredict["sign_{}".format(count)]["item_type"] = "global"

                        count += 1
                # 确定性症状提取
                else:
                    # match the global sign     
                    for j in self.glob_sign:
                        regex_rule_glob = "(?P<{0}>({1}){{1}})".format("glob_sign_name", j)
                        searchobj_glob = re.search(r"{}".format(regex_rule_glob), i)
                        if searchobj_glob:
                            # match the fever related sign, duration and frequency 
                            if searchobj_glob.group("glob_sign_name") in ["发热", "低热", "低度热", "高热", "高度热", "中低热", "中低度热", "发烧", "低烧", "高烧", "体温上升", "体温增高", "体温升高", "体温偏高", "体温反复增高", "体温反复升高", "体温反复偏高"]:
                                structuredict.setdefault("sign_{}".format(count), {})
                                structuredict["sign_{}".format(count)]["name"] = searchobj_glob.group("glob_sign_name")
                                i = i[:searchobj_glob.start()] + i[searchobj_glob.end():]

                                searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                if searchobj_time:
                                    structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                else:
                                    structuredict["sign_{}".format(count)]["duration"] = ""

                                searchobj_fre = re.search(r"(反复|间断|间歇|持续|阵发性|偶有|长期)", i)
                                if searchobj_fre:
                                    structuredict["sign_{}".format(count)]["frequency"] = searchobj_fre.group()
                                else:
                                    structuredict["sign_{}".format(count)]["frequency"] = ""
                                
                                structuredict["sign_{}".format(count)]["value"] = "有"
                                structuredict["sign_{}".format(count)]["location"] = ""
                                structuredict["sign_{}".format(count)]["item_type"] = "global"

                                count += 1
                            else:
                                structuredict.setdefault("sign_{}".format(count), {})
                                structuredict["sign_{}".format(count)]["name"] = searchobj_glob.group("glob_sign_name")
                                i = i[:searchobj_glob.start()] + i[searchobj_glob.end():]

                                searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                if searchobj_time:
                                    structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                else:
                                    structuredict["sign_{}".format(count)]["duration"] = ""

                                structuredict["sign_{}".format(count)]["frequency"] = ""
                                structuredict["sign_{}".format(count)]["value"] = "有"
                                structuredict["sign_{}".format(count)]["location"] = ""
                                structuredict["sign_{}".format(count)]["item_type"] = "global"
                                
                                count += 1
                        else:
                            continue

                    # match the locally sign
                    # to store the  matched position in the first loop
                    matched_posi = []
                    # to mark the order of matched local sign 
                    loca_sign_tag = 1
                    for m in self.loca_sign:
                        regex_rule_loca = "(?P<{0}>({1}){{1}})".format("loca_sign_name", m)
                        searchobj_loca = re.search(r"{}".format(regex_rule_loca), i)
                        if searchobj_loca:
                            # match the position dict
                            posi_tag = 1

                            if loca_sign_tag == 1:
                                loca_tag = 1
                                for n in self.posi_sign:

                                    regex_rule_posi = "(?P<{0}>({1})+)".format("posi_name", n)
                                    searchobj_posi = re.search(r"{}".format(regex_rule_posi), i)
                                    if searchobj_posi:
                                    
                                        structuredict.setdefault("sign_{}".format(count), {})
                                        structuredict["sign_{}".format(count)]["name"] = searchobj_loca.group("loca_sign_name")
                                        if loca_tag == 1:
                                            i = i[:searchobj_loca.start()] + i[searchobj_loca.end():]
                                        structuredict["sign_{}".format(count)]["location"] = searchobj_posi.group("posi_name")
                                        i = i[:searchobj_posi.start()] + i[searchobj_posi.end():]

                                        searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                        if searchobj_time:
                                            structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                        else:
                                            structuredict["sign_{}".format(count)]["duration"] = ""

                                        structuredict["sign_{}".format(count)]["frequency"] = ""
                                        structuredict["sign_{}".format(count)]["value"] = "有"
                                        structuredict["sign_{}".format(count)]["item_type"] = "locally"

                                        matched_posi.append(searchobj_posi.group("posi_name"))
                                        
                                        loca_tag += 1
                                        count += 1

                                    elif not searchobj_posi:
                                        posi_tag += 1

                                        if posi_tag == len(self.posi_sign) + 1:
                                            structuredict.setdefault("sign_{}".format(count), {})
                                            structuredict["sign_{}".format(count)]["name"] = searchobj_loca.group("loca_sign_name")
                                            structuredict["sign_{}".format(count)]["value"] = "有"
                                            structuredict["sign_{}".format(count)]["location"] = ""
                                            structuredict["sign_{}".format(count)]["frequency"] = ""

                                            searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                            if searchobj_time:
                                                structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                            else:
                                                structuredict["sign_{}".format(count)]["duration"] = ""

                                            structuredict["sign_{}".format(count)]["item_type"] = "locally"

                                            count += 1
                                        else:
                                            continue
                                loca_sign_tag += 1
                            else:
                                for k in matched_posi:
                                    structuredict.setdefault("sign_{}".format(count), {})
                                    structuredict["sign_{}".format(count)]["name"] = searchobj_loca.group("loca_sign_name")
                                    structuredict["sign_{}".format(count)]["location"] = k

                                    searchobj_time = re.search(r"(\d{1,3}|半|数|两|一|二|三|四|五|六|七|八|九|十)(多|\+|余)?(小时|天|周|年|日|月)", i)
                                    if searchobj_time:
                                        structuredict["sign_{}".format(count)]["duration"] = searchobj_time.group() 
                                    else:
                                        structuredict["sign_{}".format(count)]["duration"] = ""

                                    structuredict["sign_{}".format(count)]["frequency"] = ""
                                    structuredict["sign_{}".format(count)]["value"] = "有"
                                    structuredict["sign_{}".format(count)]["item_type"] = "locally"

                                    count += 1                  
                        else:
                            continue
                    
        else:
            structuredict = {}

        return structuredict

    def firstrecorddiffregex(self, text_data):
        """
        对首次病程记录-鉴别诊断文本进行结构化提取
        param: text_data:文本数据
        output: structuredict
        """
        structuredict = {}
        record = text_data['record'].values[0]

        if record:

            record = re.sub(r"\\r\\n", r"\r\n", record)
            record = record.strip()
            record = re.sub(r"\s{0,1}[:|：]\s{0,1}", ":", record)
            record = re.sub(r"\s*[""“”]+\s*", "", record)
            record = re.sub(r"(?<=\d)[。]+(?=\d)", ".", record)
            record = re.sub(r"\s{0,1}[;|；]\s{0,1}", "；", record)   

            regex_rule = "[\s。；][1-9]{1}[.、,，\s。）]\s?\S+?[\s:，。？]{1}"
            searchobj = re.search(r"{}".format(regex_rule), record)

            if searchobj:
                count = 1
                record_tmp = re.split(r"[。；\s\r\n][1-9]{1}[.、，）\s。]+\s*", record)
                for i in record_tmp:
                    searchobj_d = re.search(r"(警惕:|鉴别诊断:|鉴别:|考虑以下疾病:|考虑如下疾病:|考虑:|考虑部位:|需考虑以下因素:|鉴别以下疾病:|鉴别如下:)", i)
                    if searchobj_d:
                        i = i[searchobj_d.end():]
                        searchobj_name = re.search(r"([1-9]{1}[.、，\s。）]+\s*)?(?P<name>\S+?)(?=[\s:，。？]{1})", i)
                        if searchobj_name:
                            structuredict.setdefault('{}'.format(count), {})
                            if len(searchobj_name.group('name')) > 50:
                                structuredict["{}".format(count)]['name'] = "长度超过50，异常数据"
                                structuredict["{}".format(count)]['reason'] = i[searchobj_name.end()+1:]

                                count += 1
                            else:
                                structuredict["{}".format(count)]['name'] = searchobj_name.group('name')
                                structuredict["{}".format(count)]['reason'] = i[searchobj_name.end()+1:]

                                count += 1
                    else:
                        searchobj_name = re.search(r"([1-9]{1}[.、，\s。）]+\s*)?(?P<name>\S+?)(?=[\s:，。？]{1})", i)
                        if searchobj_name:
                            structuredict.setdefault('{}'.format(count), {})
                            if len(searchobj_name.group('name')) > 50:
                                structuredict["{}".format(count)]['name'] = "长度超过50，异常数据"
                                structuredict["{}".format(count)]['reason'] = i[searchobj_name.end()+1:]
                                count += 1
                            else:
                                structuredict["{}".format(count)]['name'] = searchobj_name.group('name')
                                structuredict["{}".format(count)]['reason'] = i[searchobj_name.end()+1:]

                                count += 1
            else:
                structuredict = {}   
        else:
            structuredict = {}

        return structuredict

    def chiefcom2df(self, infile, structuredict):
        """
        INPUT: infile(数据库内获取到的患者文本数据), structuredict(主诉文本结构化处理之后的返回)
        OUTPUT: news(重新组合成DataFrame的结构化后的数据)
        """
        count = 1
        news = pd.DataFrame()
        for i, j in structuredict.items():
            #生成新的df
            new = pd.DataFrame({'note_id': infile['note_id'].values[0],
                        'person_id': infile['person_id'].values[0],
                        'visit_record_id': infile['visit_record_id'].values[0],
                        'visit_record_id_new': infile['visit_record_id_new'].values[0],
                        'text_source': infile['subtype_name'].values[0],
                        'sign_type': j['sign_type'],
                        'sign_index': i,
                        'sign_name': j['symptom'],
                        'duration': j['duration'],
                        'location': j['location'],
                        'frequency': j['frequency'],
                        'time':infile['time'].values[0],
                        'provider': infile['provider'].values[0]}, index = [0])
            count += 1
            news = news.append(new, ignore_index = True)

        return news

    def dict2df(self, infile, structuredict):
        """
        INPUT: infile(数据库内获取到的患者文本数据), structuredict(除主诉之外的文本结构化处理之后的返回)
        OUTPUT: news(重新组合成DataFrame的结构化后的数据)
        """
        count = 1
        news = pd.DataFrame()
        for i, j in structuredict.items():
            #生成新的df
            new = pd.DataFrame({'note_id': infile['note_id'].values[0],
                        'person_id': infile['person_id'].values[0],
                        'visit_record_id': infile['visit_record_id'].values[0],
                        'visit_record_id_new': infile['visit_record_id_new'].values[0],
                        'text_source': infile['subtype_name'].values[0],
                        'name': i,
                        'value': j,
                        'time':infile['time'].values[0],
                        'provider': infile['provider'].values[0]}, index = [0])
            count += 1
            news = news.append(new, ignore_index = True)

        return news

    def firstrecord2df(self, infile, structuredict):
        """
        INPUT: infile(数据库内获取到的患者文本数据), structuredict("首次病程记录-症状"文本结构化处理之后的返回)
        OUTPUT: news(重新组合成DataFrame的结构化后的数据)
        """
        count = 1
        news = pd.DataFrame()
        for i, j in structuredict.items():
            #生成新的df
            new = pd.DataFrame({'note_id': infile['note_id'].values[0],
                        'person_id': infile['person_id'].values[0],
                        'visit_record_id': infile['visit_record_id'].values[0],
                        'visit_record_id_new': infile['visit_record_id_new'].values[0],
                        'text_source': infile['subtype_name'].values[0],
                        'item_type': j['item_type'],
                        'item_index': i,
                        'name': j['name'],
                        'location': j['location'],
                        'value': j['value'],
                        'duration': j['duration'],
                        'frequency': j['frequency'],
                        'time':infile['time'].values[0],
                        'provider': infile['provider'].values[0]}, index = [0])
            count += 1
            news = news.append(new, ignore_index = True)

        return news

    def firstrecorddiff2df(self, infile, structuredict):
        """
        INPUT: infile(数据库内获取到的患者文本数据), structuredict("首次病程记录-鉴别诊断"文本结构化处理之后的返回)
        OUTPUT: news(重新组合成DataFrame的结构化后的数据)
        """
        count = 1
        news = pd.DataFrame()
        for i, j in structuredict.items():
            #生成新的df
            new = pd.DataFrame({'note_id': infile['note_id'].values[0],
                        'person_id': infile['person_id'].values[0],
                        'visit_record_id': infile['visit_record_id'].values[0],
                        'visit_record_id_new': infile['visit_record_id_new'].values[0],
                        'text_source': infile['subtype_name'].values[0],
                        'item_index': i,
                        'name': j['name'],
                        'reason': j['reason'],
                        'time':infile['time'].values[0],
                        'provider': infile['provider'].values[0]}, index = [0])
            count += 1
            news = news.append(new, ignore_index = True)

        return news

class TextGeneralization():

    def __init__(self, db_name, in_table_name, in_table_name_zz):
        self.db_conn = db_connection(db_name)
        self.in_table_name = in_table_name
        self.in_table = Table(in_table_name, self.db_conn.metadata, autoload=True)
        self.in_table_name_zz = in_table_name_zz
        self.in_table_zz = Table(in_table_name_zz, self.db_conn.metadata, autoload=True)

        # read the dict file
        self.rootDir = os.path.split(os.path.realpath(__file__))[0]
        self.dictpath = os.path.join(self.rootDir, 'config/symptom_map.txt')
    
    def read_dict(self):
        
        dict_glob, dict_loca_sign, dict_loca_posi= {}, {}, {}
        with codecs.open(self.dictpath, "r", encoding = "utf-8") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue

                elif line.startswith('全身症状'):
                    items = line[5:].split(' ||| ')
                    assert len(items) == 2
                    key = items[0]
                    value = items[1].split()
                    value.sort(key= lambda i: len(i), reverse=True)
                    dict_glob[key] = value
            
            dict_glob = sorted(dict_glob.items(), key=lambda item: len(item[0]), reverse=True)
            dict_glob = dict(dict_glob)

        with codecs.open(self.dictpath, "r", encoding = "utf-8") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue

                elif line.startswith('局部症状'):
                    items = line[5:].split(' ||| ')
                    assert len(items) == 2
                    key = items[0]
                    value = items[1].split()
                    value.sort(key= lambda i: len(i), reverse=True)
                    dict_loca_sign[key] = value
            
            dict_loca_sign = sorted(dict_loca_sign.items(), key=lambda item: len(item[0]), reverse=True)
            dict_loca_sign = dict(dict_loca_sign)
        
        with codecs.open(self.dictpath, "r", encoding = "utf-8") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue

                elif line.startswith('身体部位'):
                    items = line[5:].split(' ||| ')
                    assert len(items) == 2
                    key = items[0]
                    value = items[1].split()
                    value.sort(key= lambda i: len(i), reverse=True)
                    dict_loca_posi[key] = value
            
            dict_loca_posi = sorted(dict_loca_posi.items(), key=lambda item: len(item[0]), reverse=True)
            dict_loca_posi = dict(dict_loca_posi)

        return dict_glob, dict_loca_sign, dict_loca_posi

    def read_table(self):

        s = select([self.in_table.c.sign_id,
                    self.in_table.c.sign_type,
                    self.in_table.c.sign_index,
                    self.in_table.c.location,
                    self.in_table.c.sign_name,
                    self.in_table.c.duration,
                    self.in_table.c.frequency,
                    self.in_table.c.sign_name_general
                    ])
        result = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(result, columns=['sign_id','sign_type','sign_index','location','sign_name','duration','frequency','sign_name_general'])

        return result

    def generalization_zs(self):

        data = self.read_table()
        dict_glob, dict_loca_sign, dict_loca_posi = self.read_dict()

        data_glob = data.loc[data['sign_type']=='global',]
        data_loca = data.loc[data['sign_type']=='locally',]

        print("-----------------------------------------------------------------------")
        print("The shape of data_glob is {0}".format(data_glob.shape))
        print("-----------------------------------------------------------------------")

        # generalize the global sign name
        count = 0
        for index, row in data_glob.iterrows():         
            print("-----------------------------------------------------------------------")
            print("The number of data_glob having been processed is {0}".format(count))
            print("-----------------------------------------------------------------------")
            flag =True
            count += 1
            for key, values in dict_glob.items():
                for value in values:
                    if row['sign_name'] == value:
                        sql = self.in_table.update().\
                            where(self.in_table.c.sign_id == row['sign_id']).\
                                values(sign_name_general=key)
                        self.db_conn.conn.execute(sql)
                        flag = False
                        break
                    else:
                        continue
                if not flag:
                    break
            if flag:
                sql = self.in_table.update().\
                        where(self.in_table.c.sign_id == row['sign_id']).\
                            values(sign_name_general=row['sign_name'])
                self.db_conn.conn.execute(sql)

        print("-----------------------------------------------------------------------")
        print("The update of data_glob have finished")
        print("-----------------------------------------------------------------------")

        # # generalize the locally sign name
        count = 0
        for index, row in data_loca.iterrows():

            print("-----------------------------------------------------------------------")
            print("The number of data_loca having been processed is {0}".format(count))
            print("-----------------------------------------------------------------------")
            flag =True
            count += 1
            for key, values in dict_loca_sign.items():
                for value in values:
                    if row['sign_name'] == value:
                        sql = self.in_table.update().\
                            where(self.in_table.c.sign_id == row['sign_id']).\
                                values(sign_name_general=key)
                        self.db_conn.conn.execute(sql)
                        flag = False
                        break
                    else:
                        continue
                if not flag:
                    break
            if flag:
                sql = self.in_table.update().\
                        where(self.in_table.c.sign_id == row['sign_id']).\
                            values(sign_name_general=row['sign_name'])
                self.db_conn.conn.execute(sql)

        print("-----------------------------------------------------------------------")
        print("The update of data_loca have finished")
        print("-----------------------------------------------------------------------")

        # # generalize the locally sign name
        count = 0
        for index, row in data_loca.iterrows():

            print("-----------------------------------------------------------------------")
            print("The number of data_loca having been processed is {0}".format(count))
            print("-----------------------------------------------------------------------")
            flag =True
            count += 1
            for key, values in dict_loca_posi.items():
                for value in values:
                    if row['location'] == value:
                        sql = self.in_table.update().\
                            where(self.in_table.c.sign_id == row['sign_id']).\
                                values(location_general=key)
                        self.db_conn.conn.execute(sql)
                        flag = False
                        break
                    else:
                        continue
                if not flag:
                    break
            if flag:
                sql = self.in_table.update().\
                        where(self.in_table.c.sign_id == row['sign_id']).\
                            values(location_general=row['location'])
                self.db_conn.conn.execute(sql)

        print("-----------------------------------------------------------------------")
        print("The update of data_loca have finished")
        print("-----------------------------------------------------------------------")

    def generalization_zz(self):

        s = select([self.in_table_zz.c.sign_id,
                    self.in_table_zz.c.item_type,
                    self.in_table_zz.c.item_index,
                    self.in_table_zz.c.location,
                    self.in_table_zz.c.name,
                    self.in_table_zz.c.value,
                    self.in_table_zz.c.duration,
                    self.in_table_zz.c.frequency,
                    self.in_table_zz.c.sign_name_general,
                    self.in_table_zz.c.location_general
                    ])
        result = self.db_conn.conn.execute(s).fetchall()
        data = pd.DataFrame(result, columns=['sign_id','item_type','item_index','location','name', 'value', 'duration','frequency','sign_name_general', 'location_general'])
        dict_glob, dict_loca_sign, dict_loca_posi = self.read_dict()

        data_glob = data.loc[data['item_type']=='global',]
        data_loca = data.loc[data['item_type']=='locally',]

        print("-----------------------------------------------------------------------")
        print("The shape of data_glob is {0}".format(data_glob.shape))
        print("-----------------------------------------------------------------------")

        # generalize the global sign name
        count = 0
        for index, row in data_glob.iterrows():         
            print("-----------------------------------------------------------------------")
            print("The number of data_glob having been processed is {0}".format(count))
            print("-----------------------------------------------------------------------")
            flag =True
            count += 1
            for key, values in dict_glob.items():
                for value in values:
                    if row['name'] == value:
                        sql = self.in_table_zz.update().\
                            where(self.in_table_zz.c.sign_id == row['sign_id']).\
                                values(sign_name_general=key)
                        self.db_conn.conn.execute(sql)
                        flag = False
                        break
                    else:
                        continue
                if not flag:
                    break
            if flag:
                sql = self.in_table_zz.update().\
                        where(self.in_table_zz.c.sign_id == row['sign_id']).\
                            values(sign_name_general=row['name'])
                self.db_conn.conn.execute(sql)

        print("-----------------------------------------------------------------------")
        print("The update of data_glob have finished")
        print("-----------------------------------------------------------------------")

        # # generalize the locally sign name
        count = 0
        for index, row in data_loca.iterrows():

            print("-----------------------------------------------------------------------")
            print("The number of data_loca having been processed is {0}".format(count))
            print("-----------------------------------------------------------------------")
            flag =True
            count += 1
            for key, values in dict_loca_sign.items():
                for value in values:
                    if row['name'] == value:
                        sql = self.in_table_zz.update().\
                            where(self.in_table_zz.c.sign_id == row['sign_id']).\
                                values(sign_name_general=key)
                        self.db_conn.conn.execute(sql)
                        flag = False
                        break
                    else:
                        continue
                if not flag:
                    break
            if flag:
                sql = self.in_table_zz.update().\
                        where(self.in_table_zz.c.sign_id == row['sign_id']).\
                            values(sign_name_general=row['name'])
                self.db_conn.conn.execute(sql)

        print("-----------------------------------------------------------------------")
        print("The update of data_loca have finished")
        print("-----------------------------------------------------------------------")

        # # generalize the locally sign name
        count = 0
        for index, row in data_loca.iterrows():

            print("-----------------------------------------------------------------------")
            print("The number of data_loca having been processed is {0}".format(count))
            print("-----------------------------------------------------------------------")
            flag =True
            count += 1
            for key, values in dict_loca_posi.items():
                for value in values:
                    if row['location'] == value:
                        sql = self.in_table_zz.update().\
                            where(self.in_table_zz.c.sign_id == row['sign_id']).\
                                values(location_general=key)
                        self.db_conn.conn.execute(sql)
                        flag = False
                        break
                    else:
                        continue
                if not flag:
                    break
            if flag:
                sql = self.in_table_zz.update().\
                        where(self.in_table_zz.c.sign_id == row['sign_id']).\
                            values(location_general=row['location'])
                self.db_conn.conn.execute(sql)

        print("-----------------------------------------------------------------------")
        print("The update of data_loca have finished")
        print("-----------------------------------------------------------------------")