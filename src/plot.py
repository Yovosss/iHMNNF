# -*- coding:utf-8 -*-

"""
@author: wang zhixiao
@time  : 2021-03-12 21:36
@detail: 用于对数据进行画图表征
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from sqlalchemy import Table, select, text, distinct, types, func, extract

from config.orcl import db_connection
plt.rc("font",family="Times New Roman",size="12") 
# plt.rc("font",family="Arial",size="12") 
plt.rcParams["font.sans-serif"]=["SimHei"]  #显示中文标签
plt.rcParams['axes.unicode_minus']=False



class DataAggregation():

    def __init__(self,db_name, table):

        self.db_conn = db_connection(db_name)
        self.table = Table(table, self.db_conn.metadata, autoload=True)


    def data_ex_gender(self):
        s = select([self.table.c.gender,
                      func.count(distinct(self.table.c.person_id)).label('count')]).\
                      group_by(self.table.c.gender)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['gender', 'count'])
        return result

    def data_ex_age(self):
        s = select([self.table.c.visit_record_id,
                     self.table.c.visit_age])
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['visit', 'age'])
        result = result.loc[result['age'] >= 0]
        result = result.sort_values(by='age', ascending=True)
        result.reset_index(drop=True,inplace=True)
        return result

    def data_ex_year(self):
        s = select([extract('year',self.table.c.visit_start_date).label('year'),
                     func.count(extract('year',self.table.c.visit_start_date)).label('count')]).\
                         group_by(extract('year',self.table.c.visit_start_date))
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['year', 'count'])
        result = result.sort_values(by='year', ascending=True)
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_month(self):
        s = select([extract('month',self.table.c.visit_start_date).label('month'),
                     func.count(extract('month',self.table.c.visit_start_date)).label('count')]).\
                         group_by(extract('month',self.table.c.visit_start_date))
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['month', 'count'])
        result = result.sort_values(by='month', ascending=True)
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_los(self):
        s = select([self.table.c.los,
                     func.count(self.table.c.visit_record_id).label('count')]).\
                         group_by(self.table.c.los)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['los', 'count'])
        result = result.sort_values(by='los', ascending=True)
        # result = result.iloc[1:56]
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_diag(self, diag_type):

        if diag_type == '初步诊断':
            s = select([self.table.c.diag_name,
                func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                    where(self.table.c.diag_type == '初步诊断').\
                        group_by(self.table.c.diag_name)
            res = self.db_conn.conn.execute(s).fetchall()
            result = pd.DataFrame(res, columns=['diagname', 'count'])
            result = result.sort_values(by='count', ascending=False)
            result = result.loc[result['count'] > 300]
            result.reset_index(drop=True,inplace=True)

        elif diag_type == '初步诊断主要诊断':
            s = select([self.table.c.diag_name,
                     func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                         where(self.table.c.diag_type == '初步诊断').\
                             where(self.table.c.diag_index == '1').\
                             group_by(self.table.c.diag_name)
            res = self.db_conn.conn.execute(s).fetchall()
            result = pd.DataFrame(res, columns=['diagname', 'count'])
            result = result.sort_values(by='count', ascending=False)
            result = result.loc[result['count'] > 50]
            result.reset_index(drop=True,inplace=True)

        elif diag_type == '出院诊断':
            s = select([self.table.c.diag_name,
                     func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                         where(self.table.c.diag_type == '出院诊断').\
                             group_by(self.table.c.diag_name)
            res = self.db_conn.conn.execute(s).fetchall()
            result = pd.DataFrame(res, columns=['diagname', 'count'])
            result = result.sort_values(by='count', ascending=False)
            result = result.loc[result['count'] > 600]
            result.reset_index(drop=True,inplace=True)

        elif diag_type == '出院诊断主要诊断':
            s = select([self.table.c.diag_name,
                     func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                         where(self.table.c.diag_type == '出院诊断').\
                             where(self.table.c.diag_index == '1').\
                             group_by(self.table.c.diag_name)
            res = self.db_conn.conn.execute(s).fetchall()
            result = pd.DataFrame(res, columns=['diagname', 'count'])
            result = result.sort_values(by='count', ascending=False)
            result = result.loc[result['count'] > 100]
            result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_signs(self):
        s = select([self.table.c.type,
                     func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                             group_by(self.table.c.type)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['signs', 'count'])
        result = result.sort_values(by='count', ascending=False)
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_meditech(self):
        s = select([self.table.c.tech_name_eng,
                     func.count(distinct(self.table.c.visit_record_id_new)).label('count')]).\
                             group_by(self.table.c.tech_name_eng)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['meditech', 'count'])
        result = result.sort_values(by='count', ascending=False)
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_mea(self):
        s = select([self.table.c.group_measurement_name,
                     func.count(distinct(self.table.c.visit_record_id_new)).label('count')]).\
                             group_by(self.table.c.group_measurement_name)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['mea', 'count'])
        result = result.sort_values(by='count', ascending=False)
        result = result[result['count'] > 5000]
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_note(self):
        s = select([self.table.c.subtype_name,
                     func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                             group_by(self.table.c.subtype_name)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['note', 'count'])
        result = result.sort_values(by='count', ascending=False)
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_signs_av(self):
        s = select([self.table.c.visit_record_id,
                    self.table.c.type,
                    func.count(self.table.c.value)]).\
                        group_by(self.table.c.visit_record_id, self.table.c.type)

        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['visit_record_id', 'type', 'count'])
        result_pivot = result.pivot(index = 'visit_record_id', columns = 'type', values = 'count')

        return result_pivot

    def data_ex_meditech_av(self):
        s = select([self.table.c.visit_record_id_new,
                    self.table.c.tech_name_eng,
                    func.count(self.table.c.technology_id)]).\
                        group_by(self.table.c.visit_record_id_new, self.table.c.tech_name_eng)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['visit_record_id_new', 'type', 'count'])
        result_pivot = result.pivot(index = 'visit_record_id_new', columns = 'type', values = 'count')

        return result_pivot

    def data_ex_mea_av_group(self):
        s = select([self.table.c.visit_record_id_new,
                    self.table.c.group_measurement_name,
                    func.count(distinct(self.table.c.lab_no)).label('count')]).\
                        group_by(self.table.c.visit_record_id_new, self.table.c.group_measurement_name)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['visit_record_id_new', 'type', 'count'])
        df = result.groupby('type').count()
        df1 = df[df['visit_record_id_new'] > 5000]
        df2 = result[result['type'].isin(df1.index.values)]
        result_pivot = df2.pivot(index = 'visit_record_id_new', columns = 'type', values = 'count')
        # print(result_pivot.head())
        return result_pivot

    def data_ex_mea_av_item(self):
        s = select([self.table.c.visit_record_id_new,
                    self.table.c.item_name,
                    func.count(distinct(self.table.c.lab_no)).label('count')]).\
                        group_by(self.table.c.visit_record_id_new, self.table.c.item_name)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['visit_record_id_new', 'item', 'count'])
        df = result.groupby('item').count()
        df1 = df[df['visit_record_id_new'] > 20000]
        df2 = result[result['item'].isin(df1.index.values)]
        result_pivot = df2.pivot(index = 'visit_record_id_new', columns = 'item', values = 'count')
        # print(result_pivot.head())
        return result_pivot

    def data_ex_zs(self):
        s = select([self.table.c.sign_name,
                    func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                        group_by(self.table.c.sign_name)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['chiefcomplaint', 'count'])
        result = result.sort_values(by='count', ascending = False)
        result = result[result['count'] > 100]
        result.reset_index(drop = True, inplace = True)

        return result

    def data_ex_fever_dur(self):
        s = select([self.table.c.visit_record_id,
                     self.table.c.duration_value]).where(self.table.c.sign_name=='发热').\
                         where(self.table.c.duration_value != None)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns = ['visit_record_id', 'duration_value'])
        # result = result[result['duration_value'] < 357]
        result = result[result['duration_value'] < 31]
        result = result.sort_values(by='duration_value', ascending=True)
        result.reset_index(drop=True,inplace=True)

        return result

    def data_ex_zz(self):
        s = select([self.table.c.name,
                    func.count(distinct(self.table.c.visit_record_id)).label('count')]).\
                        where(self.table.c.value == '有').\
                            group_by(self.table.c.name)
        res = self.db_conn.conn.execute(s).fetchall()
        result = pd.DataFrame(res, columns=['symptom', 'count'])
        result = result.sort_values(by='count', ascending = False)
        result = result[result['count'] > 100]
        result.reset_index(drop = True, inplace = True)

        return result







class FigurePlot():

    def __init__(self, df):

        self.input = df

    def plot_pie(self, x, y, titlename):
        """
        input: x(第一行列名，一般是要表征的类别名称)
               y(各个类别对应的数量)
               titlename(图表名称)
        output: 本地存储的饼图
        """
        fig, ax = plt.subplots(figsize=(7,5), subplot_kw=dict(aspect="equal"))

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%\n({:d})".format(pct, absolute)

        explode = (0.05,0)
        wedges, texts, autotexts = ax.pie(self.input[y], explode=explode, \
            autopct=lambda pct:func(pct,self.input[y]), textprops={'fontsize':30, 'color':'w'})
        legend = ax.legend(wedges, self.input[x], title=x.title(), ncol=1, \
            loc="center left", bbox_to_anchor=(1,0,0.5,1), fontsize=15)
        legend.get_title().set_fontproperties('Times New Roman')
        legend.get_title().set_fontsize(fontsize=18)
        plt.setp(autotexts, fontsize=1, fontproperties='Times New Roman')
        plt.setp(autotexts, fontsize=22)
        ax.set_title(titlename.title(), fontproperties='Times New Roman', fontsize=20, weight="bold")
        # plt.show()
        fig = ax.get_figure()
        fig.savefig(os.path.dirname(__file__) + "/output/Figure_{}".format(titlename.title()), dpi=600, bbox_inches="tight")

    def plot_hist(self, x, y, titlename):
        """
        input: x(横坐标名称)
               y(纵坐标名称)
               titlename(图表名称)
        output: 本地存储的histplot
        """
        plt.figure(figsize=(15,8))
        g = sns.histplot(x=x, data=self.input, binwidth=2, kde=True, line_kws={'color':'#008000', "lw": 2})   #不知道为什么此处颜色改变不了
        g.set_title(titlename.title(), fontsize=25, position=(0.5, 1.05))
        g.set_ylabel(y.title(), fontsize=20)
        g.set_xlabel(x.title(), fontsize=20)
        g.tick_params(labelsize=18)  #设置坐标轴刻度的字体大小
        # plt.show()
        fig = g.get_figure()
        fig.savefig(os.path.dirname(__file__) + "/output/Figure_{}".format(titlename.title()), dpi=400, bbox_inches="tight")

    def plot_histwithbox(self, x, y, titlename):

        fig = plt.figure(figsize=(16, 10), dpi= 600)
        grid = plt.GridSpec(5, 6, hspace=1, wspace=0.2)

        # Define the axes
        ax_main = fig.add_subplot(grid[:-1, :])
        ax_bottom = fig.add_subplot(grid[-1, :], xticklabels=[], yticklabels=[])

        sns.histplot(ax=ax_main, data=self.input, binwidth=2, kde=True, line_kws={'color':'#008000', "lw": 2})

        sns.boxplot(self.input.duration_value, ax=ax_bottom, orient="h")    #age

        # Remove x axis name for the boxplot
        ax_bottom.set(xlabel='')

        ax_main.set(title=titlename.title(), xlabel=x.title(), ylabel=y.title())
        ax_main.tick_params(labelsize=18)
        ax_main.title.set_fontsize(25)
        for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
            item.set_fontsize(20)

        fig.savefig(os.path.dirname(__file__) + "/output/Figure_{}_Withbox".format(titlename.title()), dpi=600, bbox_inches="tight")


    def plot_bar(self, x, y, titlename):

        # plt.figure(figsize=(16,10))
        plt.figure(figsize=(20,8))
        g = sns.barplot(x=x, y=y, data=self.input, palette="Blues_d")
        g.set_title(titlename.title(), fontsize=25, position=(0.5, 1.05))
        g.set_ylabel(y.title(), fontsize=20) 
        g.set_xlabel(x.title(), fontsize=20)
        g.tick_params(labelsize=16)  #设置坐标轴刻度的字体大小
        g.set_xticklabels(g.get_xticklabels(), rotation= 80, fontfamily='sans-serif')  #fontdict={'family':'sans-serif'}
        for index, row in self.input.iterrows():
            g.text(row.name, row[y], row[y], color="black",fontsize=11, ha="center", va='bottom')
        # plt.show()
        fig = g.get_figure()
        fig.savefig(os.path.dirname(__file__) + "/output/{}".format("Figure_{}".format(titlename.title())), dpi=600, bbox_inches='tight')

    def plot_barwithbox(self, x, y, titlename):

        dy=list(self.input[y])
        dx=list(self.input[x])
        boxdata=list()
        i=0
        while i!=len(dx):
            j=0
            while j!=dy[i]:
                boxdata.append(dx[i])
                j=j+1
            i=i+1

        fig = plt.figure(figsize=(20, 10), dpi= 600)
        grid = plt.GridSpec(5, 6, hspace=1, wspace=0.2)

        # Define the axes
        ax_main = fig.add_subplot(grid[:-1, :])
        ax_bottom = fig.add_subplot(grid[-1, :], xticklabels=[], yticklabels=[])

        sns.barplot(x=x, y=y, ax=ax_main, data=self.input, palette="Blues_d")

        sns.boxplot(boxdata, ax=ax_bottom, orient="h")

        # Remove x axis name for the boxplot
        ax_bottom.set(xlabel='')

        ax_main.set(title=titlename.title(), xlabel=x.title(), ylabel=y.title())
        ax_main.tick_params(labelsize=12)
        ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation= 80, fontfamily='Times New Roman')
        ax_main.title.set_fontsize(25)
        for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
            item.set_fontsize(12)

        fig.savefig(os.path.dirname(__file__) + "/output/Figure_{}_Withbox".format(titlename.title()), dpi=600, bbox_inches="tight")   
    
    def plot_bar_spines(self, x, y, titlename):

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=100, figsize=(20,5))
        # fig.subplots_adjust(hspace=0)

        ax1.bar(x=self.input[x], height=self.input[y])
        ax2.bar(x=self.input[x], height=self.input[y])

        if titlename.title() == 'Initial Diagnosis':
            ax1.set_ylim(16900, 17500)
            ax2.set_ylim(0, 6000)
        elif titlename.title() == 'Main Initial Diagnosis':
            ax1.set_ylim(12000, 16000)
            ax2.set_ylim(0,2000)
        elif titlename.title() == 'Chief Complaint':
            ax1.set_ylim(26000, 27600)
            ax2.set_ylim(0,5000)
        elif titlename.title() == 'Symptom':
            ax1.set_ylim(18500, 19200)
            ax2.set_ylim(0,8500)

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)
        ax2.xaxis.tick_bottom()
        d = .85  #设置倾斜度
        #绘制断裂处的标记
        kwargs = dict(marker=[(-1,-d),(1,d)], markersize=15,\
                    linestyle='none', color='r', mec='r', mew=1, clip_on=False)
        ax1.plot([0,1], [0,0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0,1], [1,1], transform=ax2.transAxes, **kwargs)

        ax2.tick_params(axis='x', rotation=80)
        labels = ax2.get_xticklabels()
        [label.set_fontname('sans-serif') for label in labels]
        ax1.set_title('The Distribution of {}'.format(titlename), fontsize=18, position=(0.5, 1.05))
        # ax2.set_ylabel('Visit Count', fontsize=15)
        # ax2.set_xlabel('Registration Number', fontsize=15)
        # ax2.set_xticklabels(ax2.get_xticklabels(), fontfamily='sans-serif')  #fontdict={'family':'sans-serif'}
        for index, row in self.input[1:].iterrows():
            ax2.text(row.name, row[y], row[y], color="black",fontsize=10, ha="center", va='bottom')
        plt.tight_layout()
        # plt.show()
        fig = ax2.get_figure()
        fig.savefig(os.path.dirname(__file__) + "/output/{}".format("Figure_The Distribution Of {}".format(titlename.title())), dpi=600, bbox_inches='tight')

    def plot_bar_nospines(self, x, y, titlename):

        fig, (ax2) = plt.subplots(1, 1, sharex=True, dpi=100, figsize=(20,5))
        ax2.bar(x=self.input[x], height=self.input[y])
        ax2.xaxis.tick_bottom()

        ax2.tick_params(axis='x', rotation=90)
        labels = ax2.get_xticklabels()
        [label.set_fontname('sans-serif') for label in labels]
        ax2.set_title('The Distribution Of  {}'.format(titlename.title()), fontsize=18, position=(0.5, 1.05))
        for index, row in self.input.iterrows():
            ax2.text(row.name, row[y], row[y], color="black",fontsize=10, ha="center", va='bottom')
        # plt.tight_layout()
        # plt.show()
        fig = ax2.get_figure()
        # fig.savefig(os.path.dirname(__file__) + "/output/{}".format("Figure_The Dsitribution Of {}".format(titlename.title())), dpi=400, bbox_inches='tight')
        fig.savefig("./{}".format("Figure_The Dsitribution Of {}".format(titlename.title())), dpi=400, bbox_inches='tight')

    def plot_violin(self, x, y, titlename):
        plt.figure(figsize = (20, 10))

        g = sns.violinplot(data = self.input, palette="Set3")
        g.set_title(titlename.title(), fontsize=25, position=(0.5, 1.05))
        g.set_ylabel(y.title(), fontsize=20) 
        g.set_xlabel(x.title(), fontsize=20)
        g.tick_params(labelsize=16)  #设置坐标轴刻度的字体大小
        g.set_xticklabels(g.get_xticklabels(), rotation= 80, fontfamily='sans-serif')

        fig = g.get_figure()
        fig.savefig(os.path.dirname(__file__) + "/output/{}".format("Figure_{}".format(titlename.title())), dpi=600, bbox_inches='tight')

    def plot_boxplot(self, x, y, titlename):
        plt.figure(figsize = (50, 10))   #(30, 10) for mea group

        g = sns.boxplot(data = self.input, palette="Set3")
        g.set_title(titlename.title(), fontsize=25, position=(0.5, 1.05))
        g.set_ylabel(y.title(), fontsize=20) 
        g.set_xlabel(x.title(), fontsize=20)
        g.tick_params(labelsize=16)  #设置坐标轴刻度的字体大小
        g.set_xticklabels(g.get_xticklabels(), rotation= 80, fontfamily='sans-serif')   #sans-serif

        fig = g.get_figure()
        fig.savefig(os.path.dirname(__file__) + "/output/{}".format("Figure_{}".format(titlename.title())), dpi=600, bbox_inches='tight')

def plot():

    ## 饼图-性别
    # piedata = DataAggregation('fuo', 'PERSON')
    # out = piedata.data_ex_gender()
    # pieplot = FigurePlot(out)
    # pieplot.plot_pie("gender", "count", "gender distribution")

    # 柱状图-就诊年龄
    # histdata = DataAggregation('fuo', 'VISIT_ZY')
    # out = histdata.data_ex_age()
    # hisplot = FigurePlot(out)
    # hisplot.plot_hist('age', 'count', "the distribution of visit age")

    # 柱状图+箱型图-就诊年龄
    # histdata = DataAggregation('fuo', 'VISIT_ZY')
    # out = histdata.data_ex_age()
    # hisplot = FigurePlot(out)
    # hisplot.plot_histwithbox('age', 'count', "the distribution of visit age")

    ## 柱状图-就诊年份分布
    # bardata = DataAggregation('fuo', 'VISIT_ZY')
    # out = bardata.data_ex_year()
    # barplot = FigurePlot(out)
    # barplot.plot_bar('year', 'count', 'the distribution of visit year')

    ## 柱状图-就诊月份分布
    # bardata = DataAggregation('fuo', 'VISIT_ZY')
    # out = bardata.data_ex_month()
    # barplot = FigurePlot(out)
    # barplot.plot_bar('month', 'count', 'the distribution of visit month')

    ## 柱状图-LOS分布
    # bardata = DataAggregation('fuo', 'VISIT_ZY')
    # out = bardata.data_ex_los()
    # barplot = FigurePlot(out)
    # barplot.plot_bar('los', 'count', 'the distribution of LOS')

    ## 柱状图+箱型图-LOS分布
    # bardata = DataAggregation('fuo', 'VISIT_ZY')
    # out = bardata.data_ex_los()
    # barplot = FigurePlot(out)
    # barplot.plot_barwithbox('los', 'count', 'the distribution of LOS')

    ## 柱状图-初步诊断/初步诊断主要诊断分布
    # bardata = DataAggregation('fuo', 'ORIGIN_CONDITION_ZY')
    # out = bardata.data_ex_diag('初步诊断主要诊断') ##初步诊断/初步诊断主要诊断
    # barplot = FigurePlot(out)
    # barplot.plot_bar_spines('diagname', 'count', 'Main Initial Diagnosis')

    ## 柱状图-出院诊断/出院诊断主要诊断分布
    bardata = DataAggregation('fuo', 'ORIGIN_CONDITION_ZY')
    out = bardata.data_ex_diag('出院诊断主要诊断') ##初步诊断/初步诊断主要诊断
    barplot = FigurePlot(out)
    barplot.plot_bar_nospines('diagname', 'count', 'Main Discharge Diagnosis')

    ## 柱状图-护理生命体征分布
    # bardata = DataAggregation('fuo', 'ORIGIN_SIGNS')
    # out = bardata.data_ex_signs()
    # barplot = FigurePlot(out)
    # barplot.plot_bar('signs', 'count', 'the distribution of signs')

    ## 箱型图-护理生命体征分布--人均数据量分布
    # bardata = DataAggregation('fuo', 'SIGNS')
    # out = bardata.data_ex_signs_av()
    # boxplot = FigurePlot(out)
    # boxplot.plot_boxplot("type", "count", 'Per capita data distribution of vital signs')

    ## 柱状图-影像检查分布
    # bardata = DataAggregation('fuo', 'ORIGIN_MEDITECH')
    # out = bardata.data_ex_meditech()
    # barplot = FigurePlot(out)
    # barplot.plot_bar('meditech', 'count', 'the distribution of meditech')

    ## 箱型图-影像检查分布--人均数据量分布
    # bardata = DataAggregation('fuo', 'ORIGIN_MEDITECH')
    # out = bardata.data_ex_meditech_av()
    # boxplot = FigurePlot(out)
    # boxplot.plot_boxplot("type", "count", 'Per capita data distribution of meditech')

    ## 柱状图-实验室化验分布 
    # bardata = DataAggregation('fuo', 'ORIGIN_MEASUREMENT')
    # out = bardata.data_ex_mea()
    # barplot = FigurePlot(out)
    # barplot.plot_bar_nospines('mea', 'count', 'Measurement')

    # ## 箱型图-实验室化验分布--人均数据量分布 --大项
    # bardata = DataAggregation('fuo', 'ORIGIN_MEASUREMENT')
    # out = bardata.data_ex_mea_av_group()
    # boxplot = FigurePlot(out)
    # boxplot.plot_boxplot("type", "count", 'Per capita data distribution of mea group')

    # ## 箱型图-实验室化验分布--人均数据量分布 --小项
    # bardata = DataAggregation('fuo', 'ORIGIN_MEASUREMENT')
    # out = bardata.data_ex_mea_av_item()
    # boxplot = FigurePlot(out)
    # boxplot.plot_boxplot("item", "count", 'Per capita data distribution of mea item')

    ## 柱状图-文本数据分布 
    # bardata = DataAggregation('fuo', 'ORIGIN_NOTE')
    # out = bardata.data_ex_note()
    # barplot = FigurePlot(out)
    # barplot.plot_bar('note', 'count', 'the distribution of note')

    ## 柱状图-主诉症状数据分布 
    # bardata = DataAggregation('fuo', 'NOTE_ZS')
    # out = bardata.data_ex_zs()
    # barplot = FigurePlot(out)
    # barplot.plot_bar_spines('chiefcomplaint', 'count', 'chief complaint')

    ##柱状图-发热时长分布（主诉）
    # histdata = DataAggregation('fuo', 'NOTE_ZS')
    # out = histdata.data_ex_fever_dur()
    # hisplot = FigurePlot(out)
    # hisplot.plot_histwithbox('duration', 'count', "the distribution of fever duration")

    ## 柱状图-首次病程记录症状数据分布 
    # bardata = DataAggregation('fuo', 'NOTE_ZZ')
    # out = bardata.data_ex_zz()
    # barplot = FigurePlot(out)
    # barplot.plot_bar_spines('symptom', 'count', 'symptom') 
  
if __name__ == '__main__':

    plot()

