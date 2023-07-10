
# coding = utf-8

"""
@author: wang zhixiao
@time  : 2020-12-07 11:36
@detail: 主要是对各家医院数据库进行连接
"""

import os
os.environ["ORACLE_BASE"] = '/home/zju/app'
os.environ["ORACLE_SID"] = 'orcl'
os.environ["ORACLE_HOME"] = '/home/zju/app/zju/product/11.2.0/dbhome_1'
# os.environ["LD_LIBRARY_PATH"] = '$LD_LIBRARY_PATH:$ORACLE_HOME/lib'
from sqlalchemy import create_engine, Table, MetaData
import cx_Oracle
from config.utils import read_yml
os.environ["NLS_LANG"] = "GERMAN_GERMANY.UTF8"

# 通过配置文件建立数据库引擎
def sql_engine(oracle_config, encoding = 'utf-8'):
    user = oracle_config['user']
    password = oracle_config['password']
    ip = oracle_config['ip']
    port = oracle_config['port']
    sid = oracle_config['sid']
    if user == 'sys':
        return create_engine('oracle+cx_oracle://%s:%s@%s:%s/%s' %(user, password, ip, port, sid),
            connect_args={
        "encoding": "UTF-8",
        "mode": cx_Oracle.SYSDBA,
        "events": True
    },
            echo = False)  #echo为True会将执行语句打印出来
    else:
        return create_engine('oracle+cx_oracle://%s:%s@%s:%s/%s' %(user, password, ip, port, sid),
            encoding =  "UTF-8",
            echo = False)  #echo为True会将执行语句打印出来

# 单数据库引擎配置
class db_connection:

    def __init__(self, conn_name):

        """
        conn_name: 对应配置文件db_conn.yml中的连接名
        """

        # self.config = read_yml('./config/db_conn.yml')
        self.config = read_yml(os.path.dirname(__file__) + "/db_conn.yml")
        # 连接数据库
        self.engine = sql_engine(oracle_config = self.config[conn_name],
                                encoding = 'utf-8')
        # 绑定引擎
        self.metadata = MetaData(self.engine)
        self.conn = self.engine.connect()