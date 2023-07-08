# Differential diagnosis of fever of unknown origin based on task decomposition strategy with knowledge infused deep learning

This repository implements the models for paper '[Differential diagnosis of fever of unknown origin based on task decomposition strategy with knowledge infused deep learning]'.

## ***1. DATA PREPARATION***

### ***1.1 Data Source and Storage***
We have storage the original data in the ORACLE database in server `10.12.45.53`, if you want to reconnect the database and check the data, you can first check the status of database listener, using following code:
```shell
lsnrctl status
```
and you will get the message below, which indicates that our database is not opened.
```shell
zju@w740node4:~$ lsnrctl status

LSNRCTL for Linux: Version 11.2.0.1.0 - Production on 06-JUN-2023 20:35:23

Copyright (c) 1991, 2009, Oracle.  All rights reserved.

Connecting to (DESCRIPTION=(ADDRESS=(PROTOCOL=IPC)(KEY=EXTPROC1521)))
STATUS of the LISTENER
------------------------
Alias                     LISTENER
Version                   TNSLSNR for Linux: Version 11.2.0.1.0 - Production
Start Date                27-FEB-2022 19:23:07
Uptime                    464 days 1 hr. 12 min. 16 sec
Trace Level               off
Security                  ON: Local OS Authentication
SNMP                      OFF
Listener Parameter File   /home/zju/app/zju/product/11.2.0/dbhome_1/network/admin/listener.ora
Listener Log File         /home/zju/app/zju/diag/tnslsnr/w740node4/listener/alert/log.xml
Listening Endpoints Summary...
  (DESCRIPTION=(ADDRESS=(PROTOCOL=ipc)(KEY=EXTPROC1521)))
  (DESCRIPTION=(ADDRESS=(PROTOCOL=tcp)(HOST=10.12.45.53)(PORT=1521)))
The listener supports no services
The command completed successfully
```
Then type the code below to reopen the database:
```SQL
zju@w740node4:~$ sqlplus /nolog
SQL> conn / as sysdba
SQL> startup
ORACLE instance started.

Total System Global Area 6.7344E+10 bytes
Fixed Size		    2220032 bytes
Variable Size		 3.9192E+10 bytes
Database Buffers	 2.7917E+10 bytes
Redo Buffers		  232648704 bytes
Database mounted.
Database opened.
```
if the message shown like above, then the database is correctly reopened. we have saved the database related information in `/mnt/data/wzx/jupyter_notebook/HC4FUO/config/db_conn.yml`.

### ***1.2 Data Extraction and Preparation***
During extraction process, we have done some necessary operation to improve data quality using SQL code, including visit and patient's unique index, data duplication, data clean, etc. All the SQL scripts can be found under `E:\【8】Research\【20200810】发热鉴别诊断研究\SQL` in my own computer. There are two stage of my operation:  
1. Operation in server belonging to ZHIJIANG LAB:  
    - Inspect the distribution of FUO patients and visits, create tablespaces and tables to storage my FUO related data, merge the table, delete unnecessary columns, cross check index, etc.  
    This part operation is ranging from `2020.07.02` to `2020.10.27`, and detailed information can be found in file `./0-20200702-FUO.sql`.
    - Do more preprocessing operation, including extract data into `FUO` user's tablespace, match the fever records in inpatients and outpatients, calculate the statistics of each table, re-split multiple inpatient record of same person, etc.  
    This part operation is ranging from `2020.11.09` to `2020.12.25`, and detailed information can be found in file `./1-20201109-FUO.sql`.

    ***FUCK THE STUPID GUY, DELETE THE WHOLE DATABASE!!!!!!***

2. Operation in server `10.12.45.53`:  
    - Rebuild the database in `10.12.45.53`, and extract the FUO data from ZHIJIANG LAB server to database in `10.12.45.53`. original csv file can be found under `E:\【8】Research\【20200810】发热鉴别诊断研究\Data`, and the SQL scripts building the tables in database can be found in `./2-20210112-FUO.sql`.
    - Re-preprocess the original data. First, do text structuring, and the mainly text categories are as follows:
        
        text type|count
        :-:|:-:
        主诉|33159
        个人史|33159
        既往史|33159
        现病史|33159
        婚育史|33159
        家族史|33159
        首次病程记录|33583
    
        All the codes can be found in script `0_Data_preparation.ipynb`, we have transform the free text of `主诉`,`个人史`, `既往史` into tabular data by string matching and regular expression algorithms, all the tabular data can be found in table `P1_NOTE_ZS`, `P1_NOTE_GRS`, `P1_NOTE_JWS`, respectively. However, the free text of `婚育史`, `家族史` have not been processed according to the opinion of clinicians. 

        For free text of `首次病程记录`, we first split the free text into 9 subcategories, including `主诉`, `初步诊断`, `病例特点`, `病史`, `症状`, `体征`, `辅助检查`, `鉴别诊断`, `诊疗措施`, which are all stored in table `NOTE_SCBCJL_SEMI`. We further transform of these subcategories free text into tabular data, including gender, age, symptoms, differential diagnosis, which are all stored in table `NOTE_INFO`, `P1_NOTE_ZZ`, `NOTE_JBZD`.  

        Considering that there are symptoms in both `P1_NOTE_ZZ` and `P1_NOTE_ZS`, therefore, we have done some operation to merge and generalize these information.  

        Then, we extract the weight and height of patients from table `SIGNS`, further complete the diagnostic labels, delete the abnormal value in measurement data.

        Until now, all the data are tabular data, and more detailed processing information can be found in markdown file `E:\【10】MarkDown\研究进展\日常研究笔记记录.md`.

        Second, for time-series data, we have extract the data with different time length (i.e., 48h, 72h, 96h, 120h), and get their statistics as the static features. Of course, we have also processed the abnormal value, duplicated records, etc. All the related code can be found in `2_Xhours_tsdata_preprocessing.ipynb`.

        At last, based on the aforementioned operation and data, we have transform them into `.npy` file, organized by unique `visit_record_id`. All related codes can be found in `2_multimodality_preprocessing.ipynb`. Moreover, we have gather these data together and split into K folds. All the processed data can be seen under `./data`, and we have process data several times, therefore, we use `phase_xxx` to distinguish these data. All of the experiment under this repository are based on the data under `./data/phase_viii`.

## ***2. EXPERIMENTS***
### ***2.1 Experiments Records***
This part we will record all of the experiments we have conducted. The main focus is on the logs, outputs, and relevant information regarding the hyperparameter training experiment based on Optuna, which was ultimately used in the paper.
#### ***2.1.1 Baseline models***
  - ***Logistic Regression (LR)***  
  The hyperparameter optimization related information can be seen as follows:

    item name|value
    :-:|:-:
    running server|10.12.45.53
    script name|5_hyperparameter_tuner_LR_423dim_48hrs_optuna.py
    experiment time range|2022.06.04-2022.06.05
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.45.53
    model tunning file location|./model_tuning/phase_viii/48hours/20220604/LR_423dim_48hrs_Paper(optuna)
    model tunning and validation file location|./output/phase_viii/48hours/20220604/LR_423dim_48hrs_Paper(optuna)
    test results file location|./test/phase_viii/48hours/20220604/LR_423dim_48hrs_Paper(optuna)

  - ***Support Vector Machine (SVM)***
  The hyperparameter optimization related information can be seen as follows:  

    item name|value
    :-:|:-:
    running server|10.12.45.53
    script name|5_hyperparameter_tuner_SVM_423dim_48hrs_optuna.py
    experiment time range|2022.06.04-2022.06.09
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.45.53
    model tunning file location|./model_tuning/phase_viii/48hours/20220604/SVM_423dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220604/SVM_423dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220604/SVM_423dim_48hrs_Paper(optuna)

  - ***Random Forest (RF)***
  The hyperparameter optimization related information can be seen as follows:
    item name|value
    :-:|:-:
    running server|10.12.45.53
    script name|5_hyperparameter_tuner_RF_423dim_48hrs_optuna.py
    experiment time range|2022.06.05-2022.06.05
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.45.53
    model tunning file location|./model_tuning/phase_viii/48hours/20220604/RF_423dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220604/RF_423dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220604/RF_423dim_48hrs_Paper(optuna)

  - ***Gradient Boosting Decision Tree (GBDT)***
  The hyperparameter optimization related information can be seen as follows:  

    item name|value
    :-:|:-:
    running server|10.12.43.43
    script name|5_hyperparameter_tuner_GBDT_423dim_48hrs_optuna.py
    experiment time range|2022.06.09-2022.06.09
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.43
    model tunning file location|./model_tuning/phase_viii/48hours/20220604/GBDT_423dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220604/GBDT_423dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220604/GBDT_423dim_48hrs_Paper(optuna)

#### ***2.1.2 Single-modality deep models***
  ***注意！！！此处DNN模型我们实现了两次，`DNN_423dim_48hrs` 是包含了时序数据的统计特征的423维输入的DNN模型，`DNN_393dim_48hrs` 是不包含时序数据的统计特征的393维输入的DNN模型。后者才是论文中的 `Single-modality deep models`列下的模型***。
  - ***DNN_423dim_48hrs***
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    running server|10.12.43.45
    script name|5_hyperparameter_tuner_DNN_423dim_48hrs_optuna.py
    experiment time range|2022.06.05-2022.06.05
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.45
    model tunning file location|./model_tuning/phase_viii/48hours/20220604/DNN_423dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220604/DNN_423dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220604/DNN_423dim_48hrs_Paper(optuna)

  - ***DNN_393dim_48hrs***
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    running server|10.12.45.53
    script name|5_hyperparameter_tuner_DNN_393dim_48hrs_optuna.py
    experiment time range|2022.06.05-2022.06.05
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.45.53
    model tunning file location|./model_tuning/phase_viii/48hours/20220604/DNN_393dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220604/DNN_393dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220604/DNN_393dim_48hrs_Paper(optuna)

  - ***GRUD_PreAttenSpatial_7dim_48hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    running server|10.12.43.43
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_48hrs_optuna.py
    experiment time range|2022.04.22-2022.05.09 / 2022.06.20-2022.06.28
    dataset|phase_viii 
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.43
    model tunning file location|./model_tuning/phase_viii/48hours/20220401/GRUD_PreAttenSpatial_7dim_48hrs(optuna) ./model_tuning/phase_viii/48hours/20220604/GRUD_PreAttenSpatial_7dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220401/GRUD_PreAttenSpatial_7dim_48hrs(optuna) ./output/phase_viii/48hours/20220604/GRUD_PreAttenSpatial_7dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220401/GRUD_PreAttenSpatial_7dim_48hrs(optuna) ./test/phase_viii/48hours/20220604/GRUD_PreAttenSpatial_7dim_48hrs_Paper(optuna)

    ***注意！！！该模型的超参数优化过程主要在 `10.12.43.43` 服务器上完成，但是分别在4月份和6月份跑过两次，因此任务1，任务2和任务5是取自6月份的实验结果，任务3和任务4是取自4月份的实验结果，目的是取其优化测试结果的最好表现。***

  - ***GRUD_PreAttenSpatial_7dim_72hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    running server|10.12.43.45
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_72hrs_optuna.py
    experiment time range|2022.04.25-2022.05.13 / 2022.06.13-2022.06.29
    dataset|phase_viii 
    max timestamp|72\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.45
    model tunning file location|./model_tuning/phase_viii/72hours/20220401/GRUD_PreAttenSpatial_7dim_72hrs(optuna) ./model_tuning/phase_viii/72hours/20220604/GRUD_PreAttenSpatial_7dim_72hrs_Paper(optuna)
    model validation file location|./output/phase_viii/72hours/20220401/GRUD_PreAttenSpatial_7dim_72hrs(optuna) ./output/phase_viii/72hours/20220604/GRUD_PreAttenSpatial_7dim_72hrs_Paper(optuna)
    model test file location|./test/phase_viii/72hours/20220401/GRUD_PreAttenSpatial_7dim_72hrs(optuna) ./test/phase_viii/72hours/20220604/GRUD_PreAttenSpatial_7dim_72hrs_Paper(optuna)

    ***注意！！！该模型的超参数优化过程主要在 `10.12.43.45` 服务器上完成，但是分别在4月份和6月份跑过两次，因此任务1，任务2和任务3是取自6月份的实验结果，任务4和任务5是取自4月份的实验结果，目的是取其优化测试结果的最好表现。***

  - ***GRUD_PreAttenSpatial_7dim_96hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    running server|10.12.43.46
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_96hrs_optuna.py
    experiment time range|2022.04.24-2022.05.12 / 2022.06.13-2022.06.26
    dataset|phase_viii 
    max timestamp|96\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.46
    model tunning file location|./model_tuning/phase_viii/96hours/20220401/GRUD_PreAttenSpatial_7dim_96hrs(optuna) ./model_tuning/phase_viii/96hours/20220604/GRUD_PreAttenSpatial_7dim_96hrs_Paper(optuna)
    model validation file location|./output/phase_viii/96hours/20220401/GRUD_PreAttenSpatial_7dim_96hrs(optuna) ./output/phase_viii/96hours/20220604/GRUD_PreAttenSpatial_7dim_96hrs_Paper(optuna)
    model test file location|./test/phase_viii/96hours/20220401/GRUD_PreAttenSpatial_7dim_96hrs(optuna) ./test/phase_viii/96hours/20220604/GRUD_PreAttenSpatial_7dim_96hrs_Paper(optuna)

    ***注意！！！该模型的超参数优化过程主要在 `10.12.43.46` 服务器上完成，但是分别在4月份和6月份跑过两次，由于五个任务都在6月份的实验中取得了最好的表现，因此我们的论文表2中的结果，采取的就是6月份实验的结果。***

  - ***GRUD_PreAttenSpatial_7dim_120hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    running server|10.12.45.53
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_120hrs_optuna.py
    experiment time range|2022.04.26-2022.05.21 / 2022.07.04-2022.07.15
    dataset|phase_viii 
    max timestamp|120\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.45.53
    model tunning file location|./model_tuning/phase_viii/120hours/20220401/GRUD_PreAttenSpatial_7dim_120hrs(optuna) ./model_tuning/phase_viii/120hours/20220604/GRUD_PreAttenSpatial_7dim_120hrs_Paper(optuna)
    model validation file location|./output/phase_viii/120hours/20220401/GRUD_PreAttenSpatial_7dim_120hrs(optuna) ./output/phase_viii/120hours/20220604/GRUD_PreAttenSpatial_7dim_120hrs_Paper(optuna)
    model test file location|./test/phase_viii/120hours/20220401/GRUD_PreAttenSpatial_7dim_120hrs(optuna) ./test/phase_viii/120hours/20220604/GRUD_PreAttenSpatial_7dim_120hrs_Paper(optuna)

    ***注意！！！该模型的超参数优化过程主要在 `10.12.45.53` 服务器上完成，但是分别在4月份和7月份跑过两次，我们论文表2中的结果中，任务1和任务2取自7月份的优化测试结果，任务3，任务4和任务5取自4月份的优化测试结果。***

  ***除此之外，`./test/phase_viii/Xhours/` 目录下除了上述 `20220401` 和 `20220604` 的文件夹之外，还有一个 `20220716` 的文件夹，其中的文件都是从各个服务器的测试集结果中取的表现最好的结果对应的数据文件，也是画论文中图6的数据来源。***

#### ***2.1.3 Multi-modality deep models***
  ***注意！！！对多模态的模型的超参数调优实验，是在四个服务器上插空跑的，因此分布的比较分散，最后论文中的结果也是在各个服务器的优化测试结果中挑选的最优结果。且该部分多模态模型基本都是基于48小时数据训练，也有极个别是是基于其他时间长度的数据训练的，目的可能就是对比看一下是否有更好的表现***。

  - ***HMMLF_Concat_7dim_48hrs_Paper(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    running server|10.12.43.43
    script name|5_hyperparameter_tuner_HMMLF_PreAtten_7dim_48hrs_optuna.py
    task | task 2, task 3, task 4
    experiment time range|2022.05.14-2022.05.22 / 2022.06.06-2022.06.08
    dataset|phase_viii 
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.43
    model tunning file location|./model_tuning/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs(optuna) ./model_tuning/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs(optuna) ./output/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs(optuna) ./test/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)

    ***但是最后的结果就是，该服务器上跑的这三个任务都未取得最优实验结果。因此未被写入论文中。***

    item name|value
    :-:|:-:
    running server|10.12.43.45
    script name|5_hyperparameter_tuner_HMMLF_PreAtten_7dim_48hrs_optuna.py 5_hyperparameter_tuner_HMMLF_PreAtten_7dim_48hrs_manual.ipynb
    task | task -1, task 0, task 4(两次) 以及 task 2, task 3, task 4
    experiment time range|2022.06.06-2022.06.12
    dataset|phase_viii 
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.45
    model tunning file location|./model_tuning/phase_viii/48hours/20220604/HMMLF_423dim_48hrs_Paper(manual) ./model_tuning/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220604/HMMLF_423dim_48hrs_Paper(manual) ./test/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)

    ***注意！！！本服务器上实际上只跑了 `HMMLF_Concat_7dim_48hrs_Paper(optuna)` 一个超参数优化实验，且只覆盖了任务-1，任务0，两次任务4，且论文中的任务1即取自该超参数优化实验的结果。但是 `HMMLF_423dim_48hrs_Paper(manual)` 文件夹下也有任务2，任务3， 任务4一共三个任务的文件，但是没有其超参数优化的相关log文件，不知道是哪来的数据。需要注意的是，论文中的任务3和任务4的测试集结果就来自于该文件下的任务2和任务3的结果。***

    item name|value
    :-:|:-:
    running server|10.12.43.46
    script name|5_hyperparameter_tuner_HMMLF_PreAtten_7dim_48hrs_optuna.py
    task | task 2, task 3, task 4 (4月份和6月份跑了两次)
    experiment time range|2022.06.06-2022.06.07
    dataset|phase_viii 
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.43.46
    model tunning file location|./model_tuning/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs_AUC(optuna) ./model_tuning/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs_AUC(optuna) ./output/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs_AUC(optuna) ./test/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)

    ***本服务器上在4月份和6月份跑过两次实验，4月份只跑了任务4，也是论文中任务5采取的测试集结果。6月份跑了任务2,3,4，但是表现都不是最优的。除此之外，本服务器上还跑了多模态数据输入条件下的96hours和120hours对应的两组超参数优化实验。***

    item name|value
    :-:|:-:
    running server|10.12.45.53
    script name|5_hyperparameter_tuner_HMMLF_PreAtten_7dim_48hrs_optuna.py
    task | task 1, task 3 (跑了两次)
    experiment time range|2022.06.02-2022.06.07
    dataset|phase_viii 
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100
    model tunning file server|10.12.45.53
    model tunning file location|./model_tuning/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs_AUC(optuna) ./model_tuning/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model validation file location|./output/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs_AUC(optuna) ./output/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)
    model test file location|./test/phase_viii/48hours/20220401/HMMLF_Concat_7dim_48hrs_AUC(optuna) ./test/phase_viii/48hours/20220604/HMMLF_Concat_7dim_48hrs_Paper(optuna)

    ***本服务器上也跑过两次实验，第一次只跑了任务3， 第二次跑了任务1和任务3，其中任务1的优化结果，就是论文中任务2的数据的来源。除此之外，在各个 `HMMLF_CONCAT_423dim_48hrs_Paper(last)` 文件夹下存储的文件就是多模态融合的48hours数据的最优测试集结果及其相关模型参数。***

#### ***2.1.4 Single-modality models for revisions***

  - ***GRUD_MtoAtten_7dim_48hrs_Paper***

    > nohup python 5_hyperparameter_tuner_GRUD_MtoAtten_7dim_48hrs_optuna.py >> ./shell_log/phase_viii/48hours/20230610/GRUD_MtoAtten_7dim_48hrs_Paper/task4.log &

    > nohup python 5_hyperparameter_tuner_GRUD_MtoAtten_7dim_48hrs_optuna.py >> ./shell_log/phase_viii/48hours/20230610/GRUD_MtoAtten_7dim_48hrs_Paper/task3.log &

  - ***GRUD_NoAtten_7dim_48hrs_Paper***

    > nohup python 5_hyperparameter_tuner_GRUD_NoAtten_7dim_48hrs_optuna.py >> ./shell_log/phase_viii/48hours/20230610/GRUD_NoAtten_7dim_48hrs_Paper/task2.log &

    > nohup python 5_hyperparameter_tuner_GRUD_NoAtten_7dim_48hrs_optuna.py >> ./shell_log/phase_viii/48hours/20230610/GRUD_NoAtten_7dim_48hrs_Paper/task3.log &

  - ***LSTM_7dim_48hrs_mean2_Paper***

    > nohup python 5_hyperparameter_tuner_LSTM_7dim_48hrs_optuna.py >> ./shell_log/phase_viii/48hours/20230610/LSTM_7dim_48hrs_mean2_Paper/task4.log &

    > nohup python 5_hyperparameter_tuner_LSTM_7dim_48hrs_optuna.py >> ./shell_log/phase_viii/48hours/20230610/LSTM_7dim_48hrs_mean2_Paper/task0.log &

  - ***HMMGMU_7dim_48hrs_Paper***

    > nohup python 5_hyperparameter_tuner_HMMGMU_PreAtten_7dim_48hrs_optuna.py >> ./shell_log/phase_viii/48hours/20230610/HMMGMU_7dim_48hrs_Paper/task1.log &