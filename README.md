# Integrating Medical Domain Knowledge for Early Diagnosis of Fever of Unknown Origin: An Interpretable Hierarchical Multimodal Neural Network Approach

This repository implements the models for paper '[Integrating Medical Domain Knowledge for Early Diagnosis of Fever of Unknown Origin: An Interpretable Hierarchical Multimodal Neural Network Approach]'.

There are still many changes that need to be updated, please wait......

## ***1. DATA PREPARATION***

### ***1.1 Data Source and Storage***
We have storage the original data in the ORACLE database in server `xx.xx.xx.xx`.

### ***1.2 Data Extraction and Preparation***
During extraction process, we have done some necessary operation to improve data quality using SQL code, including visit and patient's unique index, data duplication, data clean, etc. 

1. Operation in server 1:  
    - Inspect the distribution of FUO patients and visits, create tablespaces and tables to storage my FUO related data, merge the table, delete unnecessary columns, cross check index, etc.
    - Do more preprocessing operation, including extract data into `FUO` user's tablespace, match the fever records in inpatients and outpatients, calculate the statistics of each table, re-split multiple inpatient record of same person, etc.  

2. Operation in server 2:  
    - Rebuild the database, and extract the FUO data from  server 1 to database in server 2. 

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
    
        All the codes can be found in script `0_Data_preparation.ipynb`, we have transform the free text into tabular data by string matching and regular expression algorithms.

        For free text of `首次病程记录`, we first split the free text into 9 subcategories, including `主诉`, `初步诊断`, `病例特点`, `病史`, `症状`, `体征`, `辅助检查`, `鉴别诊断`, `诊疗措施`, which are all stored in table `NOTE_SCBCJL_SEMI`. We further transform of these subcategories free text into tabular data, including gender, age, symptoms, differential diagnosis, which are all stored in table `NOTE_INFO`, `P1_NOTE_ZZ`, `NOTE_JBZD`.  

        Considering that there are symptoms in both `P1_NOTE_ZZ` and `P1_NOTE_ZS`, therefore, we have done some operation to merge and generalize these information.  

        Then, we extract the weight and height of patients from table `SIGNS`, further complete the diagnostic labels, delete the abnormal value in measurement data.

        Until now, all the data are tabular data.

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
    script name|5_hyperparameter_tuner_LR_423dim_48hrs_optuna.py
    experiment time range|2022.06.04-2022.06.05
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100

  - ***Support Vector Machine (SVM)***
  The hyperparameter optimization related information can be seen as follows:  

    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_SVM_423dim_48hrs_optuna.py
    experiment time range|2022.06.04-2022.06.09
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100

  - ***Random Forest (RF)***
  The hyperparameter optimization related information can be seen as follows:
    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_RF_423dim_48hrs_optuna.py
    experiment time range|2022.06.05-2022.06.05
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100


#### ***2.1.2 Single-modality deep models***

  - ***DNN_393dim_48hrs***
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_DNN_393dim_48hrs_optuna.py
    experiment time range|2022.06.05-2022.06.05
    dataset|phase_viii
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100

  - ***GRUD_PreAttenSpatial_7dim_48hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_48hrs_optuna.py
    experiment time range|2022.04.22-2022.05.09 / 2022.06.20-2022.06.28
    dataset|phase_viii 
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100

  - ***GRUD_PreAttenSpatial_7dim_72hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_72hrs_optuna.py
    experiment time range|2022.04.25-2022.05.13 / 2022.06.13-2022.06.29
    dataset|phase_viii 
    max timestamp|72\*60\*60
    epochs|100
    batch size|32
    optuna trial|100

  - ***GRUD_PreAttenSpatial_7dim_96hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_96hrs_optuna.py
    experiment time range|2022.04.24-2022.05.12 / 2022.06.13-2022.06.26
    dataset|phase_viii 
    max timestamp|96\*60\*60
    epochs|100
    batch size|32
    optuna trial|100

  - ***GRUD_PreAttenSpatial_7dim_120hrs(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_GRUD_PreAtten_7dim_120hrs_optuna.py
    experiment time range|2022.04.26-2022.05.21 / 2022.07.04-2022.07.15
    dataset|phase_viii 
    max timestamp|120\*60\*60
    epochs|100
    batch size|32
    optuna trial|100

#### ***2.1.3 Multi-modality deep models***

  - ***HMMLF_Concat_7dim_48hrs_Paper(optuna)***  
  The hyperparameter optimization related information can be seen as follows: 

    item name|value
    :-:|:-:
    script name|5_hyperparameter_tuner_LaMNN_PreAtten_7dim_48hrs_optuna.py
    task | task 2, task 3, task 4
    experiment time range|2022.05.14-2022.05.22 / 2022.06.06-2022.06.08
    dataset|phase_viii 
    max timestamp|48\*60\*60
    epochs|100
    batch size|32
    optuna trial|100


## ***3. ENVIRONMENT***
The main environment configuration can be seen below:
```
conda env update -n iHMNNF -f environment.yml
source activate iHMNNF
pip install -e .
```
there may be some conflicts that need to be manually fixed.