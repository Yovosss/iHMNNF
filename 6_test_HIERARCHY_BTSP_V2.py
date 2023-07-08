# %% [markdown]
# ### 1-Using Bootstrap to calculate the metric of flatten classification and hierarchical classification
#     1 - Logistic Regression
#     2 - Random Forest
#     3 - HMNNF

# %%
from __future__ import absolute_import, division, print_function

import os
import re
import argparse
import joblib
import h5py
import time
import numpy as np
import pandas as pd
from scipy import interp
from datetime import datetime
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.backend import clear_session
from keras import regularizers 
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation, BatchNormalization, InputLayer, Embedding, Reshape, Concatenate, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve, average_precision_score

from Data_helper import LoadDataTrial
from utils.layers import ExternalMasking
from utils.grud_layers import GRUD, Bidirectional_for_GRUD
from BuildModel import LoadModel
from utils.fusion import Gmu as Gmu
from utils.attention_function import attention_3d_block_spatial as PreAttentionSpatial
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# %%
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--working_path', default='.')

## data
arg_parser.add_argument('--dataset_name', default='phase_viii', 
                         help='The data files should be saved in [working_path]/data/[dataset_name] directory.')
arg_parser.add_argument('--fold', type=int, default=0, 
                         help='the fold data taken to use, there are 5 folds or 10 folds')
arg_parser.add_argument('--label', default=-1, type=int, choices=[-1,0,1,2,3,4],
                         help='the label type')

## model
arg_parser.add_argument('--model_type', default='RF', choices=['LR', 'RF', 'SVM', 'DNN', 'DNN_EE', 'GRUD', 'HMM_EF', 'HMM_LF'])
arg_parser.add_argument('--max_timestep', type=int, default=200, 
                        help='Time series of at most # time steps are used. Default: 200.')
arg_parser.add_argument('--max_timestamp', type=int, default=48*60*60,choices=[48*60*60, 72*60*60, 96*60*60, 120*60*60],
                        help='Time series of at most # seconds are used. Default: 48 (hours).')
arg_parser.add_argument('--use_bidirectional_rnn', default=False)
# Train
arg_parser.add_argument('--trainORvalidation', default='Train', choices=['Train', 'Validation'])
arg_parser.add_argument('--epochs', type=int, default=100)
arg_parser.add_argument('--early_stopping_patience', type=int, default=10)
arg_parser.add_argument('--batch_size', type=int, default=32)
# set the actual arguments if running in notebook
if not (__name__ == '__main__' and '__file__' in globals()):
    ARGS = arg_parser.parse_args([
        '--model_type', 'RF',
        '--dataset_name', 'phase_viii',
        '--fold', '0',
        '--epochs', '100',
        '--trainORvalidation', 'Train'
    ])
else:
      ARGS = arg_parser.parse_args()

print('Arguments:', ARGS)

# %%
def plot_cm(types, model_info, y_true, y_pred, runtime):
    # get the confusion matrix
    C = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(C, columns=['0','1','2','3','4','5','6'])
    df_cm['bootstrap_index'] = runtime
    df_cm['label'] = types

    return df_cm
    
def save_report(types, model_info, y_true, y_pred, runtime):
    report = classification_report(y_true, y_pred,digits=5, output_dict = True)
    report = pd.DataFrame(report).transpose().reset_index()
    report['bootstrap_index'] = runtime
    report['label'] = types

    return report

# %%
if ARGS.model_type == 'LR':
    model_path = os.path.join(ARGS.working_path, 'model_tuning', 'phase_viii', '48hours', '20220604', 'LR_423dim_48hrs_Paper(optuna)')
elif ARGS.model_type == 'RF':
    model_path = os.path.join(ARGS.working_path, 'model_tuning', 'phase_viii', '48hours', '20220604', 'RF_423dim_48hrs_Paper(optuna)')
else:
    model_path = os.path.join(ARGS.working_path, 'model_tuning', 'phase_viii', '48hours', '20220604', 'HMMLF_CONCAT_423dim_48hrs_Paper(last)')

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
output_path = os.path.join(ARGS.working_path, 'test', 'phase_viii', '48hours', '20220604', 'Hierarchy_Test(last)')
if not os.path.exists(output_path):
    os.makedirs(output_path)

T = time.strftime("%Y%m%d%H%M%S", time.localtime())
LABEL_DICT = {'-1':'flatten multi-classification', 
              '0': 'infectious and non-infectious', 
              '1': 'bacterial, viral and others', 
              '2': 'NIID and tumor', 
              '3': 'AD and AID', 
              '4': 'HM and SM'}

# Load the data
dataset = LoadDataTrial(data_path=os.path.join('.', 'data', ARGS.dataset_name, '48hours', 'processed', 'raw/data4hc_v20220401'), 
                        model_type=ARGS.model_type,
                        label_name=ARGS.label,
                        max_timestep=ARGS.max_timestep,
                        max_timestamp=ARGS.max_timestamp)
X_train, y_train_1d, nclasses_train, folds_train, shapes_train = dataset.training_generator(ARGS.fold)
X_validation, y_validation_1d, nclasses_validation,folds_validation, shapes_validation = dataset.validation_generator(ARGS.fold)
X_test, y_test_1d, nclasses_test, folds_test, shapes_test = dataset.test_generator(ARGS.fold)

y_train = to_categorical(y_train_1d, num_classes=nclasses_train)           
y_validation = to_categorical(y_validation_1d, num_classes=nclasses_train)
y_test = to_categorical(y_test_1d, num_classes=nclasses_train)

# Load the model
if ARGS.model_type == 'HMM_LF':
    model_f = LoadModel(os.path.join(model_path, 'HMMLF(flatten multi-classification).h5'))
    model_0 = LoadModel(os.path.join(model_path, 'HMMLF(infectious and non-infectious).h5'))
    model_1 = LoadModel(os.path.join(model_path, 'HMMLF(bacterial, viral and others).h5'))
    model_2 = LoadModel(os.path.join(model_path, 'HMMLF(NIID and tumor).h5'))
    model_3 = LoadModel(os.path.join(model_path, 'HMMLF(AD and AID).h5'))
    model_4 = LoadModel(os.path.join(model_path, 'HMMLF(HM and SM).h5'))
elif ARGS.model_type == 'LR':
    model_f = joblib.load(os.path.join(model_path, 'LR(flatten multi-classification).pkl'))
    model_0 = joblib.load(os.path.join(model_path, 'LR(infectious and non-infectious).pkl'))
    model_1 = joblib.load(os.path.join(model_path, 'LR(bacterial, viral and others).pkl'))
    model_2 = joblib.load(os.path.join(model_path, 'LR(NIID and tumor).pkl'))
    model_3 = joblib.load(os.path.join(model_path, 'LR(AD and AID).pkl'))
    model_4 = joblib.load(os.path.join(model_path, 'LR(HM and SM).pkl'))
else:
    print("[Info] The model have been loaded!")
    model_f = joblib.load(os.path.join(model_path, 'RF(flatten multi-classification).pkl'))
    model_0 = joblib.load(os.path.join(model_path, 'RF(infectious and non-infectious).pkl'))
    model_1 = joblib.load(os.path.join(model_path, 'RF(bacterial, viral and others).pkl'))
    model_2 = joblib.load(os.path.join(model_path, 'RF(NIID and tumor).pkl'))
    model_3 = joblib.load(os.path.join(model_path, 'RF(AD and AID).pkl'))
    model_4 = joblib.load(os.path.join(model_path, 'RF(HM and SM).pkl'))


# %%
# set bootstrap on test dataset
df_cms_f = pd.DataFrame(columns=['bootstrap_index', 'label', '0','1','2','3','4','5','6'])
df_cms_h = pd.DataFrame(columns=['bootstrap_index', 'label', '0','1','2','3','4','5','6'])
df_reports_f = pd.DataFrame(columns=['bootstrap_index', 'label', 'index', 'precision', 'recall', 'f1-score', 'support'])
df_reports_h = pd.DataFrame(columns=['bootstrap_index', 'label', 'index', 'precision', 'recall', 'f1-score', 'support'])
# predict the test samples
if ARGS.model_type == 'HMM_LF':
    for runtime in range(1000):
        print("[Info] The {}-th iteration is processing!".format(runtime))
        idx = np.random.randint(0, len(X_test[0]) - 1, size=len(X_test[0]))
        X_test_boot = [x[idx] for x in X_test]
        y_test_boot = y_test_1d[idx]
        folds = folds_test[idx]
        
        y_pred = []
        for s in range(len(y_test_boot)):
            # infectious OR Noninfectious
            y_pred_proba_0 = model_0.predict([np.expand_dims(x[s], axis=0) for x in X_test_boot])
            y_pred_0 = np.argmax(y_pred_proba_0, axis=1)

            if y_pred_0[0] == 0:
                # bacterial, viral OR fungal
                y_pred_proba_1 = model_1.predict([np.expand_dims(x[s], axis=0) for x in X_test_boot])
                y_pred_1 = np.argmax(y_pred_proba_1, axis=1)
                if y_pred_1[0] == 0:
                    y_pred.append(0)
                elif y_pred_1[0] == 1:
                    y_pred.append(1)
                elif y_pred_1[0] == 2:
                    y_pred.append(2)
                    
            elif y_pred_0[0] == 1:
                # NIID OR Neoplastic
                y_pred_proba_2 = model_2.predict([np.expand_dims(x[s], axis=0) for x in X_test_boot])
                y_pred_2 = np.argmax(y_pred_proba_2, axis=1)
                if y_pred_2[0] == 0:
                    # AID OR AIFD
                    y_pred_proba_3 = model_3.predict([np.expand_dims(x[s], axis=0) for x in X_test_boot])
                    y_pred_3 = np.argmax(y_pred_proba_3, axis=1)
                    if y_pred_3[0] == 0:
                        y_pred.append(3)
                    elif y_pred_3[0] == 1:
                        y_pred.append(4)
                elif y_pred_2[0] == 1:
                    # HM OR SM
                    y_pred_proba_4 = model_4.predict([np.expand_dims(x[s], axis=0) for x in X_test_boot])
                    y_pred_4 = np.argmax(y_pred_proba_4, axis=1)
                    if y_pred_4[0] == 0:
                        y_pred.append(5)
                    elif y_pred_4[0] == 1:
                        y_pred.append(6)
        
        # predict respectively
        y_pred_proba_f = model_f.predict(X_test)
        # Process the predicted label of hierarchy
        y_pred_f = np.argmax(y_pred_proba_f, axis=1)

        
        # save the confusion matrix
        df_cm_h = plot_cm('Hierarchy', ARGS.model_type, y_test_boot.tolist(), y_pred, runtime)
        df_cms_h = pd.concat([df_cms_h, df_cm_h], axis=0, ignore_index=True)
        
        df_cm_f = plot_cm('Flatten', ARGS.model_type, y_test_boot.tolist(), y_pred_f.tolist(), runtime)
        df_cms_f = pd.concat([df_cms_f, df_cm_f], axis=0, ignore_index=True)
        
        # save the reports
        df_report_h = save_report('Hierarchy', ARGS.model_type, y_test_boot.tolist(), y_pred, runtime)
        df_reports_h = pd.concat([df_reports_h, df_report_h], axis=0, ignore_index=True)
        
        df_report_f = save_report('Flatten', ARGS.model_type, y_test_boot.tolist(),  y_pred_f.tolist(), runtime)
        df_reports_f = pd.concat([df_reports_f, df_report_f], axis=0, ignore_index=True)

    df_cms_h.to_csv(os.path.join(output_path, 'confusion_matrix(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Hierarchy')))
    df_cms_f.to_csv(os.path.join(output_path, 'confusion_matrix(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Flatten')))
    df_reports_h.to_csv(os.path.join(output_path, 'classification_reuslts(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Hierarchy')))
    df_reports_f.to_csv(os.path.join(output_path, 'classification_reuslts(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Flatten')))     
        
else:
    for runtime in range(1000):
        print("[Info] The {}-th iteration is processing!".format(runtime))
        idx = np.random.randint(0, len(X_test) - 1, size=len(X_test))
        X_test_boot = X_test[idx]
        y_test_boot = y_test_1d[idx]
        folds = folds_test[idx]
        
        y_pred = []
        for s in range(len(y_test_boot)):
            # infectious OR Noninfectious
            y_pred_0 = model_0.predict(np.expand_dims(X_test_boot[s], axis=0))

            if y_pred_0[0] == 0:
                # bacterial, viral OR fungal
                y_pred_1 = model_1.predict(np.expand_dims(X_test_boot[s], axis=0))
                if y_pred_1[0] == 0:
                    y_pred.append(0)
                elif y_pred_1[0] == 1:
                    y_pred.append(1)
                elif y_pred_1[0] == 2:
                    y_pred.append(2)
                    
            elif y_pred_0[0] == 1:
                # NIID OR Neoplastic
                y_pred_2 = model_2.predict(np.expand_dims(X_test_boot[s], axis=0))
                if y_pred_2[0] == 0:
                    # AID OR AIFD
                    y_pred_3 = model_3.predict(np.expand_dims(X_test_boot[s], axis=0))
                    if y_pred_3[0] == 0:
                        y_pred.append(3)
                    elif y_pred_3[0] == 1:
                        y_pred.append(4)
                elif y_pred_2[0] == 1:
                    # HM OR SM
                    y_pred_4 = model_4.predict(np.expand_dims(X_test_boot[s], axis=0))
                    if y_pred_4[0] == 0:
                        y_pred.append(5)
                    elif y_pred_4[0] == 1:
                        y_pred.append(6)
        # predict respectively
        y_pred_f = model_f.predict(X_test_boot)

        # save the confusion matrix
        df_cm_h = plot_cm('Hierarchy', ARGS.model_type, y_test_boot.tolist(), y_pred, runtime)
        df_cms_h = pd.concat([df_cms_h, df_cm_h], axis=0, ignore_index=True)
        
        df_cm_f = plot_cm('Flatten', ARGS.model_type, y_test_boot.tolist(), y_pred_f.tolist(), runtime)
        df_cms_f = pd.concat([df_cms_f, df_cm_f], axis=0, ignore_index=True)
        
        # save the reports
        df_report_h = save_report('Hierarchy', ARGS.model_type, y_test_boot.tolist(), y_pred, runtime)
        df_reports_h = pd.concat([df_reports_h, df_report_h], axis=0, ignore_index=True)
        
        df_report_f = save_report('Flatten', ARGS.model_type, y_test_boot.tolist(),  y_pred_f.tolist(), runtime)
        df_reports_f = pd.concat([df_reports_f, df_report_f], axis=0, ignore_index=True)

    df_cms_h.to_csv(os.path.join(output_path, 'confusion_matrix(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Hierarchy')))
    df_cms_f.to_csv(os.path.join(output_path, 'confusion_matrix(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Flatten')))
    df_reports_h.to_csv(os.path.join(output_path, 'classification_reuslts(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Hierarchy')))
    df_reports_f.to_csv(os.path.join(output_path, 'classification_reuslts(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Flatten')))   

# %% [markdown]
# ### 2 - To get the statistics of bootstraping results

# %%
df_h = pd.read_csv(os.path.join(output_path, 'classification_reuslts(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Hierarchy')))
df_f = pd.read_csv(os.path.join(output_path, 'classification_reuslts(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Flatten')))

# %%
# Get the summary data of flatten classification
accuracy_f = df_f.loc[df_f['index']=='accuracy', :]['precision'].values
macro_precision_f = df_f.loc[df_f['index'] == 'macro avg', :]['precision'].values
macro_recall_f = df_f.loc[df_f['index'] == 'macro avg', :]['recall'].values
macro_f1_f = df_f.loc[df_f['index'] == 'macro avg', :]['f1-score'].values

summary = {'accuracy': {'mean': accuracy_f.mean(), 'std': accuracy_f.std()},
           'macro_precison':{'mean': macro_precision_f.mean(), 'std': macro_precision_f.std()},
           'macro_recall': {'mean': macro_recall_f.mean(), 'std': macro_recall_f.std()},
           'macro_f1': {'mean': macro_f1_f.mean(), 'std': macro_f1_f.std()}}
df_sum = pd.DataFrame(summary)
df_sum.to_csv(os.path.join(output_path, 'Summary(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Flatten')))
# df_sum

# %%
# Get the summary data of flatten classification
accuracy_h = df_h.loc[df_h['index']=='accuracy', :]['precision'].values
macro_precision_h = df_h.loc[df_h['index'] == 'macro avg', :]['precision'].values
macro_recall_h = df_h.loc[df_h['index'] == 'macro avg', :]['recall'].values
macro_f1_h = df_h.loc[df_h['index'] == 'macro avg', :]['f1-score'].values

summary = {'accuracy': {'mean': accuracy_h.mean(), 'std': accuracy_h.std()},
           'macro_precison':{'mean': macro_precision_h.mean(), 'std': macro_precision_h.std()},
           'macro_recall': {'mean': macro_recall_h.mean(), 'std': macro_recall_h.std()},
           'macro_f1': {'mean': macro_f1_h.mean(), 'std': macro_f1_h.std()}}
df_sum = pd.DataFrame(summary)
df_sum.to_csv(os.path.join(output_path, 'Summary(BTSP)({0})({1}).csv'.format(ARGS.model_type, 'Hierarchy')))
# df_sum

# %%


# %%



