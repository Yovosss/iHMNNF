# %%
from __future__ import absolute_import, division, print_function

import os
import time
import joblib
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from datetime import datetime
import matplotlib.pyplot as plt

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.samplers import RandomSampler
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report, precision_recall_curve, average_precision_score

from Data_helper import LoadDataTrial

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# %%
# parse arguments
## general
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
arg_parser.add_argument('--max_timestamp', type=int, default=48*60*60,
                        help='Time series of at most # seconds are used. Default: 48 (hours).')

## training
arg_parser.add_argument('--trainORvalidation', default='Validation', choices=['Train', 'Validation'])
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
        '--trainORvalidation', 'Validation'
    ])
else:
      ARGS = arg_parser.parse_args()

print('Arguments:', ARGS)

# %%
def plot_cm(label_info, y_validation, y_pred, output_path):
    # plot the confusion matrix
    C=confusion_matrix(y_validation, y_pred)
    fig, ax = plt.subplots(figsize = (4, 4))
    ax.matshow(C, cmap = plt.cm.Blues, alpha = 0.6)
    for n in range(C.shape[0]):
        for m in range(C.shape[1]):
            ax.text(x = m, y = n,
                s = C[n, m], 
                va = 'center', 
                ha = 'center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(output_path, "confusion_matrix(validation)({}).png".format(label_info)),dpi=600)
    plt.show()
    plt.close()

def save_report(label_info, y_validation, y_pred, output_path):
    report = classification_report(y_validation, y_pred,digits=5, output_dict = True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(os.path.join(output_path, 'classification_reuslts(validation)({}).csv'.format(label_info)))

def plot_roc(label_info, ydim, y_validation, probas, output_path):
    '''
    label_info: The label information to tag the figure name
    ydim: The dim of y, which determines the way of plot
    y_validation: The y value of validation dataset
    probas: The probability of model results
    '''
    # Compute ROC curve and area the curve
    if ydim == 2:
        fpr, tpr, thresholds = roc_curve(y_validation, probas[:, 1])
        roc_auc = auc(fpr, tpr)

        # save the data
        roc_value = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        np.save(os.path.join(output_path, 'ROC({}).npy'.format(label_info)), roc_value)

        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck', lw=1)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {}'.format(label_info))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_path, "ROC(validation)({}).png".format(label_info)),dpi=600)
        plt.show()
        plt.close()
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for m in range(ydim):
            fpr[m], tpr[m], _ = roc_curve(y_validation[:, m], probas[:, m])
            roc_auc[m] = auc(fpr[m], tpr[m])
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[n] for n in range(ydim)]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(ydim):
            mean_tpr += interp(all_fpr, fpr[k], tpr[k])
        # Finally average it and compute AUC
        mean_tpr /= ydim
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # save the data
        roc_value = {'fpr': fpr["macro"], 'tpr': tpr["macro"], 'roc_auc': roc_auc["macro"]}
        np.save(os.path.join(output_path, 'ROC({}).npy'.format(label_info)), roc_value)

        plt.plot(fpr["macro"], tpr["macro"],
                label='ROC (area = %0.2f)' % (roc_auc["macro"]),
                lw=1)

        plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6),label='Luck', lw=1)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {}'.format(label_info))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_path, "ROC(validation)({}).png".format(label_info)),dpi=600)
        plt.show()
        plt.close()

def plot_prc(label_info, ydim, y_validation, y_probas, y_pred, output_path):
    '''
    label_info: The label information to tag the figure name
    ydim: The dim of y, which determines the way of plot
    y_validation: The y value of validation dataset
    probas: The probability of model results
    '''
    # Compute PRC curve and area the curve
    if ydim == 2:

        precision, recall, _ = precision_recall_curve(y_validation, y_probas[:, 1])
        f1, auprc = f1_score(y_validation, y_pred), auc(recall, precision)
        # average_precision = average_precision_score(y_validation, y_probas[:, 1])   # equal to the auprc
        
        # save the data
        roc_value = {'precision': precision, 'recall': recall, 'auprc': auprc}
        np.save(os.path.join(output_path, 'PRC({}).npy'.format(label_info)), roc_value)

        # the way provided by zhihu
        # no_skill = len(y_validation[y_validation==1]) / len(y_validation)
        # plt.plot([0, 1], [no_skill, no_skill], color=(0.6, 0.6, 0.6), lw=1, linestyle='--', label='No Skill')
        # plt.plot(recall, precision, lw=1, marker='.', color="black", label='AUPRC (area = %0.2f)' % (auprc))
        plt.step(recall, precision, color='black', alpha=0.2, lw=1, where='post', label='AUPRC (area = %0.2f)' % (auprc))
        # the way provided by sklearn
        # plt.step(recall, precision, color='b', alpha=0.2, where='post')
        # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        # plt.plot([1,0], [1,0], '--', color=(0.6, 0.6, 0.6), label='Luck', lw=1)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('PRC of {}'.format(label_info))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_path, "PRC(validation)({}).png".format(label_info)),dpi=600)
        plt.show()
        plt.close()
    else:
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(ydim):
            precision[i], recall[i], _ = precision_recall_curve(y_validation[:, i], y_probas[:, i])
            average_precision[i] = average_precision_score(y_validation[:, i], y_probas[:, i])
        
        precision["macro"], recall["macro"], _ = precision_recall_curve(y_validation.ravel(), y_probas.ravel())
        average_precision["macro"] = average_precision_score(y_validation, y_probas, average="macro")

        # save the data
        roc_value = {'precision': precision["macro"], 'recall': recall["macro"], 'auprc': average_precision["macro"]}
        np.save(os.path.join(output_path, 'PRC({}).npy'.format(label_info)), roc_value)

        # no_skill = len(y_validation[y_validation==1]) / len(y_validation)
        # plt.plot([0, 1], [no_skill, no_skill], color=(0.6, 0.6, 0.6), lw=1, linestyle='--', label='No Skill')
        # plt.plot(recall["macro"], precision["macro"], lw=1, marker='.', color="black", label='AUPRC (area = %0.2f)' % (average_precision["macro"]))
        plt.step(recall["macro"], precision["macro"], color='black', alpha=0.2, lw=1, where='post', label='AUPRC (area = %0.2f)' % (average_precision["macro"]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('PRC of {}'.format(label_info))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_path, "PRC(validation)({}).png".format(label_info)),dpi=600)
        plt.show()
        plt.close()
    
# %%
# define the label, -1 represent the multi-class classification
model_path = os.path.join(ARGS.working_path, 'model_tuning', 'phase_viii', '48hours', '20220604', 'RF_423dim_48hrs_Paper(optuna)')
if not os.path.exists(model_path):
    os.makedirs(model_path)
output_path = os.path.join(ARGS.working_path, 'output', 'phase_viii', '48hours', '20220604', 'RF_423dim_48hrs_Paper(optuna)')
if not os.path.exists(output_path):
    os.makedirs(output_path)

T = time.strftime("%Y%m%d%H%M%S", time.localtime())
LABEL_DICT = {'-1':'flatten multi-classification', 
              '0': 'infectious and non-infectious', 
              '1':'bacterial, viral and others', 
              '2':'NIID and tumor', 
              '3': 'AD and AID', 
              '4':'HM and SM'}
# Load the data
dataset = LoadDataTrial(data_path=os.path.join('.', 'data', 'phase_viii', '48hours', 'processed', 'raw/data4hc_v20220401'), 
                        model_type=ARGS.model_type,
                        label_name=ARGS.label,
                        max_timestep=ARGS.max_timestep,
                        max_timestamp=ARGS.max_timestamp)
X_train, y_train, nclasses_train, folds_train, shapes_train = dataset.training_generator(ARGS.fold)
X_validation, y_validation, nclasses_validation,folds_validation, shapes_validation = dataset.validation_generator(ARGS.fold)
# check the data
assert  nclasses_train == nclasses_validation, 'The classes of train is not equal to the classes of valiadtion'

# %%
print("[Info]: The RF hyperparameter tuning of label {0} is processing".format(LABEL_DICT[str(ARGS.label)]))

def objective(trial):
    model = RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators', 10, 250),
                                    max_depth=int(trial.suggest_loguniform('max_depth', 4, 25)),
                                    bootstrap =trial.suggest_categorical('bootstrap',['True','False']),
                                    max_features=trial.suggest_categorical('max_features', ['auto', 'sqrt','log2']),
                                    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy']),
                                    min_samples_split = trial.suggest_int('min_samples_split', 2, 20, 1),
                                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20, 1),
                                    random_state=1).fit(X_train, y_train)

    # auc = roc_auc_score(y_validation, model.predict_proba(X_validation)[:,1])

    auc = roc_auc_score(y_validation, model.predict_proba(X_validation), multi_class='ovr')

    return auc

# %%
if __name__ == '__main__':
    
    if ARGS.trainORvalidation == 'Train':
        storage = optuna.storages.RDBStorage(
            url='postgresql://postgres:postgres@10.12.45.53:5432/postgres',
            engine_kwargs={
                'pool_size': 20,
                'max_overflow': 0
                }
        )
        study = optuna.create_study(direction='maximize', 
                                    storage=storage,
                                    sampler= optuna.samplers.TPESampler(),
                                    study_name='{0}({1})_{2}'.format(ARGS.model_type, LABEL_DICT[str(ARGS.label)], T))
        study.optimize(objective, n_trials=100)
        
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # save the visualization plot
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(output_path, "optimization_history({}).png".format(LABEL_DICT[str(ARGS.label)])))

        fig = plot_param_importances(study)
        fig.write_image(os.path.join(output_path, "param_importances({}).png".format(LABEL_DICT[str(ARGS.label)])))

        fig = plot_slice(study)
        fig.write_image(os.path.join(output_path, "plot_slice({}).png".format(LABEL_DICT[str(ARGS.label)])))

        fig = plot_intermediate_values(study)
        fig.write_image(os.path.join(output_path, "plot_intermediate_values({}).png".format(LABEL_DICT[str(ARGS.label)])))

        # Get the optimal hyperparameters and save
        best_hps_dict = {'best_hps': trial.params}
        np.save(os.path.join(model_path, 'BestHPs({}).npy'.format(LABEL_DICT[str(ARGS.label)])), best_hps_dict)

        print("The training process finished!")
    else:
        # get the hyperparameter under optuna
        hps = np.load(os.path.join(model_path, 'BestHPs({}).npy'.format(LABEL_DICT[str(ARGS.label)])), allow_pickle=True).tolist()['best_hps']
        n_estimators, max_depth, bootstrap, max_features, criterion,min_samples_split, min_samples_leaf = hps['n_estimators'], hps['max_depth'], hps['bootstrap'], hps['max_features'], hps['criterion'], hps['min_samples_split'], hps['min_samples_leaf']

        if ARGS.label == -1:
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           criterion = criterion,
                                           min_samples_split = min_samples_split,
                                           min_samples_leaf = min_samples_leaf,
                                           random_state=1).fit(X_train, y_train)

            # save the model
            model_file = os.path.join(model_path, 'RF({}).pkl'.format(LABEL_DICT[str(ARGS.label)]))
            joblib.dump(model, model_file)

            # predict the validation dataset
            y_pred = model.predict(X_validation)
            probas_ = model.predict_proba(X_validation)
            y_validation_one_hot = label_binarize(y_validation, classes=[0,1,2,3,4,5,6])
            
            # plot the ROC curve
            plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, probas_, output_path)
            plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, probas_, y_pred, output_path)
            # plot the confusion matrix
            plot_cm(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
            # save the validation dataset results
            save_report(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
        elif ARGS.label == 0:
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           criterion = criterion,
                                           min_samples_split = min_samples_split,
                                           min_samples_leaf = min_samples_leaf,
                                           random_state=1).fit(X_train, y_train)

            # save the model
            model_file = os.path.join(model_path, 'RF({}).pkl'.format(LABEL_DICT[str(ARGS.label)]))
            joblib.dump(model, model_file)

            # predict the validation dataset
            y_pred = model.predict(X_validation)
            probas_ = model.predict_proba(X_validation)

            # plot the ROC curve
            plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, output_path)
            plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, y_pred, output_path)
            # plot the confusion matrix
            plot_cm(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
            # save the validation dataset results
            save_report(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
        elif ARGS.label == 1:
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           criterion = criterion,
                                           min_samples_split = min_samples_split,
                                           min_samples_leaf = min_samples_leaf,
                                           random_state=1).fit(X_train, y_train)

            # save the model
            model_file = os.path.join(model_path, 'RF({}).pkl'.format(LABEL_DICT[str(ARGS.label)]))
            joblib.dump(model, model_file)

            # predict the validation dataset
            y_pred = model.predict(X_validation)
            probas_ = model.predict_proba(X_validation)
            y_validation_one_hot = label_binarize(y_validation, classes=[0,1,2])

            # plot the ROC curve
            plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, probas_, output_path)
            plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, probas_, y_pred, output_path)
            # plot the confusion matrix
            plot_cm(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
            # save the validation dataset results
            save_report(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
        elif ARGS.label == 2:
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           criterion = criterion,
                                           min_samples_split = min_samples_split,
                                           min_samples_leaf = min_samples_leaf,
                                           random_state=1).fit(X_train, y_train)

            # save the model
            model_file = os.path.join(model_path, 'RF({}).pkl'.format(LABEL_DICT[str(ARGS.label)]))
            joblib.dump(model, model_file)

            # predict the validation dataset
            y_pred = model.predict(X_validation)
            probas_ = model.predict_proba(X_validation)

            # plot the ROC curve
            plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, output_path)
            plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, y_pred, output_path)
            # plot the confusion matrix
            plot_cm(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
            # save the validation dataset results
            save_report(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
        elif ARGS.label == 3:
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           criterion = criterion,
                                           min_samples_split = min_samples_split,
                                           min_samples_leaf = min_samples_leaf,
                                           random_state=1).fit(X_train, y_train)

            # save the model
            model_file = os.path.join(model_path, 'RF({}).pkl'.format(LABEL_DICT[str(ARGS.label)]))
            joblib.dump(model, model_file)

            # predict the validation dataset
            y_pred = model.predict(X_validation)
            probas_ = model.predict_proba(X_validation)

            # plot the ROC curve
            plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, output_path)
            plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, y_pred, output_path)
            # plot the confusion matrix
            plot_cm(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
            # save the validation dataset results
            save_report(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           criterion = criterion,
                                           min_samples_split = min_samples_split,
                                           min_samples_leaf = min_samples_leaf,
                                           random_state=1).fit(X_train, y_train)

            # save the model
            model_file = os.path.join(model_path, 'RF({}).pkl'.format(LABEL_DICT[str(ARGS.label)]))
            joblib.dump(model, model_file)
            
            # predict the validation dataset
            y_pred = model.predict(X_validation)
            probas_ = model.predict_proba(X_validation)

            # plot the ROC curve
            plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, output_path)
            plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation, probas_, y_pred, output_path)
            # plot the confusion matrix
            plot_cm(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
            # save the validation dataset results
            save_report(LABEL_DICT[str(ARGS.label)], y_validation, y_pred, output_path)
        print("The validation process finished!")