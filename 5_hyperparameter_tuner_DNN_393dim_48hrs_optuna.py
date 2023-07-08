# %%
from __future__ import absolute_import, division, print_function

import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.backend import clear_session
from keras import regularizers 
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Activation, BatchNormalization, InputLayer
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from focal_loss import SparseCategoricalFocalLoss

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

from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, auc, classification_report, precision_recall_curve, average_precision_score

from Data_helper import LoadDataTrial

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
arg_parser.add_argument('--label', default=1, type=int, choices=[-1,0,1,2,3,4],
                         help='the label type')

## model
arg_parser.add_argument('--model_type', default='DNN', choices=['LR', 'RF', 'SVM', 'DNN', 'DNN_EE', 'GRUD', 'HMM_EF', 'HMM_LF'])
arg_parser.add_argument('--max_timestep', type=int, default=200, 
                        help='Time series of at most # time steps are used. Default: 200.')
arg_parser.add_argument('--max_timestamp', type=int, default=48*60*60,
                        help='Time series of at most # seconds are used. Default: 48 (hours).')

## training
arg_parser.add_argument('--trainORvalidation', default='Train', choices=['Train', 'Validation'])
arg_parser.add_argument('--epochs', type=int, default=100)
arg_parser.add_argument('--early_stopping_patience', type=int, default=10)
arg_parser.add_argument('--batch_size', type=int, default=32)

# set the actual arguments if running in notebook
if not (__name__ == '__main__' and '__file__' in globals()):
    ARGS = arg_parser.parse_args([
        '--model_type', 'DNN',
        '--dataset_name', 'phase_viii',
        '--fold', '0',
        '--epochs', '100',
        '--trainORvalidation', 'Train'
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
        # fpr, tpr, thresholds = roc_curve(y_validation, probas[:, 0], pos_label=1)  #for BCE
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
        
        # precision, recall, _ = precision_recall_curve(y_validation, y_probas[:, 0]) # for loss=BCE
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
model_path = os.path.join(ARGS.working_path, 'model_tuning', 'phase_viii', '48hours', '20220604', 'DNN_393dim_48hrs_Paper(optuna)')
if not os.path.exists(model_path):
    os.makedirs(model_path)
output_path = os.path.join(ARGS.working_path, 'output', 'phase_viii', '48hours', '20220604', 'DNN_393dim_48hrs_Paper(optuna)')
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
X_train, y_train_1d, nclasses_train, folds_train, shapes_train = dataset.training_generator(ARGS.fold)
X_validation, y_validation_1d, nclasses_validation,folds_validation, shapes_validation = dataset.validation_generator(ARGS.fold)
#[OPTINAL!] this is adopted when loss = categorical_crossentropy
y_train = to_categorical(y_train_1d, num_classes=nclasses_train)           
y_validation = to_categorical(y_validation_1d, num_classes=nclasses_train)

# check the data
assert  nclasses_train == nclasses_validation, 'The classes of train is not equal to the classes of valiadtion'
print("[Info]: The DNN hyperparameter tuning of label {0} is processing".format(LABEL_DICT[str(ARGS.label)]))
print("[Info]: The shape of DNN Input is {0}".format(shapes_train))

class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=None):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
 
    def on_batch_begin(self, batch, logs={}):
        pass
 
    def on_batch_end(self, batch, logs={}):
        pass
 
    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')
 
    def on_train_end(self, logs={}):
        pass
 
    def on_epoch_begin(self, epoch, logs={}):
        pass
 
    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val'] = float('-inf')
        if (self.validation_data):
            # logs['roc_auc_val'] = roc_auc_score(np.argmax(self.validation_data[1],axis=1),
            #                                     self.model.predict(self.validation_data[0],
            #                                                    batch_size=self.predict_batch_size)[:,1])
            logs['roc_auc_val'] = roc_auc_score(self.validation_data[1],
                                                self.model.predict(self.validation_data[0],
                                                               batch_size=self.predict_batch_size), 
                                                multi_class='ovo', 
                                                average='macro')
        print(' - ---------------------------------------------------------------------------val_auc: %.4f' % (logs['roc_auc_val']))

def build_model(trial):

    model = Sequential()
    model.add(InputLayer(input_shape=(X_train.shape[1],)))

    n_layers = trial.suggest_int("n_layers", 1,6, step=1)
    for i in range(n_layers):
        model.add(Dense(units=trial.suggest_int("n_units_{}".format(i), 32, 512, step=8),
                        activation=trial.suggest_categorical("activation_{}".format(i), ['relu', 'tanh', 'sigmoid']),
                        kernel_regularizer = regularizers.l1_l2(
                        l1=trial.suggest_categorical("l1_{}".format(i), [0.0001, 0.0005, 0.001,0.005, 0.01,0.05]),
                        l2=trial.suggest_categorical("l2_{}".format(i), [0.0001, 0.0005, 0.001,0.005, 0.01,0.05]))
                        ))
        dropout_choice = trial.suggest_categorical("dropout_{}".format(i), [True, False])
        if dropout_choice == True:
            model.add(Dropout(rate=trial.suggest_float("dropout_val_{}".format(i), 0.0, 0.5, step=0.05)))

    model.add(Dense(nclasses_train, activation=None))

    batchnorm_choice = trial.suggest_categorical("batchnorm", [True, False])
    if batchnorm_choice == True:
        model.add(BatchNormalization())
    model.add(Activation('softmax'))

    kwargs = {}
    optimizer_selected = trial.suggest_categorical("optimizer", ["RMSprop", "Adam", "SGD", "Adadelta", "Adagrad"])
    if optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float("rmsprop_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float("sgd_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adadelta":
        kwargs["learning_rate"] = trial.suggest_float("Adadelta_opt_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adagrad":
        kwargs["learning_rate"] = trial.suggest_float("Adagrad_learning_rate", 1e-5, 1e-1, log=True)

    optimizer = getattr(keras.optimizers, optimizer_selected)(**kwargs)
    model.compile(loss='categorical_crossentropy',  # binary_crossentropy sparse_categorical_crossentropy categorical_crossentropy
                    optimizer=optimizer,
                    metrics=['accuracy'])

    return model

class Objective():
    def __init__(self):
        self.best_model = None
        self._model = None

    def __call__(self, trial):
        # create the model
        model = build_model(trial)

        model.fit(X_train,
                y_train,
                validation_data=(X_validation, y_validation),
                shuffle=False,
                batch_size=trial.suggest_categorical("batchsize", [8, 16, 32, 64, 128]),
                epochs=ARGS.epochs,
                verbose=2,
                callbacks=[RocAucMetricCallback(),
                           TFKerasPruningCallback(trial, "roc_auc_val"),
                           EarlyStopping(monitor='roc_auc_val', mode='max', patience=10, restore_best_weights=True)]
                )
        self._model = model
        # Evaluate the model accuracy on the validation set.
        # auc = roc_auc_score(y_validation_1d, model.predict(X_validation)[:,1])
        auc = roc_auc_score(y_validation, model.predict(X_validation), multi_class='ovo', average='macro')
        return auc
    
    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_model = self._model

# %%
if __name__ == '__main__':

    objective = Objective()
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
                                pruner=optuna.pruners.MedianPruner(),
                                study_name='{0}({1})_{2}'.format(ARGS.model_type, LABEL_DICT[str(ARGS.label)], T))
    study.optimize(objective, 
                    n_trials=100, 
                    callbacks=[objective.callback])
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Number of finished trials: {}".format(len(study.trials)))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
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

    # save the model
    best_model = objective.best_model
    best_model.save(os.path.join(model_path, 'DNN({}).h5'.format(LABEL_DICT[str(ARGS.label)])))
    
    # predict the validation dataset
    y_pred_proba = best_model.predict(X_validation)
    y_pred = np.argmax(y_pred_proba, axis=1)
    # y_pred = best_model.predict_classes(X_validation) # when BCE
    
    print("The training process finished!")
    # plot the ROC curve
    if ARGS.label == -1:
        y_validation_one_hot = label_binarize(y_validation_1d, classes=[0,1,2,3,4,5,6])
        plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, y_pred_proba, output_path)
        plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, y_pred_proba, y_pred, output_path)
    elif ARGS.label == 1:
        y_validation_one_hot = label_binarize(y_validation_1d, classes=[0,1,2])
        plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, y_pred_proba, output_path)
        plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_one_hot, y_pred_proba, y_pred, output_path)
    else:
        plot_roc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_1d, y_pred_proba, output_path)
        plot_prc(LABEL_DICT[str(ARGS.label)], nclasses_train, y_validation_1d, y_pred_proba, y_pred, output_path)
    
    # plot the confusion matrix
    plot_cm(LABEL_DICT[str(ARGS.label)], y_validation_1d, y_pred, output_path)
    # save the validation dataset results
    save_report(LABEL_DICT[str(ARGS.label)], y_validation_1d, y_pred, output_path)

    # retrain the model based on train and validation dataset
    plot_model(best_model, show_shapes=True, to_file=os.path.join(model_path, 'model_structure({}).png'.format(LABEL_DICT[str(ARGS.label)])))