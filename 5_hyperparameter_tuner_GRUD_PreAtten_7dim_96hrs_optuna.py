# %%
from __future__ import absolute_import, division, print_function

import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from unittest import mock

import keras
from keras import backend as K
from keras.backend import clear_session
from keras import regularizers 
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation, BatchNormalization, InputLayer, Embedding, Reshape, Concatenate, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical

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
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report, precision_recall_curve, average_precision_score

from Data_helper import LoadDataTrial
from utils.layers import ExternalMasking
from utils.grud_layers import GRUD, Bidirectional_for_GRUD
from BuildModel import LoadModel
from utils.pre_attention import attention_3d_block_spatial as PreAttentionSpatial

# parse arguments
## general
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--working_path', default='.')

## data
arg_parser.add_argument('--dataset_name', default='phase_viii', 
                         help='The data files should be saved in [working_path]/data/[dataset_name] directory.')
arg_parser.add_argument('--fold', type=int, default=0, 
                         help='the fold data taken to use, there are 5 folds or 10 folds')
arg_parser.add_argument('--label', default=4, type=int, choices=[-1,0,1,2,3,4],
                         help='the label type')

## model
arg_parser.add_argument('--model_type', default='GRUD', choices=['LR', 'RF', 'SVM', 'DNN', 'DNN_EE', 'GRUD', 'HMM_EF', 'HMM_LF'])
arg_parser.add_argument('--max_timestep', type=int, default=200, 
                        help='Time series of at most # time steps are used. Default: 200.')
arg_parser.add_argument('--max_timestamp', type=int, default=96*60*60,choices=[48*60*60, 72*60*60, 96*60*60, 120*60*60],
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
        '--model_type', 'GRUD',
        '--dataset_name', 'phase_viii',
        '--fold', '0',
        '--epochs', '100',
        '--trainORvalidation', 'Train'
    ])
else:
      ARGS = arg_parser.parse_args()

print('Arguments:', ARGS)
# %%
# define the label, -1 represent the multi-class classification
model_path = os.path.join(ARGS.working_path, 'model_tuning', 'phase_viii', '96hours', '20220604', 'GRUD_PreAttenSpatial_7dim_96hrs_Paper(optuna)')
if not os.path.exists(model_path):
    os.makedirs(model_path)
output_path = os.path.join(ARGS.working_path, 'output', 'phase_viii', '96hours', '20220604', 'GRUD_PreAttenSpatial_7dim_96hrs_Paper(optuna)')
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
dataset = LoadDataTrial(data_path=os.path.join('.', 'data', 'phase_viii', '96hours', 'processed', 'raw/data4hc_v20220401'), 
                        model_type=ARGS.model_type,
                        label_name=ARGS.label,
                        max_timestep=ARGS.max_timestep,
                        max_timestamp=ARGS.max_timestamp)
X_train, y_train_1d, nclasses_train, folds_train, shapes_train = dataset.training_generator(ARGS.fold)
X_validation, y_validation_1d, nclasses_validation,folds_validation, shapes_validation = dataset.validation_generator(ARGS.fold)
y_train = to_categorical(y_train_1d, num_classes=nclasses_train)           
y_validation = to_categorical(y_validation_1d, num_classes=nclasses_train)

# check the data
assert  nclasses_train == nclasses_validation, 'The classes of train is not equal to the classes of valiadtion'

print("[Info]: The GRUD hyperparameter tuning of label {0} is processing".format(LABEL_DICT[str(ARGS.label)]))
print("[Info] The max time steps of time series data is {0}!".format(dataset.max_timestep))

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
        x_val = [self.validation_data[0], self.validation_data[1], self.validation_data[2]]
        y_val = np.argmax(self.validation_data[3], axis=1)
        # with mock.patch("sklearn.utils.validation._assert_all_finite"):
        logs['roc_auc_val'] = roc_auc_score(y_val,
                                            self.model.predict(x_val, batch_size=self.predict_batch_size)[:,1])
        # logs['roc_auc_val'] = roc_auc_score(y_val,
        #                                     self.model.predict(x_val, batch_size=self.predict_batch_size),
        #                                     multi_class='ovo', 
        #                                     average='macro')
        print(' - ---------------------------------------------------------------------------val_auc: %.4f' % (logs['roc_auc_val']))

def build_model(trial):

    K.clear_session()
     
    input_x = Input(shape=(dataset.max_timestep, dataset.input_dim_t))
    input_m = Input(shape=(dataset.max_timestep, dataset.input_dim_t))
    input_s = Input(shape=(dataset.max_timestep, 1))
    input_list = [input_x, input_m, input_s]

    input_x = PreAttentionSpatial(input_x, dataset.input_dim_t)

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=trial.suggest_int("units", 24, 156, step=4),
                      return_sequences=False,
                      activation=trial.suggest_categorical("activation", ['relu', 'tanh', 'sigmoid']),
                      dropout=trial.suggest_float("dropout", 0.0, 0.7, step=0.05),
                      recurrent_dropout=trial.suggest_float("recurrent_dropout", 0.0, 0.7, step=0.05)
                      )
    grud_model = grud_layer([input_x, input_m, input_s])
    dropout_choice_grud = trial.suggest_categorical("dropout_grud", [True, False])
    if dropout_choice_grud == True:
        grud_model = Dropout(rate=trial.suggest_float("dropout_grud_val", 0.0, 0.7, step=0.05))(grud_model)

    n_layers = trial.suggest_categorical("n_layers", [True, False])
    if n_layers == True:
        grud_model = Dense(units=trial.suggest_int("units_fc", 24, 256, step=4),
                             activation=trial.suggest_categorical("activation_fc", ['relu', 'tanh']),
                             kernel_regularizer = regularizers.l1_l2(
                             l1=trial.suggest_categorical("l1_fc", [0.0001, 0.0005, 0.001,0.005, 0.01,0.05]),
                             l2=trial.suggest_categorical("l2_fc", [0.0001, 0.0005, 0.001,0.005, 0.01,0.05]))
                             )(grud_model)
        dropout_choice = trial.suggest_categorical("dropout_fc", [True, False])
        if dropout_choice == True:
            grud_model = Dropout(rate=trial.suggest_float("dropout_fc_val", 0.0, 0.7, step=0.05))(grud_model)

    grud_model = Dense(nclasses_train, activation=None)(grud_model)

    batchnorm_choice = trial.suggest_categorical("batchnorm", [True, False])
    if batchnorm_choice == True:
        grud_model = BatchNormalization()(grud_model)
    grud_model = Activation('softmax')(grud_model)
    output_list = [grud_model]

    model = Model(inputs=input_list, outputs=output_list)

    kwargs = {}
    optimizer_selected = trial.suggest_categorical("optimizer", ["RMSprop", "Adam", "Adagrad"])
    if optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float("rmsprop_learning_rate", 1e-5, 1e-1, log=True)
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
    elif optimizer_selected == "Adagrad":
        kwargs["learning_rate"] = trial.suggest_float("Adagrad_learning_rate", 1e-5, 1e-1, log=True)

    optimizer = getattr(keras.optimizers, optimizer_selected)(**kwargs)
    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy']) 
    return model

class Objective():
    def __init__(self):
        self._model = None

    def __call__(self, trial):
        # create the model
        model = build_model(trial)
        model.fit(X_train,
                    y_train,
                    validation_data=(X_validation, y_validation),
                    shuffle=True,
                    batch_size=trial.suggest_categorical("batchsize", [8, 16, 24, 32, 64]),
                    epochs=100,
                    verbose=2,
                    class_weight="auto",
                    callbacks=[RocAucMetricCallback(),
                        # TFKerasPruningCallback(trial, "roc_auc_val"), # val_accuracy roc_auc_val
                        EarlyStopping(monitor='roc_auc_val', mode='max', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='roc_auc_val',
                                                factor=0.1,
                                                patience=5,
                                                verbose=1,
                                                mode='max',
                                                epsilon=0.0001,
                                                cooldown=0,
                                                min_lr=0)]
                    )

        self._model = model
        # Evaluate the model accuracy on the validation set.
        # score = model.evaluate(X_validation, y_validation, verbose=0)
        # return score[1]
        auc = roc_auc_score(y_validation_1d, model.predict(X_validation)[:,1])
        #   
        return auc

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self._model.save(os.path.join(model_path, 'GRUD({0})_trial#{1}.h5'.format(LABEL_DICT[str(ARGS.label)], trial.number))) 

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
    print("[Info] The sampler of optuna is {0}!".format('TPESampler'))
    # study = optuna.create_study(direction='maximize', 
    #                             storage=storage,
    #                             sampler= optuna.samplers.TPESampler(),
    #                             # pruner=optuna.pruners.MedianPruner(),
    #                             study_name='{0}({1})_{2}'.format(ARGS.model_type, LABEL_DICT[str(ARGS.label)], T))
    study = optuna.create_study(direction='maximize', 
                                storage=storage,
                                sampler= optuna.samplers.TPESampler(),
                                # pruner=optuna.pruners.MedianPruner(),
                                load_if_exists=True,
                                study_name='GRUD(HM and SM)_20220625213318')
    study.optimize(objective, 
                   n_trials=1,
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
