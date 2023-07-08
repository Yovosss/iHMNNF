# %%
from __future__ import absolute_import, division, print_function

import matplotlib as plt
import argparse
from datetime import datetime
import numpy as np
import os

from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
from sklearn.metrics import roc_auc_score

from Data_helper import LoadDataMM
from BuildModel import BuildModel_HMM, LoadModel_HMM
from utils.callbacks import ModelCheckpointwithBestWeights


# %%
# set GPU usage for tensorflow backend
if K.backend() == 'tensorflow':
    import tensorflow as tf
    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = .1
    config.gpu_options.allow_growth = True
    # K.set_session(tf.Session(config=config))
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# %%
# parse arguments
## general
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--working_path', default='.')

## data
arg_parser.add_argument('--dataset_name', default='phase_iii', 
                         help='The data files should be saved in [working_path]/data/[dataset_name] directory.')
arg_parser.add_argument('--label', default=0, type=int, choices=[0, 1, 2, 3, 4],
                         help='[hy-0]: for L1_model, including infec- and no-infec- data;\
                               [hy-1]: for L21_model, including bac-, vir- and fun-;\
                               [hy-2]: for L22_model, including NIID and neo-;\
                               [hy-3]: for L31_model, including aid- and afd-;\
                               [hy-4]: for L32_model, including hm- and sm-')
arg_parser.add_argument('--nClasses', type=int, default=1, help='The classes of each label')
arg_parser.add_argument('--max_timesteps', type=int, default=200, 
                        help='Time series of at most # time steps are used. Default: 200.')
arg_parser.add_argument('--max_timestamp', type=int, default=48*60*60,
                        help='Time series of at most # seconds are used. Default: 48 (hours).')

## model
arg_parser.add_argument('--model', default='L1_model', choices=['L1_model', 'L21_model', 'L22_model', 'L31_model', 'L32_model'])
arg_parser.add_argument('--recurrent_dim', type=lambda x: x and [int(xx) for xx in x.split(',')] or [], default='64')
arg_parser.add_argument('--hidden_dim', type=lambda x: x and [int(xx) for xx in x.split(',')] or [], default='64')
arg_parser.add_argument('--use_bidirectional_rnn', default=False)

## training
arg_parser.add_argument('--epochs', type=int, default=5)
arg_parser.add_argument('--early_stopping_patience', type=int, default=10)
arg_parser.add_argument('--batch_size', type=int, default=64)

## set the actual arguments if running in notebook
# if not (__name__ == '__main__' and '__file__' in globals()):
#     ARGS = arg_parser.parse_args([
#         'phase_iii',
#         '0',
#         '2',
#         '--model', 'L1_model',
#         '--hidden_dim', '4',
#         '--epochs', '3'
#     ])
# else:
ARGS = arg_parser.parse_args()

print('Arguments:', ARGS)

# %%
# get dataset
dataset = LoadDataMM(
    data_path=os.path.join(ARGS.working_path, 'data', ARGS.dataset_name, 'processed', 'data_model_v20211207'), 
    label_name=ARGS.label,
    nclasses = ARGS.nClasses,
    max_steps=ARGS.max_timesteps,
    max_timestamp=ARGS.max_timestamp
)

# %%
# k-fold cross-validation
pred_y_list_all = []
auc_score_list_all = []

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
print('Timestamp:{}'.format(timestamp))


for i_fold in range(dataset.folds):
    print('{}-th fold...'.format(i_fold))
    # Load or train the model.
    model = BuildModel_HMM(input_dim_t=dataset.input_dim_t, # the shape of temporal input
                           input_dim_s_val = dataset.input_dim_s_val,
                           embedding_size = dataset.embedding_size,
                           recurrent_dim=ARGS.recurrent_dim,   # the number of unit of hidden layer
                           output_dim=dataset.output_dim,   # here is 1
                           output_activation=dataset.output_activation,
                           use_bidirectional_rnn=ARGS.use_bidirectional_rnn
                           )
    if i_fold == 0:
        model.summary()
    model_path = os.path.join(ARGS.working_path, 'model_dl', 'HMM', timestamp + '_' + str(i_fold))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    log_path = os.path.join(ARGS.working_path, 'log', 'tb_logs', timestamp + '_' + str(i_fold))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    result_path = os.path.join(ARGS.working_path, 'output', 'phase_iii', timestamp)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model.compile(optimizer='RMSprop', loss=dataset.loss_function)
    plot_model(model, show_shapes=True, to_file=os.path.join(model_path, 'HMM.png'))
    model.fit_generator(
        generator=dataset.training_generator(i_fold, batch_size=ARGS.batch_size),
        steps_per_epoch=dataset.training_steps(i_fold, batch_size=ARGS.batch_size),
        epochs=ARGS.epochs,
        verbose=1,
        validation_data=dataset.validation_generator(i_fold, batch_size=ARGS.batch_size),
        validation_steps=dataset.validation_steps(i_fold, batch_size=ARGS.batch_size),
        callbacks=[
            EarlyStopping(patience=ARGS.early_stopping_patience),
            ModelCheckpointwithBestWeights(file_dir=model_path),
            TensorBoard(log_dir=log_path)
        ]
    )
    model.save(os.path.join(model_path, 'model.h5'))
    
    # evaluate the model
    true_y_list = [
        dataset.training_y(i_fold), dataset.validation_y(i_fold), dataset.testing_y(i_fold)
    ]
    pred_y_list = [
        model.predict_generator(dataset.training_generator_x(i_fold, batch_size=ARGS.batch_size),
                                steps=dataset.training_steps(i_fold, batch_size=ARGS.batch_size)),
        model.predict_generator(dataset.validation_generator_x(i_fold, batch_size=ARGS.batch_size),
                                steps=dataset.validation_steps(i_fold, batch_size=ARGS.batch_size)),
        model.predict_generator(dataset.testing_generator_x(i_fold, batch_size=ARGS.batch_size),
                                steps=dataset.testing_steps(i_fold, batch_size=ARGS.batch_size)),
    ]
    auc_score_list = [roc_auc_score(ty, py) for ty, py in zip(true_y_list, pred_y_list)] # [3, n_task]
    print('AUC score of this fold: {}'.format(auc_score_list))
    pred_y_list_all.append(pred_y_list)
    auc_score_list_all.append(auc_score_list)

print('Finished!', '='*20)
auc_score_list_all = np.stack(auc_score_list_all, axis=0)
print('Mean AUC score: {}; Std AUC score: {}'.format(
    np.mean(auc_score_list_all, axis=0),
    np.std(auc_score_list_all, axis=0)))

np.savez_compressed(os.path.join(result_path, 'predictions.npz'),
                    pred_y_list_all=pred_y_list_all)
np.savez_compressed(os.path.join(result_path, 'auroc_score.npz'),
                    auc_score_list_all=auc_score_list_all)


