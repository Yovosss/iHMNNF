from __future__ import absolute_import, division, print_function

import os
import keras
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import keras_tuner as kt

from keras import backend as K
from keras import regularizers 
from keras.regularizers import l2
from keras.utils import plot_model
from keras.layers.recurrent import GRU
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model, load_model
from keras.utils.generic_utils import custom_object_scope
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import custom_object_scope
from keras.layers import Dense, Input, Flatten, Reshape, Embedding, Concatenate, Dropout, Bidirectional, Activation, Masking, Lambda
from keras import optimizers

from utils.layers import ExternalMasking
from utils.grud_layers import Bidirectional_for_GRUD, GRUD


def BuildModel_DNN(Shape, nClasses, nLayers=3,Number_Node=100, dropout=0.5, use_batchnorm=False):
    '''
    buildModel_DNN(nFeatures, nClasses, nLayers=3,Numberof_NOde=100, dropout=0.5)
    Build Deep neural networks (Multi-layer perceptron) Model
    Shape is input feature space
    nClasses is number of classes
    nLayers is number of hidden Layer
    Number_Node is number of unit in each hidden layer
    dropout is dropout value for solving overfitting problem
    '''
    model = Sequential()
    model.add(Dense(Number_Node, input_dim=Shape))  #, activity_regularizer = regularizers.l2(0.01) , kernel_regularizer = regularizers.l1_l2(l1=3e-4, l2=3e-4)
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(Number_Node, activation='relu'))
        model.add(Dropout(dropout))
    # model.add(Dense(nClasses, activation='softmax'))

    model.add(Dense(nClasses, activation=None))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(Activation('softmax'))

    optimizer = keras.optimizers.SGD(learning_rate=0.001)
    # optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    # optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def BuildModel_DNN_EE(embedding_size, input_size_val, nclasses, use_batchnorm=False):

    input_models = []
    output_embeddings = []

    input_val = Input(shape=(input_size_val,))
    embedding_val = Dense(input_size_val)(input_val)
    input_models.append(input_val)
    output_embeddings.append(embedding_val)

    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_models.append(input_shape)
        output_embeddings.append(output_embedding)

    output_model = Concatenate()(output_embeddings)
    output_model = Dense(592, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(592, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(296, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(296, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(148, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    # output_model = Dense(nclasses, activation='softmax')(output_model)

    output_model = Dense(nclasses, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation('softmax')(output_model)

    # define and conpile the model
    model = Model(inputs=input_models, outputs=output_model)
    # RMS = keras.optimizers.RMSprop(learning_rate=0.001)
    ADAM = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=ADAM,   # 'sgd'
                    metrics=['accuracy'])

    return model

def BuildModel_GRUD(input_dim_t, 
                    t_length, 
                    recurrent_dim, 
                    output_dim, 
                    output_activation, 
                    use_bidirectional_rnn=False, 
                    use_batchnorm=False, 
                    return_sequences=False, 
                    **kwargs):
    if return_sequences:
        input_x = Input(shape=(38, input_dim_t))
        input_m = Input(shape=(38, input_dim_t))
        input_s = Input(shape=(38, 1))
    else:
        input_x = Input(shape=(None, input_dim_t))
        input_m = Input(shape=(None, input_dim_t))
        input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=recurrent_dim[0],
                      return_sequences=return_sequences,
                      activation='sigmoid',
                      dropout=0.25,
                      recurrent_dropout=0.25,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    if return_sequences:
        grud_model = Lambda(lambda x: x, output_shape=lambda s:s)(grud_model)
        grud_model = Flatten()(grud_model)
        grud_model = Dropout(.25)(grud_model)
        grud_model = Dense(380, activation=output_activation)(grud_model)
        grud_model = Dropout(.25)(grud_model)
        grud_model = Dense(190, activation=output_activation)(grud_model)
        grud_model = Dropout(.25)(grud_model)
        grud_model = Dense(95, activation=output_activation)(grud_model)

    if use_batchnorm:
        grud_model = BatchNormalization()(grud_model)

    output_model = Dense(output_dim, activation=output_activation)(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    # RMS = keras.optimizers.RMSprop(learning_rate=0.001)
    ADAM = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',  # sparse_categorical_crossentropy binary_crossentropy
                    optimizer=ADAM, # RMSprop  sgd
                    metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file='./embeddings.png')

    return model

def BuildModel_HMM_EarlyFusion(input_dim_t, 
                               t_length, 
                               input_dim_s_val, 
                               embedding_size, 
                               recurrent_dim,
                               output_dim, 
                               output_activation,
                               use_bidirectional_rnn=False, 
                               use_batchnorm=True, 
                               return_sequences=False, 
                               **kwargs):
    # Input
    input_list = []
    embedding_list = []
    output_list = []

    ## The temporal variables
    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list += [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    # GRU layers
    grud_layer = GRUD(units=recurrent_dim[0],
                      return_sequences=return_sequences,  # True  False
                      activation='sigmoid',
                      dropout=0.25,
                      recurrent_dropout=0.25,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])
    # if return_sequences:
    #     grud_model = Flatten()(grud_model)
    grud_model = Dropout(.25)(grud_model)

    ## The numerical variables
    input_val = Input(shape=(input_dim_s_val,))
    embedding_val = Dense(input_dim_s_val)(input_val)
    input_list.append(input_val)
    embedding_list.append(embedding_val)

    ## The categorical variables
    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_list.append(input_shape)
        embedding_list.append(output_embedding)

    embedding_model = Concatenate()(embedding_list)
    embedding_model = Dense(592, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(embedding_model)
    embedding_model = Dropout(.25)(embedding_model)
    embedding_model = Dense(592, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(embedding_model)
    embedding_model = Dropout(.25)(embedding_model)
    embedding_model = Dense(592, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(embedding_model)
    embedding_model = Dropout(.25)(embedding_model)

    output_model = Concatenate()([grud_model, embedding_model])
    output_model = Dense(962, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(962, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(481, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(240, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(output_model)
    output_model = Dropout(.25)(output_model)
    output_model = Dense(output_dim, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation(output_activation)(output_model)
    
    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    ADAM = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',  # sparse_categorical_crossentropy binary_crossentropy
                    optimizer=ADAM, # RMSprop  sgd
                    metrics=['accuracy'])

    return model

def BuildModel_HMM_LateFuison(input_dim_t, 
                              t_length, 
                              input_dim_s_val, 
                              embedding_size, 
                              recurrent_dim,
                              output_dim, output_activation,
                              use_bidirectional_rnn=False, 
                              use_batchnorm=True, 
                              return_sequences=False, 
                              **kwargs):
    # Input
    input_list = []
    embedding_list = []
    output_list = []

    ## The temporal variables
    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list += [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    # GRU layers
    grud_layer = GRUD(units=recurrent_dim[0],
                      return_sequences=return_sequences,  # True  False
                      activation='sigmoid',
                      dropout=0.25,
                      recurrent_dropout=0.25,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])
    # grud_model = Lambda(lambda x: x, output_shape=lambda s:s)(grud_model)
    # GRU layers
    # grud_model = GRU(units=recurrent_dim[0],
    #                   return_sequences=False,  # True  False
    #                   activation='sigmoid',
    #                   dropout=0.25,
    #                   recurrent_dropout=0.25,
    #                   **kwargs
    #                  )(grud_model)
    # grud_model = Lambda(lambda x: x, output_shape=lambda s:s)(grud_model)
    # if return_sequences:
    #     grud_model = Flatten()(grud_model)
    grud_model = Dropout(.25)(grud_model)

    ## The numerical variables
    input_val = Input(shape=(input_dim_s_val,))
    embedding_val = Dense(input_dim_s_val)(input_val)
    input_list.append(input_val)
    embedding_list.append(embedding_val)

    ## The categorical variables
    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_list.append(input_shape)
        embedding_list.append(output_embedding)

    embedding_model = Concatenate()(embedding_list)
    embedding_model = Dense(592, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(embedding_model)   #,kernel_regularizer = regularizers.l1_l2(l1=3e-4, l2=3e-4)
    embedding_model = Dropout(.25)(embedding_model)
    embedding_model = Dense(296, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(embedding_model)
    embedding_model = Dropout(.25)(embedding_model)
    embedding_model = Dense(148, activation='relu', kernel_initializer="random_uniform", 
                            bias_initializer='random_uniform')(embedding_model)
    embedding_model = Dropout(.25)(embedding_model)

    output_model = Concatenate()([grud_model, embedding_model])
    output_model = Dense(output_dim, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation(output_activation)(output_model)
    
    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    # optimizer = keras.optimizers.SGD(learning_rate=0.001)
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    # optimizer = keras.optimizers.Adam(learning_rat    e=0.001)
    model.compile(loss='sparse_categorical_crossentropy',  # sparse_categorical_crossentropy binary_crossentropy
                    optimizer=optimizer, # RMSprop  sgd
                    metrics=['accuracy'])

    return model

def LoadModel(file_name):
    from utils import _get_scope_dict
    with custom_object_scope(_get_scope_dict()):
        model = load_model(file_name)
    return model

def BuildModel_GRUD_manual(input_dim_t, 
                            nClasses, 
                            use_bidirectional_rnn=False, 
                            use_batchnorm=True, 
                            return_sequences=False, 
                            **kwargs):

    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=64,
                      return_sequences=False,
                      activation='sigmoid', # sigmoid
                      dropout=0.25,  #0.15
                      recurrent_dropout=0.25, # 0.1
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    grud_model = Dense(nClasses, activation=None)(grud_model)
    grud_model = BatchNormalization()(grud_model)
    output_model = Activation('softmax')(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008),
                    metrics=['accuracy'])

    return model

def BestModel_GRUD_4(input_dim_t, 
                    t_length, 
                    nClasses, 
                    use_bidirectional_rnn=False, 
                    use_batchnorm=True, 
                    return_sequences=False, 
                    **kwargs):

    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=56,
                      return_sequences=False,
                      activation='relu',
                      dropout=0.15,
                      recurrent_dropout=0.4,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    grud_model = Dense(32, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    grud_model = Dense(32, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    grud_model = Dense(nClasses, activation=None)(grud_model)
    grud_model = BatchNormalization()(grud_model)
    output_model = Activation('softmax')(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008901867437371687),
                    metrics=['accuracy'])

    return model

def BestModel_GRUD_3(input_dim_t, 
                    t_length, 
                    nClasses, 
                    use_bidirectional_rnn=False, 
                    use_batchnorm=True, 
                    return_sequences=False, 
                    **kwargs):

    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=80,
                      return_sequences=False,
                      activation='tanh',
                      dropout=0.35,
                      recurrent_dropout=0.45,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    grud_model = Dense(160, activation='tanh', kernel_regularizer = regularizers.l1_l2(l1=0.005, l2=0.0005))(grud_model)
    grud_model = Dropout(0.45)(grud_model)
    grud_model = Dense(128, activation='tanh', kernel_regularizer = regularizers.l1_l2(l1=0.005, l2=0.0005))(grud_model)
    grud_model = Dropout(0.05)(grud_model)
    grud_model = Dense(192, activation='tanh', kernel_regularizer = regularizers.l1_l2(l1=0.005, l2=0.0005))(grud_model)
    grud_model = Dropout(0.0)(grud_model)
    grud_model = Dense(64, activation='tanh', kernel_regularizer = regularizers.l1_l2(l1=0.005, l2=0.0005))(grud_model)
    grud_model = Dense(nClasses, activation=None)(grud_model)
    grud_model = BatchNormalization()(grud_model)
    output_model = Activation('softmax')(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.00022149269189931809),
                    metrics=['accuracy'])

    return model

def BestModel_GRUD_2(input_dim_t, 
                    t_length, 
                    nClasses, 
                    use_bidirectional_rnn=False, 
                    use_batchnorm=True, 
                    return_sequences=False, 
                    **kwargs):

    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=64,
                      return_sequences=False,
                      activation='sigmoid', # sigmoid
                      dropout=0.25,  #0.15
                      recurrent_dropout=0.25, # 0.1
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    # grud_model = Dense(32, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    # grud_model = Dense(32, activation='sigmoid', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    # grud_model = Dense(32, activation='sigmoid', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    # grud_model = Dense(32, activation='sigmoid', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    grud_model = Dense(nClasses, activation=None)(grud_model)
    grud_model = BatchNormalization()(grud_model)
    output_model = Activation('softmax')(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008),
                    metrics=['accuracy'])

    return model

def BestModel_GRUD_1(input_dim_t, 
                    t_length, 
                    nClasses, 
                    use_bidirectional_rnn=False, 
                    use_batchnorm=True, 
                    return_sequences=False, 
                    **kwargs):

    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=72,
                      return_sequences=False,
                      activation='sigmoid',
                      dropout=0.3,
                      recurrent_dropout=0.1,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    grud_model = Dense(32, activation='sigmoid', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    output_model = Dense(nClasses, activation='softmax')(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.008633892527819412),
                    metrics=['accuracy'])

    return model

def BestModel_GRUD_0(input_dim_t, 
                    t_length, 
                    nClasses, 
                    use_bidirectional_rnn=False, 
                    use_batchnorm=True, 
                    return_sequences=False, 
                    **kwargs):

    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=96,
                      return_sequences=False,
                      activation='relu',
                      dropout=0.05,
                      recurrent_dropout=0.2,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    grud_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.05))(grud_model)
    grud_model = Dense(32, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.05))(grud_model)
    grud_model = Dense(96, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.05))(grud_model)
    grud_model = Dense(224, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.05))(grud_model)
    output_model = Dense(nClasses, activation='softmax')(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.01),
                    metrics=['accuracy'])

    return model

def BestModel_GRUD_flatten(input_dim_t, 
                    t_length, 
                    nClasses, 
                    use_bidirectional_rnn=False, 
                    use_batchnorm=True, 
                    return_sequences=False, 
                    **kwargs):

    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list = [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    grud_layer = GRUD(units=96,
                      return_sequences=False,
                      activation='sigmoid',
                      dropout=0.4,
                      recurrent_dropout=0.35,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    grud_model = Dense(32, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    grud_model = Dense(32, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001))(grud_model)
    output_model = Dense(nClasses, activation='softmax')(grud_model)

    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0021299066766227647),
                    metrics=['accuracy'])

    return model

def BuildModel_HMMLF_Manual_4(input_dim_t, 
                              t_length, 
                              input_dim_s_val, 
                              embedding_size, 
                              nClasses,
                              use_bidirectional_rnn=False, 
                              use_batchnorm=False, 
                              **kwargs):
    # accuracy = 0.8459
    # Input
    input_list = []
    embedding_list = []
    output_list = []

    ## The temporal variables
    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list += [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    # GRU layers
    grud_layer = GRUD(units=86,
                      return_sequences=False,  # True  False
                      activation='tanh',  # sigmoid
                      dropout=0.15,
                      recurrent_dropout=0.2,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    ## The numerical variables
    input_val = Input(shape=(input_dim_s_val,))
    embedding_val = Dense(input_dim_s_val)(input_val)
    input_list.append(input_val)
    embedding_list.append(embedding_val)

    ## The categorical variables
    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_list.append(input_shape)
        embedding_list.append(output_embedding)

    embedding_model = Concatenate()(embedding_list)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)   #,kernel_regularizer = regularizers.l1_l2(l1=3e-4, l2=3e-4)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(512, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(128, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)

    output_model = Concatenate()([grud_model, embedding_model])
    output_model = Dense(nClasses, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation('softmax')(output_model)
    
    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008), # RMSprop  sgd
                    metrics=['accuracy'])

    return model

def BuildModel_HMMLF_Manual_3(input_dim_t, 
                              t_length, 
                              input_dim_s_val, 
                              embedding_size, 
                              nClasses,
                              use_bidirectional_rnn=False, 
                              use_batchnorm=False, 
                              **kwargs):
    # accuracy = 0.8459
    # Input
    input_list = []
    embedding_list = []
    output_list = []

    ## The temporal variables
    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list += [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    # GRU layers
    grud_layer = GRUD(units=86,
                      return_sequences=False,  # True  False
                      activation='tanh',  # sigmoid
                      dropout=0.15,
                      recurrent_dropout=0.2,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    ## The numerical variables
    input_val = Input(shape=(input_dim_s_val,))
    embedding_val = Dense(input_dim_s_val)(input_val)
    input_list.append(input_val)
    embedding_list.append(embedding_val)

    ## The categorical variables
    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_list.append(input_shape)
        embedding_list.append(output_embedding)

    embedding_model = Concatenate()(embedding_list)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)   #,kernel_regularizer = regularizers.l1_l2(l1=3e-4, l2=3e-4)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(512, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(128, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)

    output_model = Concatenate()([grud_model, embedding_model])
    output_model = Dense(nClasses, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation('softmax')(output_model)
    
    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008), # RMSprop  sgd
                    metrics=['accuracy'])

    return model

def BuildModel_HMMLF_Manual_2(input_dim_t, 
                              t_length, 
                              input_dim_s_val, 
                              embedding_size, 
                              nClasses,
                              use_bidirectional_rnn=False, 
                              use_batchnorm=False, 
                              **kwargs):
    # Input
    input_list = []
    embedding_list = []
    output_list = []

    ## The temporal variables
    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list += [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    # GRU layers
    grud_layer = GRUD(units=86,
                      return_sequences=False,  # True  False
                      activation='tanh',  # sigmoid
                      dropout=0.15,
                      recurrent_dropout=0.2,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    ## The numerical variables
    input_val = Input(shape=(input_dim_s_val,))
    embedding_val = Dense(input_dim_s_val)(input_val)
    input_list.append(input_val)
    embedding_list.append(embedding_val)

    ## The categorical variables
    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_list.append(input_shape)
        embedding_list.append(output_embedding)

    embedding_model = Concatenate()(embedding_list)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)   #,kernel_regularizer = regularizers.l1_l2(l1=3e-4, l2=3e-4)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(512, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(128, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)

    output_model = Concatenate()([grud_model, embedding_model])
    output_model = Dense(nClasses, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation('softmax')(output_model)
    
    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008), # RMSprop  sgd
                    metrics=['accuracy'])

    return model

def BuildModel_HMMLF_Manual_1(input_dim_t, 
                              t_length, 
                              input_dim_s_val, 
                              embedding_size, 
                              nClasses,
                              use_bidirectional_rnn=False, 
                              use_batchnorm=False, 
                              **kwargs):
    # accuracy = 0.8459
    # Input
    input_list = []
    embedding_list = []
    output_list = []

    ## The temporal variables
    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list += [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    # GRU layers
    grud_layer = GRUD(units=86,
                      return_sequences=False,  # True  False
                      activation='tanh',  # sigmoid
                      dropout=0.15,
                      recurrent_dropout=0.2,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    ## The numerical variables
    input_val = Input(shape=(input_dim_s_val,))
    embedding_val = Dense(input_dim_s_val)(input_val)
    input_list.append(input_val)
    embedding_list.append(embedding_val)

    ## The categorical variables
    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_list.append(input_shape)
        embedding_list.append(output_embedding)

    embedding_model = Concatenate()(embedding_list)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)   #,kernel_regularizer = regularizers.l1_l2(l1=3e-4, l2=3e-4)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(512, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(128, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)

    output_model = Concatenate()([grud_model, embedding_model])
    output_model = Dense(nClasses, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation('softmax')(output_model)
    
    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008), # RMSprop  sgd
                    metrics=['accuracy'])

    return model

def BuildModel_HMMLF_Manual_0(input_dim_t, 
                              t_length, 
                              input_dim_s_val, 
                              embedding_size, 
                              nClasses,
                              use_bidirectional_rnn=False, 
                              use_batchnorm=False, 
                              **kwargs):
    # accuracy = 0.8459
    # Input
    input_list = []
    embedding_list = []
    output_list = []

    ## The temporal variables
    input_x = Input(shape=(None, input_dim_t))
    input_m = Input(shape=(None, input_dim_t))
    input_s = Input(shape=(None, 1))
    input_list += [input_x, input_m, input_s]

    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)

    # GRU layers
    grud_layer = GRUD(units=86,
                      return_sequences=False,  # True  False
                      activation='tanh',  # sigmoid
                      dropout=0.15,
                      recurrent_dropout=0.2,
                      **kwargs
                     )
    if use_bidirectional_rnn:
        grud_layer = Bidirectional_for_GRUD(grud_layer)
    grud_model = grud_layer([input_x, input_m, input_s])

    ## The numerical variables
    input_val = Input(shape=(input_dim_s_val,))
    embedding_val = Dense(input_dim_s_val)(input_val)
    input_list.append(input_val)
    embedding_list.append(embedding_val)

    ## The categorical variables
    for num_unique_cat, embed_size in embedding_size:
        input_shape = Input(shape=(1,))
        embedding = Embedding(num_unique_cat, embed_size, input_length=1)(input_shape)
        output_embedding = Reshape(target_shape=(embed_size,))(embedding)
        input_list.append(input_shape)
        embedding_list.append(output_embedding)

    embedding_model = Concatenate()(embedding_list)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)   #,kernel_regularizer = regularizers.l1_l2(l1=3e-4, l2=3e-4)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(512, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(256, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)
    embedding_model = Dense(128, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.001))(embedding_model)
    embedding_model = Dropout(0.35)(embedding_model)

    output_model = Concatenate()([grud_model, embedding_model])
    output_model = Dense(nClasses, activation=None)(output_model)
    if use_batchnorm:
        output_model = BatchNormalization()(output_model)
    output_model = Activation('softmax')(output_model)
    
    output_list = [output_model]

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=0.0008), # RMSprop  sgd
                    metrics=['accuracy'])

    return model