from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from keras import backend as K
from keras import activations
from keras.layers import Dense, Activation, Concatenate, Layer, Dropout

debug_flag = int(os.environ.get('KERAS_ATTENTION_DEBUG', 0))

class Mag(Layer):
    '''
    @ https://github.com/emnlp-mimic/mimic/blob/main/base.py#L141
    @ https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/modeling.py#L13
    '''

    def __init__(self, activation=None, dropout=None, **kwargs):
        super(Mag, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.dropout = dropout
        # self.beta = tf.Variable(initial_value=tf.random.normal((1,)), trainable=True)
        self.beta = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('input_shape of MAG should be a list of 2 inputs.')

        input_dim_main = int(input_shape[0][-1])
        input_dim_aux = int(input_shape[1][-1])

        self.concat = Concatenate()
        self.fc1 = Dense(1, use_bias=False)  # later change the use_bias=True for HM and SM
        self.fc2 = Dense(input_dim_main, use_bias=False) # later change the use_bias=True for HM and SM
        self.dropout = Dropout(rate=self.dropout)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], int(input_shape[0][-1])

    def call(self, inputs, **kwargs):
        input = self.concat(inputs)
        input = self.fc1(input)
        input = self.activation(input)
        adjust = self.fc2(input * inputs[1])

        one = K.constant([1.], dtype=K.dtype(adjust))
        alpha = tf.reduce_min([tf.norm(inputs[0]) / tf.norm(adjust) * self.beta, one])
        output = inputs[0] + alpha * adjust
        output = self.dropout(output)

        return output

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = {
            'activation': activations.serialize(self.activation),
            'dropout': self.dropout
        }
        base_config = super(Mag, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class WeightedAverage(Layer):

    
    def __init__(self, n_output):
        super(WeightedAverage, self).__init__()
        self.W = tf.Variable(initial_value=tf.random.uniform(shape=[1,1,n_output], minval=0, maxval=1),
            trainable=True) # (1,1,n_inputs)

    def call(self, inputs):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = Concatenate(axis=-1)(inputs) # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim
        
        return tf.reduce_sum(weights * inputs, axis=-1) # (n_batch, n_feat)

class Gmu(object if debug_flag else Layer):
    '''
    @ https://github.com/terenceylchow124/Meme-MultiModal/blob/main/models.py
    @ https://github.com/xkaple01/multimodal-classification/blob/v1.1/multimodal_classification/gaited_multimodal_unit.ipynb
    '''
    
    def __init__(self, units=128, **kwargs):
        super(Gmu, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim_1 = int(input_shape[0][-1])
        input_dim_2 = int(input_shape[1][-1])
        with K.name_scope(self.name if not debug_flag else 'gmu'):
            self.weight_sigmoid = self.add_weight(shape=(self.units * 2, 1), initializer='uniform', trainable=True)
            self.h_i = Dense(self.units, use_bias=False, activation='tanh', name='h_i')
            self.h_t = Dense(self.units, use_bias=False, activation='tanh', name='h_t')
            self.concat = Concatenate(name='concat_h')
            self.activation_z = Activation('sigmoid', name='activation_z')

        if not debug_flag:
            # debug: the call to build() is done in call().
            super(Gmu, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.units

    def __call__(self, inputs, **kwargs):
        if debug_flag:
            return self.call(inputs, **kwargs)
        else:
            return super(Gmu, self).__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        if debug_flag:
            self.build(inputs)
        h_i = self.h_i(inputs[0])
        h_t = self.h_t(inputs[1])
        h = self.concat([h_i, h_t])
        z = self.activation_z(K.dot(h, self.weight_sigmoid))
        f = z * h_i + (1-z) * h_t
        return f

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = {
            'units': self.units
        }
        base_config = super(Gmu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))