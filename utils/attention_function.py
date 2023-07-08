import numpy as np
import pandas as pd
import keras.backend as K
from keras.layers import multiply
from keras.layers.core import Dense, Reshape, Lambda, RepeatVector, Permute, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Model, Input
import matplotlib.pyplot as plt

def get_activations(model, inputs, print_shape_only=False, layer_name=None, verbose=False):
    """
    Get activations from a model
    Args:
        model: a keras model
        inputs: the inputs for the model
        print_shape_only: whether to print the shape of the layer or the whole activation layer
        layer_name: name of specific layer to return
        verbose: whether to show all outputs
    Returns:
        activations: list, list of activations
    """
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if verbose:
            print('----- activations -----')
            if print_shape_only:
                print(layer_activations.shape)
            else:
                print(layer_activations)
    return activations

def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is random except that first value equals the target y.
    network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    Args:
        n: the number of samples to retrieve.
        time_steps: the number of time steps of your series.
        input_dim: the number of dimensions of each element in the series.
        attention_column: the column linked to the target. Everything else is purely random.
    Returns:
        x: model inputs
        y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y

def attention_3d_block_temporal(inputs, TIME_STEPS):
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """ 
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def attention_3d_block_spatial(inputs, FEATURES):
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """ 
    n_steps = int(inputs.shape[1]) 
    a = Permute((1, 2))(inputs) 
    a = Reshape((n_steps, FEATURES))(a) 
    a = Dense(FEATURES, activation='softmax', name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a])
    return output_attention_mul

def attention_3d_block_plus(inputs, FEATURES, TIME_STEPS):
    a = Permute((2, 1))(inputs)
    a = Reshape((FEATURES, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec_time')(a)

    b = Permute((1, 2))(inputs) 
    b = Reshape((TIME_STEPS, FEATURES))(b) 
    b_probs = Dense(FEATURES, activation='softmax', name='attention_vec_spatial')(b)

    output_attention_mul = multiply([inputs, a_probs, b_probs])

    return output_attention_mul

def attention_3d_block_flatten(inputs, TIME_STEPS):
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """ 
    input_dim = int(inputs.shape[2])
    a = Flatten()(inputs)
    a = Dense(TIME_STEPS*input_dim, activation='softmax')(a)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul





