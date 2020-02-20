import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.objectives import mean_squared_error

def mse_loss(x, x_decoded, original_shape):
    original_dim = np.float32(np.prod(original_shape))
    return K.mean(original_dim * mean_squared_error(x, x_decoded))

def reg_loss(mean, log_var):
    return K.mean(0.5 * K.sum(- 1 - log_var + K.square(mean) + K.exp(log_var), axis=-1))

def size_loss(mean):
    loss = 0.5 * K.sum(K.square(mean), axis=-1)
    return K.mean(loss)
