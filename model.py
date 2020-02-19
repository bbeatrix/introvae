import math
import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, \
     AveragePooling2D, Input, Dense, Flatten, UpSampling2D, Layer, Reshape, Concatenate, Lambda, MaxPooling2D, LeakyReLU, Conv2DTranspose
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import RandomNormal
import numpy as np
from SpectralNormalizationKeras import ConvSN2D

def encoder_layers_baseline_mnist(image_size, image_channels, base_channels, bn_allowed, wd, seed, encoder_use_sn):
    """
    Following Nalisnick et al. (arxiv: 1810.09136), for MNIST, we use the encoder architecture
    described in Rosca et al. (arxiv: 1802.06847) in appendix K table 4.
    """
    
    layers = []

    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(0.02), seed=seed)
    if encoder_use_sn:
        layers.append(ConvSN2D(8 , (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu')) 

        layers.append(ConvSN2D(16, (5, 5), strides=(1, 1), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))

        layers.append(ConvSN2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))

        layers.append(ConvSN2D(64, (5, 5), strides=(1, 1), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))

        layers.append(ConvSN2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))
    else:
        layers.append(Conv2D(8 , (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu')) 

        layers.append(Conv2D(16, (5, 5), strides=(1, 1), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))

        layers.append(Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))

        layers.append(Conv2D(64, (5, 5), strides=(1, 1), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))

        layers.append(Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
        layers.append(Activation('relu'))

    layers.append(Flatten())
    return layers


def generator_layers_baseline_mnist(image_size, image_channels, base_channels, bn_allowed, wd, seed):
    """
    We follow Nalisnick et al. (arxiv: 1810.09136).
    """

    layers = []

    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(0.02), seed=seed)

    layers.append(Dense(64*7*7))
    layers.append(Reshape((64, 7, 7)))

    layers.append(Conv2DTranspose(32 , (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))

    layers.append(Conv2DTranspose(32 , (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))

    layers.append(Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))
    layers.append(Activation('relu'))    

    layers.append(Conv2D(image_channels, (5, 5), strides=(1, 1), padding='same', kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(wd)))

    return layers 


def add_observable_output(generator_output, args):
    generator_output = Activation('sigmoid')(generator_output)
    return generator_output


def encoder_layers_dcgan_univ(image_size, image_channels, base_channels, bn_allowed, wd, encoder_use_sn):

    n_upsample = 0
    n = image_size[0]
    while n % 2 == 0 and n>=8:
        n = n // 2
        n_upsample += 1
    start_width = n

    kernel = 4
    upsamples = 0

    channels = base_channels
    width = image_size[0]
    layers = []
    idx = 0
    while width >= 8:
        if n_upsample <= upsamples:
            border_mode="same"
            stride = 1
            activation = "linear"
            use_bn = False
        elif idx == 0:
            border_mode = "same"
            stride = 1
            activation = "relu"
            use_bn = bn_allowed
        else:
            border_mode = "same"
            stride = 2
            width = width // 2
            channels = 2*channels
            upsamples += 1
            activation = "relu"
            use_bn = bn_allowed
        if encoder_use_sn:
            layers.append(ConvSN2D(channels, (kernel, kernel), strides=(stride, stride), padding=border_mode, use_bias=False, kernel_regularizer=l2(wd)))
        else:
            layers.append(Conv2D(channels, (kernel, kernel), strides=(stride, stride), padding=border_mode, use_bias=False, kernel_regularizer=l2(wd)))

        if use_bn:
            layers.append(BatchNormalization(axis=1))
        if activation == "relu":
            #layers.append(LeakyReLU(name="encoder_{}".format(idx)))    
            layers.append(Activation(activation, name="encoder_{}".format(idx)))

        else:
            layers.append(Activation(activation, name="encoder_{}".format(idx)))
        idx += 1
    layers.append(Flatten())
    return layers


def generator_layers_dcgan_univ(image_size, image_channels, base_channels, bn_allowed, wd):

    n = image_size[0]
    n_upsample = 0

    while n % 2 == 0 and n>=8:
        n = n // 2
        n_upsample += 1
    start_width = n
    print("start_width", start_width)
    layers = []
    channels = 2**n_upsample*base_channels
    layers.append(Dense(channels*start_width*start_width))
    layers.append(Reshape((-1, start_width, start_width)))

    size = start_width
    stride = 2
    kernel = 4
    border_mode="same"
    idx = 0
    while size < image_size[0]:

        activation="relu"
        use_bn = bn_allowed

        channels = channels // 2
        layers.append(Conv2D(channels, (kernel, kernel), use_bias=False, strides=(1, 1), padding=border_mode, kernel_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization(axis=1))
        if activation == "relu":
            #layers.append(LeakyReLU(name="generator_{}".format(idx)))    
            layers.append(Activation(activation, name="generator_{}".format(idx)))
        else:
            layers.append(Activation(activation, name="generator_{}".format(idx)))

        if image_size[0] != size:
            layers.append(UpSampling2D((stride, stride)))
            size = size * 2
        idx += 1
    layers.append(Conv2D(image_channels, (kernel, kernel), use_bias=False, strides=(1, 1), padding=border_mode, kernel_regularizer=l2(wd)))
    #layers.append(Activation("tanh", name="generator_{}".format(idx)))

    return layers


def add_sampling(hidden, sampling, sampling_std, batch_size, latent_dim, wd, z_mean_layer, z_log_var_layer):
    z_mean = z_mean_layer(hidden)
    if not sampling:
        z_log_var = Lambda(lambda x: 0*x, output_shape=[latent_dim])(z_mean)
        return z_mean, z_mean, z_log_var
    else:
        if sampling_std > 0:
            z_log_var = Lambda(lambda x: 0*x + K.log(K.square(sampling_std)), output_shape=[latent_dim])(z_mean)
        else:
            z_log_var = z_log_var_layer(hidden)

        def sampling(inputs):
            z_mean, z_log_var = inputs
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        return z, z_mean, z_log_var
