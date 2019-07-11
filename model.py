import math
import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, \
     AveragePooling2D, Input, Dense, Flatten, UpSampling2D, Layer, Reshape, Concatenate, Lambda, MaxPooling2D, LeakyReLU, Conv2DTranspose
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import RandomNormal
import numpy as np
import tensorflow_probability as tfp

def encoder_layers_baseline_mnist(image_size, image_channels, base_channels, bn_allowed, wd, seed):
    """
    Following Nalisnick et al. (arxiv: 1810.09136), for MNIST, we use the encoder architecture
    described in Rosca et al. (arxiv: 1802.06847) in appendix K table 4.
    """
    
    layers = []

    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(0.02), seed=seed)

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



def add_observable_output(generator_output, args, gamma):

    def _add_observable_output_inner(generator_output):
        if args.obs_noise_model == 'bernoulli':
            return tfp.distributions.Independent(tfp.distributions.Bernoulli(generator_output), len(args.original_shape))
        elif args.obs_noise_model == 'gaussian':
            return tfp.distributions.Independent(tfp.distributions.Normal(loc=generator_output, scale=gamma), len(args.original_shape))
        else:
            raise Exception("ob_noise_modell {} is not implemented.".format(args.obs_noise_model))
        return generator_output

    return Lambda(_add_observable_output_inner, output_shape=(args.batch_size,)+args.original_shape)(generator_output)


def encoder_layers_introvae(image_size, base_channels, bn_allowed):
    layers = []
    layers.append(Conv2D(base_channels, (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='encoder_conv_0'))
    if bn_allowed:
        layers.append(BatchNormalization(axis=1, name='encoder_bn_0'))
    layers.append(Activation('relu'))
    layers.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='encoder_avgpool_0'))

    map_size = image_size[0] // 2

    block = 1
    channels = base_channels * 8
    while map_size > 4:
        layers.append(residual_block('encoder', [(3, 3), (3, 3)], channels, block=block, bn_allowed=bn_allowed))
        layers.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='encoder_avgpool_'+ str(block)))
        block += 1
        map_size = map_size // 2
        channels = channels * 2 if channels <= 256  else 512

    layers.append(residual_block('encoder', kernels=[(3, 3), (3, 3)], filters=channels, block=block, bn_allowed=bn_allowed, last_activation="linear"))
    layers.append(Flatten(name='encoder_reshape'))
    return layers


def generator_layers_introvae(image_size, base_channels, bn_allowed):
    layers = []
    layers.append(Dense(512 * 4 * 4, name='generator_dense'))
    #layers.append(LeakyReLU())
    layers.append(Activation('relu'))
    layers.append(Reshape((512, 4, 4), name='generator_reshape'))
    layers.append(residual_block('generator', kernels=[(3, 3), (3, 3)], filters=512, block=1, bn_allowed=bn_allowed))

    map_size = 4
    upsamples = int(math.log2(image_size[0]) - 2)
    block = 2
    channels = 512

    for i in range(upsamples - 6):
        layers.append(UpSampling2D(size=(2, 2), name='generator_upsample_' + str(block)))
        layers.append(residual_block('generator', [(3, 3), (3, 3)], 512, block=block, bn_allowed=bn_allowed))
        map_size = map_size * 2
        block += 1

    while map_size < image_size[0]: # 4
        channels = channels // 2 if channels >= 32 else 16
        layers.append(UpSampling2D(size=(2, 2), name='generator_upsample_' + str(block)))
        layers.append(residual_block('generator', [(3, 3), (3, 3)], channels, block=block, bn_allowed=bn_allowed))
        map_size = map_size * 2
        block += 1

    layers.append(Conv2D(3, (5, 5), padding='same', kernel_initializer='he_normal', name='generator_conv_0'))
    #layers.append(Activation('tanh'))
    return layers


def residual_block(model_type, kernels, filters, block, bn_allowed, stage='a', last_activation="relu"):

    def identity_block(input_tensor, filters=filters):
        if isinstance(filters, int):
            filters = [filters] * len(kernels)
        assert len(filters) == len(kernels), 'Number of filters and number of kernels differs.'

        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        bn_name_base = model_type + '_resblock_bn_' + stage + str(block) + '_branch_'
        conv_name_base = model_type + '_resblock_conv_' + stage + str(block) + '_branch_'

        if K.int_shape(input_tensor[-1]) != filters[0]:
            input_tensor = Conv2D(filters[0], (1, 1), padding='same', kernel_initializer='he_normal', name=conv_name_base + str('00'), data_format='channels_first')(input_tensor)
            if bn_allowed:
                input_tensor = BatchNormalization(axis=bn_axis, name=bn_name_base + str('00'))(input_tensor)
            input_tensor = Activation('relu')(input_tensor)
            #input_tensor = LeakyReLU()(input_tensor)

        x = input_tensor
        for idx in range(len(filters)):
            x = Conv2D(filters[idx], kernels[idx], padding='same', kernel_initializer='he_normal', name=conv_name_base + str(idx), data_format='channels_first')(x)
            if bn_allowed:
                x = BatchNormalization(axis=bn_axis, name=bn_name_base + str(idx))(x)
            if idx < len(filters) - 1:
                #x = LeakyReLU()(x)
                x = Activation('relu')(x)


        x = Add(name=model_type + '_resblock_add_' + stage + str(block))([x, input_tensor])
        x = Activation(last_activation)(x)
        return x

    return identity_block
