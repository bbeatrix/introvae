import math
import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Add, \
    AveragePooling2D, Input, Dense, Flatten, UpSampling2D, Layer, Reshape, Concatenate, Lambda, MaxPooling2D, LeakyReLU
from keras.models import Model
from keras.regularizers import l2

def encoder_layers_deepsvdd(image_size, base_channels=32, bn_allowed=True):
    layers = []
    for i in range(0, 3):
        layers.append(Conv2D(base_channels*(2**i), (5, 5), padding='same', kernel_initializer='glorot_uniform', name='encoder_conv_'+str(i)))
        if bn_allowed:
            layers.append(BatchNormalization(axis=1, name='encoder_bn_'+str(i)))
        # layers.append(Activation('relu'))
        layers.append(LeakyReLU(alpha=0.1))
        layers.append(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', name='encoder_maxpool_'+str(i)))
    layers.append(Flatten(name='encoder_reshape'))
    return layers

def generator_layers_deepsvdd(image_size, base_channels=32, bn_allowed=True):
    layers = []
    layers.append(Reshape((-1, 4, 4), name='generator_reshape'))
    # layers.append(Activation('relu'))
    layers.append(LeakyReLU(alpha=0.1))
    for i in reversed(range(0, 3)):
        layers.append(Conv2D(base_channels*(2**i), (5, 5), padding='same', kernel_initializer='glorot_uniform', name='generator_conv_'+str(2-i)))
        if bn_allowed:
            layers.append(BatchNormalization(axis=1, name='generator_bn_'+str(2-i)))
        # layers.append(Activation('relu'))
        layers.append(LeakyReLU(alpha=0.1))
        layers.append(UpSampling2D(size=(2, 2), name='generator_upsample_'+str(2-i)))
    layers.append(Conv2D(3, (5, 5), padding='same', kernel_initializer='glorot_uniform', name='generator_conv_'+str(3)))
    if bn_allowed:
            layers.append(BatchNormalization(axis=1, name='generator_bn_'+str(3)))
    layers.append(Activation('sigmoid'))
    return layers

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
    layers.append(Activation('sigmoid'))
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
            input_tensor = Conv2D(filters[0], (1, 1), padding='same', kernel_initializer='glorot_normal', name=conv_name_base + str('00'), data_format='channels_first')(input_tensor)
            if bn_allowed:
                input_tensor = BatchNormalization(axis=bn_axis, name=bn_name_base + str('00'))(input_tensor)
            input_tensor = Activation('relu')(input_tensor)

        x = input_tensor
        for idx in range(len(filters)):
            x = Conv2D(filters[idx], kernels[idx], padding='same', kernel_initializer='he_normal', name=conv_name_base + str(idx), data_format='channels_first')(x)
            if bn_allowed:
                x = BatchNormalization(axis=bn_axis, name=bn_name_base + str(idx))(x)
            if idx <= len(filters) - 1:
                x = Activation('relu')(x)

        x = Add(name=model_type + '_resblock_add_' + stage + str(block))([x, input_tensor])
        x = Activation(last_activation)(x)
        return x

    return identity_block


def add_sampling(hidden, sampling, sampling_std, batch_size, latent_dim, wd):
    z_mean = Dense(latent_dim, kernel_regularizer=l2(wd))(hidden)
    if not sampling:
        z_log_var = Lambda(lambda x: 0*x, output_shape=[latent_dim])(z_mean)
        return z_mean, z_mean, z_log_var
    else:
        if sampling_std > 0:
            z_log_var = Lambda(lambda x: 0*x + K.log(K.square(sampling_std)), output_shape=[latent_dim])(z_mean)
        else:
            z_log_var = Dense(latent_dim, kernel_regularizer=l2(wd))(hidden)

        def sampling(inputs):
            z_mean, z_log_var = inputs
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        return z, z_mean, z_log_var
