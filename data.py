import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import os

K.set_image_data_format('channels_first')


def get_dataset(dataset, split, batch_size, limit, augment=False, normal_class=-1, outliers=False):

    from glob import glob
    if dataset == 'tiny-imagenet-200':
        directory = "/home/csadrian/datasets/tiny-imagenet-200/train/*/"
        dirs = [d + "images/*.JPEG" for d in glob(directory)]
        ds = tf.data.Dataset.list_files(dirs)
        ds = ds.map(lambda x: {'image':tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x), channels=3), [32, 32])})
    else:
        ds = tfds.load(name=dataset, split=split)

    if split == tfds.Split.TRAIN:
        ds = ds.shuffle(100000)

    if normal_class != -1:
        if outliers:
            ds = ds.filter(lambda x: tf.not_equal(x['label'], normal_class))
        else:
            ds = ds.filter(lambda x: tf.equal(x['label'], normal_class))

    ds = ds.take((limit // batch_size) * batch_size) \
        .map(lambda x: x['image']) \
        .map(lambda x: tf.cast(x, tf.float32)) \
        .map(lambda x: x + tf.random.uniform(x.shape)) \
        .map(lambda x: x / 255.)

    if augment:
        ds = ds.map(lambda x: augment_transforms(x)) \
               .map(lambda x: tf.clip_by_value(x, -1, 1))
    ds = ds.map(lambda x: tf.transpose(x, [2, 0, 1])) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)


    iterator = ds.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return ds, iterator, iterator_init_op, get_next


def augment_transforms(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    #x = tf.image.random_hue(x, 0.08)
    #x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.contrib.image.rotate(x, tf.random_uniform(shape=[],minval=0., maxval=np.pi/2, dtype=tf.float32))
    x = tf.contrib.image.translate(x, tf.random_uniform([2,], minval=0., maxval=4., dtype=tf.float32), interpolation='NEAREST')
    return x
