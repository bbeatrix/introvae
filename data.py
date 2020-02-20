import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import os

K.set_image_data_format('channels_first')


def get_dataset(args, dataset, split, batch_size, limit, augment=False, normal_class=-1, outliers=False, add_obs_noise=False, add_iso_noise=False):

    if dataset == 'emnist-letters':
        dataset = 'emnist/letters'
    elif dataset == 'imagenet':
        dataset = 'downsampled_imagenet/32x32'
        if split == tfds.Split.TEST:
            split = tfds.Split.VALIDATION

    if dataset == 'uniform-noise':
        def random_uniform_generator():
            while True:
                yield {'image': np.random.randint(0, high=255, size=(28,28,1))}
        ds = tf.data.Dataset.from_generator(random_uniform_generator, output_types={'image': tf.int32}, output_shapes={'image': (28, 28, 1)})
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
        .map(lambda x: tf.cast(x, tf.float32))

    if add_obs_noise:
        if dataset == 'downsampled_imagenet/32x32':
            ds = ds.map(lambda x: x + tf.random.uniform([32,32,3]))
        else:
            ds = ds.map(lambda x: x + tf.random.uniform(x.shape))


    image_width = ds.output_shapes[0].value
    image_height = ds.output_shapes[1].value
    image_channels = ds.output_shapes[2].value

    if image_width != args.shape[0] or image_height != args.shape[1]:
        print('Resize (crop/pad) images to taget shape.')
        ds = ds.map(lambda x: tf.image.resize_image_with_crop_or_pad(x, args.shape[0], args.shape[1]))
    if image_channels != 3 and args.color:
        print('Transform grayscale images to rgb.')
        ds = ds.map(lambda x: tf.image.grayscale_to_rgb(x))
    elif image_channels !=1 and not args.color:
        print('Transform rgb images to grayscale.')
        ds = ds.map(lambda x: tf.image.rgb_to_grayscale(x))

    ds = ds.map(lambda x: x / 255.)

    if add_iso_noise:
       if split == tfds.Split.TRAIN:
           print("Adding iso noise to train of {}.".format(dataset))
           ds = ds.map(lambda x: x + tf.random.normal(x.shape, stddev=.25))
           ds = ds.map(lambda x: tf.clip_by_value(x, 0, 1))

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
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.contrib.image.rotate(x, tf.random_uniform(shape=[],minval=0., maxval=np.pi/2, dtype=tf.float32))
    x = tf.contrib.image.translate(x, tf.random_uniform([2,], minval=0., maxval=4., dtype=tf.float32), interpolation='NEAREST')
    return x
