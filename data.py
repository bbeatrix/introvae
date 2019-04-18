import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

K.set_image_data_format('channels_first')

def read_npy_file(item):
    data = np.transpose(np.load(item.decode()), (0,3,1,2))[0,:,:,:]
    return data.astype(np.float32)


def create_svhn_test_dataset(batch_size):
    test_dataset = tfds.load("svhn_cropped", split=tfds.Split.TEST)
    dataset = test_dataset \
        .take((26032 // batch_size) * batch_size) \
        .map(lambda x: x['image']) \
        .map(lambda x: tf.transpose(x, [2, 0, 1])) \
        .map(lambda x: tf.cast(x, tf.float32)) \
        .map(lambda x: x / 255.) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return (dataset, iterator, iterator_init_op, get_next)


def create_dataset(path, batch_size, limit):
    dataset = tf.data.Dataset.list_files(path, shuffle=True) \
        .take((limit // batch_size) * batch_size) \
        .map(lambda x: tf.py_func(read_npy_file, [x], [tf.float32])) \
        .map(lambda x: x / 255.) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return (dataset, iterator, iterator_init_op, get_next)


def create_dataset_from_ndarray(x, batch_size, limit):
    x_placeholder = tf.placeholder(np.float32, x.shape)
    dataset = tf.data.Dataset.from_tensor_slices(x_placeholder) \
        .take((limit // batch_size) * batch_size) \
        .map(lambda x: x / 255.) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return (x, x_placeholder, dataset, iterator, iterator_init_op, get_next)


def create_cifar10_unsup_dataset(batch_size, train_limit, test_limit, fixed_limit, normal_class, gcnorm, augment):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape)
    if augment:
        datagen = ImageDatagenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
        datagen.fit(x_train)
        x_train = datagen.flow(x_train, y=None, batch_size=batch_size)
        print('Using real-time data augmentation.')
    if gcnorm == "std":
        x_train_mean = np.mean(x_train, axis=(1, 2, 3), keepdims=True)
        x_test_mean = np.mean(x_test, axis=(1, 2, 3), keepdims=True)
        x_train_scale = np.std(x_train, axis=(1, 2, 3), keepdims=True)
        x_test_scale = np.std(x_test, axis=(1, 2, 3), keepdims=True)
        x_train = (x_train - x_train_mean) / x_train_scale
        x_test = (x_test - x_test_mean) / x_test_scale
    elif gcnorm == "l1":
        x_train_mean = np.mean(x_train, axis=(1, 2, 3), keepdims=True)
        x_test_mean = np.mean(x_test, axis=(1, 2, 3), keepdims=True)
        x_train_scale = np.sum(np.absolute(x_train), axis=(1, 2, 3), keepdims=True)
        x_test_scale = np.sum(np.absolute(x_test), axis=(1, 2, 3), keepdims=True)
        x_train = (x_train - x_train_mean) / x_train_scale
        x_test = (x_test - x_test_mean) / x_test_scale
    if normal_class == -1:
        print(x_train.shape)
        np.random.shuffle(x_train)
        train_tuple = create_dataset_from_ndarray(x_train, batch_size, train_limit)
    else:
        label_mask = y_train == normal_class
        x_train_oneclass = x_train[label_mask.flatten()]
        print(x_train_oneclass.shape)
        np.random.shuffle(x_train_oneclass)
        train_tuple = create_dataset_from_ndarray(x_train_oneclass, batch_size, train_limit)
    test_tuple = create_dataset_from_ndarray(x_test, batch_size, test_limit)
    fixed_tuple = create_dataset_from_ndarray(x_train, batch_size, fixed_limit)
    return (train_tuple, test_tuple, fixed_tuple)


def get_dataset(dataset, split, batch_size, limit):
    ds = tfds.load(name=dataset, split=split) \
        .take((limit // batch_size) * batch_size) \
        .map(lambda x: x['image']) \
        .map(lambda x: tf.transpose(x, [2, 0, 1])) \
        .map(lambda x: tf.cast(x, tf.float32)) \
        .map(lambda x: x / 255.) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = ds.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return ds, iterator, iterator_init_op, get_next




