import tensorflow as tf
import numpy as np


def read_npy_file(item):
    data = np.transpose(np.load(item.decode()), (0,3,1,2))[0,:,:,:]
    return data.astype(np.float32)


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
