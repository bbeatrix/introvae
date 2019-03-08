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

def create_dataset_from_tensor_slices(imgs, batch_size, limit):
    dataset = tf.data.Dataset.from_tensor_slices(imgs) \
        .take((limit // batch_size) * batch_size) \
        .map(lambda x: tf.tile(x, (3,1,1))) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return (dataset, iterator, iterator_init_op, get_next)


def create_dsprites_datasets(batch_size, train_size, test_size, latent_cloud_size):

    dataset_zip = np.load('/home/csadrian/disent/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='bytes')

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs'].astype(np.float32)
    imgs = np.expand_dims(imgs, axis=1)
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]
    print(latents_classes)
    print('Metadata: \n', metadata)


    np.random.shuffle(imgs)
    train_data = imgs[:50000]
    test_data = imgs[50000:60000]
    latent_cloud_data = imgs[50000:60000]
    
    train_ds = create_dataset_from_tensor_slices(train_data, batch_size, 50000)
    test_ds = create_dataset_from_tensor_slices(test_data, batch_size, 10000)
    latent_cloud_ds = create_dataset_from_tensor_slices(latent_cloud_data, batch_size, 10000)
    return train_ds, test_ds, latent_cloud_ds