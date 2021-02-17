import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from keras.objectives import mean_squared_error
from datetime import datetime
startTime = datetime.now()


def calc_logprobs(var=1.0, bs=256, num_test_imgs=1):
    cifar_train = tfds.as_numpy(tfds.load(name='cifar10', split='train', as_supervised=False).batch(bs, drop_remainder=True))
    distributions = []
    for idx, d in enumerate(cifar_train):
        print('train batch: ', idx)
        cifar_train_batch = d['image'].astype(np.float32).reshape(bs, -1)
        dist = tfp.distributions.MultivariateNormalDiag(loc=cifar_train_batch,
                                                        scale_identity_multiplier=var * np.ones(bs, dtype=np.float32))
        distributions.append(dist)

    cifar_test = tfds.as_numpy(tfds.load(name='cifar10', split='test', as_supervised=False, shuffle_files=True).take(num_test_imgs))
    svhn_test = tfds.as_numpy(tfds.load(name='svhn_cropped', split='test', as_supervised=False, shuffle_files=True).take(num_test_imgs))

    svhn_logprobs = []
    cifar_logprobs = []

    with tf.Session() as session:
        for i in range(num_test_imgs):
            print('test image: ', i)
            cifar_test_img = next(cifar_test)['image'].astype(np.float32)
            svhn_test_img = next(svhn_test)['image'].astype(np.float32)

            cifar_test_batch = np.tile(np.expand_dims(cifar_test_img, axis=0), (bs, 1, 1, 1)).reshape(bs, -1)
            svhn_test_batch = np.tile(np.expand_dims(svhn_test_img, axis=0), (bs, 1, 1, 1)).reshape(bs, -1)

            svhn_logprob = []
            cifar_logprob = []

            for dist in distributions:
                svhn_batch_logprob = dist.log_prob(svhn_test_batch).eval()
                cifar_batch_logprob = dist.log_prob(cifar_test_batch).eval()
                #svhn_batch_logprob, cifar_batch_logprob = session.run([dist.log_prob(svhn_test_batch), dist.log_prob(cifar_test_batch)])
                svhn_logprob.append(np.mean(svhn_batch_logprob))
                cifar_logprob.append(np.mean(cifar_batch_logprob))

            svhn_logprobs.append(np.mean(svhn_logprob))
            cifar_logprobs.append(np.mean(cifar_logprob))

        cifar_logprob_mean = np.mean(cifar_logprobs)
        svhn_logprob_mean = np.mean(svhn_logprobs)

    return cifar_logprob_mean, svhn_logprob_mean

variances = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
cifar_means = []
svhn_means = []
cifar_stds = []
svhn_stds = []

for var_value in variances:
    print('variance value: ', var_value)
    cifar_mean, svhn_mean = calc_logprobs(var=var_value, bs=1024, num_test_imgs=5)
    cifar_means.append(cifar_mean)
    svhn_means.append(svhn_mean)

print('variances: ', variances)
print('cifar logprob means: ', cifar_means)
print('svhn logprob means: ', svhn_means)

with open("check_results.txt", 'wb') as f:
    pickle.dump(variances, f)
    pickle.dump(cifar_means, f)
    pickle.dump(svhn_means, f)
print(datetime.now() - startTime)
