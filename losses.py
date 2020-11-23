import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.objectives import mean_squared_error
import tensorflow_probability as tfp


tfd = tfp.distributions


def new_eubo_loss_fn(reconst_loss, mean, log_var, eubo_vau, cubo_power, cubo=False, margin=-1):
    if margin >= 0:
        loss = - eubo_vau * reconst_loss + K.maximum(0., margin - tf.reduce_mean(0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)))
    else:
        loss = - eubo_vau * reconst_loss + tf.reduce_mean(0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1))
    if cubo:
        loss = tf.exp(cubo_power * loss)
    return loss


def eubo_loss_fn(args, z, z_mean, z_log_var, reconst_loss_2, cubo=False):
    z = tf.reshape(z, (args.batch_size, args.z_num_samples, args.latent_dim))
    reconst_loss_2 = tf.reshape(reconst_loss_2, (args.batch_size, args.z_num_samples))
    sigma = tf.expand_dims(tf.exp(z_log_var), axis=1)

    z_mean = tf.tile(tf.expand_dims(z_mean, axis=1), (1, args.z_num_samples, 1))
    z_log_var = tf.tile(tf.expand_dims(z_log_var, axis=1), (1, args.z_num_samples, 1))

    p_z = tfd.Normal(loc=0.0, scale=1.0)
    log_p_z = p_z.log_prob(z)
    print('log_p_z', log_p_z)

    q_z = tfd.Normal(loc=z_mean, scale=tf.sqrt(sigma + 10e-12))
    log_q_z = q_z.log_prob(z)
    print('log_q_z', log_q_z)

    log_p_z = tf.reduce_sum(log_p_z, axis=-1)
    log_q_z = tf.reduce_sum(log_q_z, axis=-1)

    log_w = args.phi * args.train_size * reconst_loss_2 + args.chi * log_p_z - args.psi * log_q_z

    w = tf.exp(log_w - tf.reduce_max(log_w, axis=1, keep_dims=True))
    w_hat = w / tf.reduce_sum(w, axis=1, keep_dims=True)
    if cubo:
        w_hat = tf.square(w_hat)
    eubo_loss = tf.stop_gradient(-w_hat) * log_q_z

    eubo_loss = tf.reduce_mean(eubo_loss)
    return eubo_loss


HALF_LOG_TWO_PI = 0.91893

def reconstruction_loss(args, gamma, log_gamma, x, xr):
    x = tf.expand_dims(x, axis=1)
    x = tf.tile(x, (1, args.z_num_samples, 1, 1, 1))
    x = tf.reshape(x, (args.batch_size * args.z_num_samples, )+ args.original_shape)

    if args.obs_noise_model == 'bernoulli':
        return -tf.reduce_sum(x * tf.log(tf.maximum(xr, 1e-8)) + (1-x) * tf.log(tf.maximum(1-xr, 1e-8)), [1, 2, 3])
    else:
        return tf.reduce_sum(tf.square((x - xr) / gamma) / 2 + log_gamma + HALF_LOG_TWO_PI, [1, 2, 3])


def mse_loss(x, x_decoded, original_shape):
    original_dim = np.float32(np.prod(original_shape))
    return K.mean(original_dim * mean_squared_error(x, x_decoded))


def reg_loss(mean, log_var):
    return K.mean(0.5 * K.sum(- 1 - log_var + K.square(mean) + K.exp(log_var), axis=-1))


def augmented_variance_loss(mean, log_var):
    variance = K.exp(log_var)
    mean_variance = K.var(mean, axis=0, keepdims=True)
    total_variance = variance + mean_variance
    loss = 0.5 * K.sum(-1 - K.log(total_variance) + total_variance, axis=-1)
    return K.mean(loss)


def size_loss(mean):
    loss = 0.5 * K.sum(K.square(mean), axis=-1)
    return K.mean(loss)


def reg_loss_new(mean, log_var):
    return augmented_variance_loss(mean, log_var) + size_loss(mean)

def mmd_penalty(args, sample_qz, sample_pz):
        opts = {'pz_scale': 1.0, 'mmd_kernel': 'IMQ', 'verbose': True, 'pz': 'normal' , 'zdim': args.latent_dim}
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = args.batch_size # utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        # if opts['verbose']:
        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]],
        #         'Maximal Qz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_qz)],
        #                         'Average Qz squared pairwise distance:')

        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_pz, [-1]), 1).values[0]],
        #         'Maximal Pz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_pz)],
        #                         'Average Pz squared pairwise distance:')

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            res1 += tf.exp( - distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp( - distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            if opts['pz'] == 'normal':
                Cbase = 2. * opts['zdim'] * sigma2_p
            elif opts['pz'] == 'sphere':
                Cbase = 2.
            elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
                Cbase = opts['zdim']
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat

