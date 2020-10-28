import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras, keras.backend as K

from keras.layers import Activation, Input, Dense, Flatten
from keras.models import Model
from keras.regularizers import l2

import os, sys, time
from collections import OrderedDict

import model, params, losses, utils, data

from sklearn.metrics import roc_auc_score

from keras.utils import to_categorical
from transformations import Transformer
import datetime

now = datetime.datetime.now()

import neptune

args = params.getArgs()
print(args)

args.prefix = args.prefix + now.strftime("%Y%m%d_%H%M%S%f")

neptune.init(project_qualified_name="csadrian/oneclass")
neptune.create_experiment(params=vars(args), name=args.name)
for tag in args.tags.split(','):
    neptune.append_tag(tag)

#
# Config
#

np.random.seed(args.seed)
tf.set_random_seed(args.seed)

print('Keras version: ', keras.__version__)
print('Tensorflow version: ', tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

#
# Datasets
#

K.set_image_data_format('channels_first')

data_path = os.path.join(args.datasets_dir, args.dataset)

iterations = args.nb_epoch * args.train_size // args.batch_size
iterations_per_epoch = args.train_size // args.batch_size

if args.normal_class != -1:
    train_size = args.train_size // args.num_classes
    test_size_a = args.test_size // args.num_classes
    test_size_b = args.test_size - test_size_a
else:
    train_size = args.train_size
    test_size_a = args.test_size
    test_size_b = args.test_size

print("train_size: ", train_size)
print("test_size_a: ", test_size_a)
print("test_size_b: ", test_size_b)

train_data, train_iterator, train_iterator_init_op, train_next = data.get_dataset(args, args.dataset, tfds.Split.TRAIN, args.batch_size, train_size, args.augment, args.normal_class, add_obs_noise=args.add_obs_noise)
fixed_data, fixed_iterator, fixed_iterator_init_op, fixed_next = data.get_dataset(args, args.dataset, tfds.Split.TRAIN, args.batch_size, args.latent_cloud_size, args.augment, args.normal_class, add_obs_noise=args.add_obs_noise)
test_data_a, test_iterator_a, test_iterator_init_op_a, test_next_a = data.get_dataset(args, args.test_dataset_a, tfds.Split.TEST, args.batch_size, test_size_a, args.augment, args.normal_class, outliers=False, add_obs_noise=args.add_obs_noise)
test_data_b, test_iterator_b, test_iterator_init_op_b, test_next_b = data.get_dataset(args, args.test_dataset_b, tfds.Split.TEST, args.batch_size, test_size_b, args.augment, args.normal_class, outliers=True, add_obs_noise=args.add_obs_noise)

if args.neg_dataset is not None:
    neg_train_size = args.neg_train_size
    neg_test_size = args.neg_test_size
    neg_data, neg_iterator, neg_iterator_init_op, neg_next = data.get_dataset(args, args.neg_dataset, tfds.Split.TRAIN, args.batch_size, neg_train_size, args.augment, add_obs_noise=args.add_obs_noise, add_iso_noise=args.add_iso_noise_to_neg)
    neg_test_data, neg_test_iterator, neg_test_iterator_init_op, neg_test_next = data.get_dataset(args, args.neg_dataset, tfds.Split.TEST, args.batch_size, neg_test_size, args.augment, add_obs_noise=args.add_obs_noise, add_iso_noise=args.add_iso_noise_to_neg)

args.n_channels = 3 if args.color else 1
args.original_shape = (args.n_channels, ) + args.shape

transformer = Transformer()

#
# Build networks
#

if args.model_architecture == 'deepsvdd':
    encoder_layers = model.encoder_layers_deepsvdd(args.shape, args.base_filter_num, args.encoder_use_bn)
    generator_layers = model.generator_layers_deepsvdd(args.shape, args.base_filter_num, args.generator_use_bn)

elif args.model_architecture == 'baseline_mnist':
    encoder_layers = model.encoder_layers_baseline_mnist(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.encoder_wd, args.seed, args.encoder_use_sn)
    generator_layers = model.generator_layers_baseline_mnist(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.generator_wd, args.seed, args)

elif args.model_architecture == 'dcgan':
    encoder_layers = model.encoder_layers_dcgan(args.shape, args.base_filter_num, args.encoder_use_bn, args.encoder_wd)
    generator_layers = model.generator_layers_dcgan(args.shape, args.base_filter_num, args.encoder_use_bn, args.generator_wd)
elif args.model_architecture == 'dcgan_univ':
    encoder_layers = model.encoder_layers_dcgan_univ(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.encoder_wd, args.encoder_use_sn)
    generator_layers = model.generator_layers_dcgan_univ(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.generator_wd)
else:
    encoder_layers = model.encoder_layers_introvae(args.shape, args.base_filter_num, args.encoder_use_bn)
    generator_layers = model.generator_layers_introvae(args.shape, args.base_filter_num, args.generator_use_bn)


encoder_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='encoder_input')

generator_s_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='generator_input')
generator_input = Input(batch_shape=(args.batch_size * args.z_num_samples, args.latent_dim), name='generator_input')
if args.aux:
    aux_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='aux_input')
if args.neg_dataset is not None:
    neg_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='neg_input')

encoder_output = encoder_input
for layer in encoder_layers:
    encoder_output = layer(encoder_output)
if args.aux:
    aux_encoder_output = aux_input
    for layer in encoder_layers:
        aux_encoder_output = layer(aux_encoder_output)


generator_output = generator_input
generator_s_output = generator_s_input

for layer in generator_layers:
    generator_output = layer(generator_output)
    generator_s_output = layer(generator_s_output)

generator_output = model.add_observable_output(generator_output, args)
generator_s_output = model.add_observable_output(generator_s_output, args)

z_mean_layer = Dense(args.latent_dim, kernel_regularizer=l2(args.encoder_wd))
z_log_var_layer = Dense(args.latent_dim, kernel_regularizer=l2(args.encoder_wd))
z, z_mean, z_log_var = model.add_sampling(encoder_output, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd, z_mean_layer, z_log_var_layer, z_num_samples=args.z_num_samples)
print(z, z_mean)
if args.trained_gamma:
    log_gamma = tf.get_variable('log_gamma', [], tf.float32, tf.constant_initializer(value=args.initial_log_gamma))
else:
    log_gamma = tf.constant(args.initial_log_gamma)
gamma = tf.exp(log_gamma)

encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var])
generator = Model(inputs=generator_input, outputs=generator_output)
generator_s = Model(inputs=generator_s_input, outputs=generator_s_output)
if args.aux:
    aux_encoder = Model(inputs=aux_input, outputs=[aux_encoder_output])


if args.separate_discriminator:
    discriminator_output = encoder_input
    for layer in encoder_layers:
        discriminator_output = layer(discriminator_output)
    _, zd_mean, zd_log_var = model.add_sampling(discriminator_output, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd, z_num_samples=args.z_num_samples)
    discriminator = Model(inputs=encoder_input, outputs=[zd_mean, zd_log_var])
else:
    discriminator = encoder
    zd_mean, zd_log_var = z_mean, z_log_var

xr = generator(z)
reconst_latent_input = Input(batch_shape=(args.batch_size * args.z_num_samples, args.latent_dim), name='reconst_latent_input')
zr_mean, zr_log_var = discriminator(generator(reconst_latent_input))
zr_mean_ng, zr_log_var_ng = discriminator(tf.stop_gradient(generator(reconst_latent_input)))
xr_latent = generator(reconst_latent_input)

sampled_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='sampled_latent_input')
zpp_gen = generator_s(sampled_latent_input)
zpp_mean, zpp_log_var = discriminator(zpp_gen)
zpp_mean_ng, zpp_log_var_ng = discriminator(tf.stop_gradient(zpp_gen))
zg = tf.stop_gradient(zpp_gen)
for layer in encoder_layers:
    zg = layer(zg)
zg, zg_mean, zg_log_var = model.add_sampling(zg, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd, z_mean_layer, z_log_var_layer, z_num_samples=args.z_num_samples)
xgr = generator(zg)

if args.neg_dataset is not None:
    neg_encoder_output = neg_input
    for layer in encoder_layers:
        neg_encoder_output = layer(neg_encoder_output)
    zn, zn_mean, zn_log_var = model.add_sampling(neg_encoder_output, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd, z_mean_layer, z_log_var_layer, z_num_samples=args.z_num_samples)
    xnr = generator(zn)
    neg_latent_input = Input(batch_shape=(args.batch_size * args.z_num_samples, args.latent_dim), name='neg_latent_input')
    xnr_latent = generator(neg_latent_input)

global_step = tf.Variable(0, trainable=False)

starter_learning_rate = args.lr
if args.lr_schedule == 'exponential':
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.96, staircase=True)
elif args.lr_schedule == 'piecewise_constant':
    boundaries = [100000,150000]
    values = [starter_learning_rate, 0.1*starter_learning_rate, 0.01*starter_learning_rate]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
else:
    learning_rate = tf.constant(args.lr)


if args.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    encoder_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    joint_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    if args.aux:
        aux_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    encoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    joint_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if args.aux:
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


if args.separate_discriminator:
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    print('Discriminator')
    discriminator.summary()

print('Encoder')
encoder.summary()
print('Generator')
generator.summary()

#
# Define losses
#

if args.use_augmented_variance_loss:
    train_reg_loss = losses.augmented_variance_loss
else:
    train_reg_loss = losses.reg_loss


if args.neg_dataset is not None:
    l_reg_neg = train_reg_loss(zn_mean, zn_log_var)

l_reg_z = train_reg_loss(z_mean, z_log_var)
l_reg_zr_ng = train_reg_loss(zr_mean_ng, zr_log_var_ng)
l_reg_zpp_ng = train_reg_loss(zpp_mean_ng, zpp_log_var_ng)
l_reg_zd = train_reg_loss(zd_mean, zd_log_var)
l_reg_zr = train_reg_loss(zr_mean, zr_log_var)
l_reg_zpp = train_reg_loss(zpp_mean, zpp_log_var)


reconst_loss = losses.reconstruction_loss(encoder_input, xr)

rec_loss_per_sample = losses.reconstruction_loss(encoder_input, xr)
l_ae = tf.reduce_mean(rec_loss_per_sample)

rec_loss_2_per_sample = losses.reconstruction_loss(encoder_input, xr_latent)
l_ae2 = tf.reduce_mean(rec_loss_2_per_sample)

gen_rec_loss_per_sample = losses.reconstruction_loss(zpp_gen, xgr)

if args.neg_dataset is not None:
    neg_rec_loss_per_sample = losses.reconstruction_loss(neg_input, xnr)
    l_ae_neg = tf.reduce_mean(neg_rec_loss_per_sample)

    neg_rec_loss_2_per_sample = losses.reconstruction_loss(neg_input, xnr_latent)
    l_ae_neg2 = tf.reduce_mean(neg_rec_loss_2_per_sample)

zpp_gradients = tf.gradients(l_reg_zpp, [zpp_gen])[0]

assert args.gradreg == 0.0, "Not implemented"


if args.margin_inf or args.m < 0.:
    margin_variable = tf.Variable(0., trainable=False)
    margin_update_op = tf.assign(margin_variable, margin_variable + 1/1000)
    margin = 10 * tf.math.log(margin_variable + 1)
else:
    margin = tf.Variable(args.m, trainable=False, dtype=tf.float32)
    margin_update_op = tf.assign(margin, margin)

if args.neg_prior:
    assert (args.priors_means_same_coords < args.latent_dim), "Number of same mean coordinates of 1st and 2nd prior should be smaller than dimension of latent code."
    neg_prior_mean = args.neg_prior_mean_coeff * tf.concat((tf.zeros(shape=(args.batch_size, args.priors_means_same_coords)),
                                                            tf.ones(shape=(args.batch_size, args.latent_dim - args.priors_means_same_coords))),
                                                            axis=1)
    if args.mml:
        discriminator_loss = args.reg_lambda * l_reg_zd
    else:
        l_reg_zr_ng = train_reg_loss(zr_mean_ng - neg_prior_mean, zr_log_var_ng)
        l_reg_zpp_ng = train_reg_loss(zpp_mean_ng - neg_prior_mean, zpp_log_var_ng)
        discriminator_loss = args.reg_lambda * l_reg_zd + args.alpha_reconstructed * l_reg_zr_ng + args.alpha_generated * l_reg_zpp_ng
else:
    discriminator_loss = args.reg_lambda * l_reg_zd + args.alpha_reconstructed * K.maximum(0., margin - l_reg_zr_ng) + args.alpha_generated * K.maximum(0., margin - l_reg_zpp_ng)

if args.neg_dataset is not None:
    if args.neg_prior:
        if args.mml:
            l_reg_neg = train_reg_loss(zn_mean - neg_prior_mean, zn_log_var)
            discriminator_loss += args.alpha_neg * l_reg_neg - args.beta_neg * l_ae_neg
        else:
            l_reg_neg = train_reg_loss(zn_mean - neg_prior_mean, zn_log_var)
            discriminator_loss += args.alpha_neg * l_reg_neg + args.beta_neg * l_ae_neg
    else:
        discriminator_loss +=  args.alpha_neg * K.maximum(0., margin - l_reg_neg) + args.beta_neg * l_ae_neg

if args.random_images_as_negative:
    zn_mean, zn_log_var = encoder(tf.clip_by_value(tf.abs(tf.random_normal( [args.batch_size] + list(args.original_shape) )), 0.0, 1.0))
    l_reg_noise = train_reg_loss(zn_mean, zn_log_var)
    discriminator_loss += args.reg_lambda * K.maximum(0., margin - l_reg_noise)

if args.fixed_gen_as_negative:
    fixed_gen_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='fixed_gen_input')
    z_fg_mean, z_fg_log_var = discriminator(fixed_gen_input)
    if args.neg_prior:
        l_reg_fixed_gen = train_reg_loss(z_fg_mean - neg_prior_mean, z_fg_log_var)
        discriminator_loss += args.alpha_fixed_gen * l_reg_fixed_gen
    else:
        l_reg_fixed_gen = train_reg_loss(z_fg_mean, z_fg_log_var)
        discriminator_loss += args.alpha_fixed_gen * K.maximum(0., margin - l_reg_fixed_gen)
    fixed_gen_index = 0
    if args.fixed_negatives_npy is not None:
        fixed_gen_np = np.load(args.fixed_negatives_npy)
        print('Fixed negatives loaded from {}'.format(args.fixed_negatives_npy))
    else:
        fixed_gen_np = np.zeros([args.fixed_gen_num] + list(args.original_shape))

if args.separate_discriminator:
    encoder_l_adv = args.reg_lambda * l_reg_z
else:
    encoder_l_adv = discriminator_loss

encoder_loss = encoder_l_adv + args.beta * l_ae

eubo_pos_loss = losses.eubo_loss_fn(z, z_mean, z_log_var, rec_loss_2_per_sample, args.cubo)
eubo_gen_loss = losses.eubo_loss_fn(zg, zg_mean, zg_log_var, gen_rec_loss_per_sample, args.cubo)
if args.neg_dataset is not None:
    eubo_neg_loss = losses.eubo_loss_fn(zn, zn_mean, zn_log_var, neg_rec_loss_2_per_sample, args.cubo)
else:
    eubo_neg_loss = tf.constant(0.0)

encoder_loss += args.eubo_lambda * eubo_pos_loss + args.eubo_gen_lambda * eubo_gen_loss
if args.neg_dataset is not None:
    encoder_loss += args.eubo_neg_lambda * eubo_neg_loss


if args.generator_adversarial_loss:
    generator_l_adv = args.alpha_reconstructed * l_reg_zr + args.alpha_generated * l_reg_zpp
    generator_loss = generator_l_adv + args.beta * l_ae2
else:
    generator_l_adv = 0.0
    generator_loss = args.beta * l_ae2

    if args.mml:
        generator_loss += -args.beta_neg * l_ae_neg2
    else:
        generator_loss += args.beta_neg * l_ae_neg2
if args.neg_dataset is not None:
    generator_loss += args.eubo_neg_lambda * eubo_neg_loss

generator_loss += args.eubo_lambda * eubo_pos_loss + args.eubo_gen_lambda * eubo_gen_loss

if args.aux:
    aux_y = Input(batch_shape=(args.batch_size, transformer.n_transforms), name='aux_y')

    aux_dense = Dense(transformer.n_transforms)
    aux_head = aux_dense(aux_encoder(aux_input))
    aux_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=aux_y, logits=aux_head))
    encoder_loss += 10.0*aux_loss 

    aux_head_gen = aux_dense(aux_encoder(zpp_gen))
    aux_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=aux_y, logits=aux_head_gen))
    generator_loss += 10.0*aux_loss_gen
else:
    aux_loss = tf.constant(0.0)

joint_loss = generator_l_adv + encoder_loss

#
# Define training step operations
#

encoder_params = encoder.trainable_weights
generator_params = generator.trainable_weights
if args.trained_gamma:
    generator_params.append(log_gamma)
if args.aux:
    encoder_params.extend(aux_dense.trainable_weights)


if args.separate_discriminator:
    discriminator_params = discriminator.trainable_weights
    discriminator_grads = discriminator_optimizer.compute_gradients(discriminator_loss, var_list=discriminator_params)
    discriminator_apply_grads_op = discriminator_optimizer.apply_gradients(discriminator_grads)
    for v in discriminator_params:
        tf.summary.histogram(v.name, v)

encoder_grads = optimizer.compute_gradients(encoder_loss, var_list=encoder_params)
if args.gradient_clipping:
    enc_gradients, enc_variables = zip(*encoder_grads)
    enc_gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in enc_gradients]
    encoder_grads = zip(enc_gradients, enc_variables)
encoder_apply_grads_op = optimizer.apply_gradients(encoder_grads)

generator_grads = optimizer.compute_gradients(generator_loss, var_list=generator_params)
if args.gradient_clipping:
    gen_gradients, gen_variables = zip(*generator_grads)
    gen_gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in gen_gradients]
    generator_grads = zip(gen_gradients, gen_variables)
generator_apply_grads_op = optimizer.apply_gradients(generator_grads, global_step=global_step)

if args.aux:
    aux_grads = aux_optimizer.compute_gradients(aux_loss)
    aux_apply_grads_op = aux_optimizer.apply_gradients(aux_grads, global_step=global_step)

joint_grads = joint_optimizer.compute_gradients(joint_loss)
joint_apply_grads_op = joint_optimizer.apply_gradients(joint_grads, global_step=global_step)


for v in encoder_params:
    tf.summary.histogram(v.name, v)
for v in generator_params:
    tf.summary.histogram(v.name, v)
summary_op = tf.summary.merge_all()

#
# Main loop
#


print('Start session')
global_iters = 0
start_epoch = 0

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run([init, train_iterator_init_op, test_iterator_init_op_a, test_iterator_init_op_b, fixed_iterator_init_op])
    if args.neg_dataset is not None:
        session.run([neg_iterator_init_op, neg_test_iterator_init_op])

    if args.save:
        summary_writer = tf.summary.FileWriter(args.prefix+"/", graph=tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=None)
    if args.model_path is not None and tf.train.checkpoint_exists(args.model_path):
        saver.restore(session, tf.train.latest_checkpoint(args.model_path))
        print('Model restored from ' + args.model_path)
        ckpt = tf.train.get_checkpoint_state(args.model_path)
        global_iters = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        start_epoch = (global_iters * args.batch_size) // args.train_size
    print('Global iters: ', global_iters)


    for iteration in range(iterations):
        epoch = global_iters * args.batch_size // args.train_size
        global_iters += 1

        x, _, margin_np = session.run([train_next, margin_update_op, margin])
        z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
        z_x, x_r, x_p = session.run([z, xr, generator_s_output], feed_dict={encoder_input: x, generator_s_input: z_p})

        if args.aux:
            transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x[0:4]))
            aux_y_np = to_categorical(transformations_inds)
            x_transformed = transformer.transform_batch(np.repeat(x[0:4], transformer.n_transforms, axis=0), transformations_inds)

        train_feed_dict = {encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p}
        if args.aux:
            train_feed_dict[aux_input] = x_transformed[:args.batch_size]
            train_feed_dict[aux_y] = aux_y_np[:args.batch_size]
        if args.neg_dataset is not None:
            x_n, = session.run([neg_next])
            z_n = session.run(zn, feed_dict={neg_input: x_n})
            train_feed_dict[neg_input] = x_n
            train_feed_dict[neg_latent_input] = z_n
        if args.fixed_gen_as_negative:
            if fixed_gen_index + args.batch_size > args.fixed_gen_num:
                fixed_gen_index = 0
            if epoch <= args.fixed_gen_max_epoch and args.fixed_negatives_npy is None:
                z_fg = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
                x_fg = session.run([generator_output], feed_dict={generator_input: z_fg})[0]
                fixed_gen_np[fixed_gen_index:(fixed_gen_index+args.batch_size), :, :, :] = x_fg
            else:
                x_fg = fixed_gen_np[fixed_gen_index:(fixed_gen_index+args.batch_size), :, :, :]
            fixed_gen_index += args.batch_size
            _ = session.run([encoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
            _ = session.run([generator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
        elif args.joint_training:
            _ = session.run([joint_apply_grads_op], feed_dict=train_feed_dict)
        else:
            _ = session.run([encoder_apply_grads_op], feed_dict=train_feed_dict)
            _ = session.run([generator_apply_grads_op], feed_dict=train_feed_dict)

        if args.separate_discriminator:
            if args.fixed_gen_as_negative:
                _ = session.run([discriminator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
            else:
                _ = session.run([discriminator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})

        if global_iters % args.frequency == 0 and args.save:
            summary, = session.run([summary_op], feed_dict={encoder_input: x})
            summary_writer.add_summary(summary, global_iters)

        if (global_iters % args.frequency) == 0:
            if args.fixed_gen_as_negative:
                eubo_loss_np, eubo_neg_loss_np, eubo_gen_loss_np, gamma_np, aux_loss_np, enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, l_reg_zd_np, disc_loss_np, l_reg_z_fixed_gen_np = \
                    session.run([eubo_pos_loss, eubo_neg_loss, eubo_gen_loss, gamma, aux_loss, encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp, learning_rate, l_reg_zd, discriminator_loss, l_reg_fixed_gen],
                                feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg}) #, aux_input: x_transformed[:args.batch_size], aux_y: aux_y_np[:args.batch_size]})
                neptune.send_metric('l_reg_fixed_gen', x=global_iters, y=l_reg_z_fixed_gen_np)
            else:
                eubo_loss_np, eubo_neg_loss_np, eubo_gen_loss_np, gamma_np, aux_loss_np, enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, l_reg_zd_np, disc_loss_np = \
                    session.run([eubo_pos_loss, eubo_neg_loss, eubo_gen_loss, gamma, aux_loss, encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp, learning_rate, l_reg_zd, discriminator_loss],
                                feed_dict=train_feed_dict)

            neptune.send_metric('disc_loss', x=global_iters, y=disc_loss_np)
            neptune.send_metric('l_reg_zd', x=global_iters, y=l_reg_zd_np)
            neptune.send_metric('enc_loss', x=global_iters, y=enc_loss_np)
            neptune.send_metric('l_ae', x=global_iters, y=enc_l_ae_np)
            neptune.send_metric('l_reg_z', x=global_iters, y=l_reg_z_np)
            neptune.send_metric('generator_loss', x=global_iters, y=generator_loss_np)
            neptune.send_metric('dec_l_ae', x=global_iters, y=dec_l_ae_np)
            neptune.send_metric('l_reg_zr', x=global_iters, y=l_reg_zr_np)
            neptune.send_metric('l_reg_zpp', x=global_iters, y=l_reg_zpp_np)
            neptune.send_metric('lr', x=global_iters, y=lr_np)
            neptune.send_metric('margin', x=global_iters, y=margin_np)
            neptune.send_metric('aux', x=global_iters, y=aux_loss_np)
            neptune.send_metric('gamma', x=global_iters, y=gamma_np)

            neptune.send_metric('eubo_loss', x=global_iters, y=eubo_loss_np)
            neptune.send_metric('eubo_neg_loss', x=global_iters, y=eubo_neg_loss_np)
            neptune.send_metric('eubo_gen_loss', x=global_iters, y=eubo_gen_loss_np)

            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch+1, args.nb_epoch, iteration+1, iterations))
            print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}, lr={}'.format(enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, lr_np))
            print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}, lr={}'.format(generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np))
            print(' Disc_loss: {}, l_reg_zd: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}'.format(disc_loss_np, l_reg_zd_np, l_reg_zr_ng_np, l_reg_zpp_ng_np))

        if global_iters % iterations_per_epoch == 0:
            _ = session.run([test_iterator_init_op_a, test_iterator_init_op_b])
            _ = utils.save_output(session, '_'.join([args.prefix, args.dataset]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: fixed_next}), OrderedDict({"train_mean": z_mean, "train_log_var": z_log_var, "train_reconstloss": reconst_loss}), args.latent_cloud_size, save=args.save)
            a_result_dict = utils.save_output(session, '_'.join([args.prefix, args.test_dataset_a]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_a}), OrderedDict({"test_a_mean": z_mean, "test_a_log_var": z_log_var, "test_a_reconstloss": reconst_loss}), test_size_a, save=args.save)
            b_result_dict = utils.save_output(session, '_'.join([args.prefix, args.test_dataset_b]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_b}), OrderedDict({"test_b_mean": z_mean, "test_b_log_var": z_log_var, "test_b_reconstloss": reconst_loss}), test_size_b, save=args.save)
            if args.neg_dataset is not None:
                neg_result_dict = utils.save_output(session, '_'.join([args.prefix, args.neg_dataset]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: neg_test_next}), OrderedDict({"test_neg_mean": z_mean, "test_neg_log_var": z_log_var, "test_neg_reconstloss": reconst_loss}), neg_test_size, save=args.save)
            if args.alpha_generated > 0.0:
                sampled_latent_tensor = tf.random_normal([args.batch_size, args.latent_dim])
                _ = utils.save_output(session, '_'.join([args.prefix, args.dataset]), epoch, global_iters, args.batch_size, OrderedDict({sampled_latent_input: sampled_latent_tensor}), OrderedDict({"adv_gen_mean": zpp_mean, "adv_gen_log_var": zpp_log_var}), args.latent_cloud_size, save=args.save)

        if (global_iters % iterations_per_epoch == 0) and args.save_fixed_gen and ((epoch+1 <= 10) or ((epoch+1)%10 == 0)):
            z_fixed_gen = tf.random_normal([args.batch_size, args.latent_dim])
            _ = utils.save_output(session, '_'.join([args.prefix, args.dataset]), epoch, global_iters, args.batch_size, OrderedDict({generator_input: z_fixed_gen}), OrderedDict({"fixed_gen": generator_output}), args.fixed_gen_num, save=args.save)

        if ((global_iters % iterations_per_epoch == 0) and (epoch + 1) % 10 == 0):

            _ = session.run([test_iterator_init_op_a, test_iterator_init_op_b])
            xt_a, xt_b = session.run([test_next_a, test_next_b])
            xt_a_r, = session.run([xr], feed_dict={encoder_input: xt_a})
            xt_b_r, = session.run([xr], feed_dict={encoder_input: xt_b})

            def make_observations(data):
                return data

            n_x = 5
            n_y = min(args.batch_size // n_x, 50)
            print('Save original images.')

            orig_img = utils.plot_images(np.transpose(make_observations(x), (0, 2, 3, 1)), n_x, n_y, "{}_original_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None, save=args.save)
            print('Save generated images.')
            gen_img = utils.plot_images(np.transpose(make_observations(x_p), (0, 2, 3, 1)), n_x, n_y, "{}_sampled_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None, save=args.save)
            print('Save reconstructed images.')
            rec_img = utils.plot_images(np.transpose(make_observations(x_r), (0, 2, 3, 1)), n_x, n_y, "{}_reconstructed_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None, save=args.save)
            print('Save A test images.')
            test_a_img = utils.plot_images(np.transpose(make_observations(xt_a_r), (0, 2, 3, 1)), n_x, n_y, "{}_test_a_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None, save=args.save)
            print('Save B test images.')
            test_b_img = utils.plot_images(np.transpose(make_observations(xt_b_r), (0, 2, 3, 1)), n_x, n_y, "{}_test_b_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None, save=args.save)

            neptune.send_image('original', orig_img)
            neptune.send_image('generated', gen_img)
            neptune.send_image('reconstruction', rec_img)
            neptune.send_image('test_a', test_a_img)
            neptune.send_image('test_b', test_b_img)
            if args.neg_dataset:
                print('Save negative images.')
                neg_img = utils.plot_images(np.transpose(make_observations(x_n), (0, 2, 3, 1)), n_x, n_y, "{}_negative_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None, save=args.save)
                neptune.send_image('train_neg', neg_img)

            if args.fixed_gen_as_negative:
                print('Save fixed generated images used as negative samples.')
                fixed_gen_as_neg = utils.plot_images(np.transpose(x_fg, (0, 2, 3, 1)), n_x, n_y, "{}_fixed_gen_as_negatives_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None, save=args.save)
                neptune.send_image('fixed_gen_as_negatives', fixed_gen_as_neg)


        if ((global_iters % iterations_per_epoch == 0) and ((epoch + 1) % 10 == 0)):
            if args.model_path is not None and args.save:
                saved = saver.save(session, args.model_path + "/model", global_step=global_iters)
                print('Saved model to ' + saved)

        if ((global_iters % iterations_per_epoch == 0) and args.oneclass_eval):

            def kldiv(mean, log_var):
                return 0.5 * np.sum( - log_var + np.square(mean) + np.exp(log_var) - 1, axis=-1)

            def compare(a_result_dict, b_result_dict, a_name, b_name, postfix=""):
                kl_a = kldiv( np.array(a_result_dict['test_a_mean']), np.array(a_result_dict['test_a_log_var']))
                kl_b = kldiv( np.array(b_result_dict['test_b_mean']), np.array(b_result_dict['test_b_log_var']))
                mean_a = np.mean(a_result_dict['test_a_mean'], axis=1)
                mean_b = np.mean(b_result_dict['test_b_mean'], axis=1)
                rec_a = a_result_dict['test_a_reconstloss']
                rec_b = b_result_dict['test_b_reconstloss']
                l2_mean_a = np.linalg.norm(a_result_dict['test_a_mean'], axis=1)
                l2_mean_b = np.linalg.norm(b_result_dict['test_b_mean'], axis=1)
                l2_var_a = np.linalg.norm(a_result_dict['test_a_log_var'], axis=1)
                l2_var_b = np.linalg.norm(b_result_dict['test_b_log_var'], axis=1)
                nll_a = kl_a + rec_a
                nll_b = kl_b + rec_b
                nllwrl_a = np.float32(args.reg_lambda) * kl_a + rec_a
                nllwrl_b = np.float32(args.reg_lambda) * kl_b + rec_b

                original_dim = np.float32(np.prod(args.original_shape))
                bpd_a = nll_a / original_dim
                bpd_b = nll_b / original_dim
                bpdwrl_a = nllwrl_a / original_dim
                bpdwrl_b = nllwrl_b / original_dim

                normed_nll_a = kl_a + (rec_a / original_dim)
                normed_nll_b = kl_b + (rec_b / original_dim)

                neptune.send_metric('test_mean_a{}'.format(postfix), x=global_iters, y=np.mean(mean_a))
                neptune.send_metric('test_mean_b{}'.format(postfix), x=global_iters, y=np.mean(mean_b))
                neptune.send_metric('test_var_a{}'.format(postfix), x=global_iters, y=np.mean(np.exp(a_result_dict['test_a_log_var']), axis=(0,1)))
                neptune.send_metric('test_var_b{}'.format(postfix), x=global_iters, y=np.mean(np.exp(b_result_dict['test_b_log_var']), axis=(0,1)))
                neptune.send_metric('test_rec_a{}'.format(postfix), x=global_iters, y=np.mean(rec_a))
                neptune.send_metric('test_rec_b{}'.format(postfix), x=global_iters, y=np.mean(rec_b))
                neptune.send_metric('test_kl_a{}'.format(postfix), x=global_iters, y=np.mean(kl_a))
                neptune.send_metric('test_kl_b{}'.format(postfix), x=global_iters, y=np.mean(kl_b))

                auc_kl = roc_auc_score(np.concatenate([np.zeros_like(kl_a), np.ones_like(kl_b)]), np.concatenate([kl_a, kl_b]))
                auc_mean = roc_auc_score(np.concatenate([np.zeros_like(mean_a), np.ones_like(mean_b)]), np.concatenate([mean_a, mean_b]))
                auc_rec = roc_auc_score(np.concatenate([np.zeros_like(rec_a), np.ones_like(rec_b)]), np.concatenate([rec_a, rec_b]))
                auc_l2_mean = roc_auc_score(np.concatenate([np.zeros_like(l2_mean_a), np.ones_like(l2_mean_b)]), np.concatenate([l2_mean_a, l2_mean_b]))
                auc_l2_var = roc_auc_score(np.concatenate([np.zeros_like(l2_var_a), np.ones_like(l2_var_b)]), np.concatenate([l2_var_a, l2_var_b]))
                auc_nll = roc_auc_score(np.concatenate([np.zeros_like(nll_a), np.ones_like(nll_b)]), np.concatenate([nll_a, nll_b]))
                auc_normed_nll = roc_auc_score(np.concatenate([np.zeros_like(normed_nll_a), np.ones_like(normed_nll_b)]), np.concatenate([normed_nll_a, normed_nll_b]))
                auc_nllwrl = roc_auc_score(np.concatenate([np.zeros_like(nllwrl_a), np.ones_like(nllwrl_b)]), np.concatenate([nllwrl_a, nllwrl_b]))

                neptune.send_metric('auc_kl_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_kl)
                neptune.send_metric('auc_mean_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_mean)
                neptune.send_metric('auc_rec_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_rec)
                neptune.send_metric('auc_l2_mean_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_l2_mean)
                neptune.send_metric('auc_l2_var_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_l2_var)
                neptune.send_metric('auc_neglog_likelihood_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_nll)
                neptune.send_metric('auc_bpd{}'.format(postfix), x=global_iters, y=auc_nll)
                neptune.send_metric('auc_normed_nll{}'.format(postfix), x=global_iters, y=auc_normed_nll)
                neptune.send_metric('auc_nllwrl{}'.format(postfix), x=global_iters, y=auc_nllwrl)

                neptune.send_metric('test_bpd_a{}'.format(postfix), x=global_iters, y=np.mean(bpd_a))
                neptune.send_metric('test_bpd_b{}'.format(postfix), x=global_iters, y=np.mean(bpd_b))

                if postfix == "":
                    neptune.send_metric('auc', x=global_iters, y=auc_nll)
                return kl_a, kl_b, rec_a, rec_b

            kl_a, kl_b, rec_a, rec_b = compare(a_result_dict, b_result_dict, args.test_dataset_a, args.test_dataset_b, "")

        if args.save and (global_iters % iterations_per_epoch == 0) and ((epoch + 1) % 10 == 0):
            np.savez("{}_kl_epoch{}_iter{}".format(args.prefix, epoch+1, global_iters), labels=np.concatenate([np.zeros_like(kl_a), np.ones_like(kl_b)]), kl=np.concatenate([kl_a, kl_b]))
            np.savez("{}_rec_epoch{}_iter{}".format(args.prefix, epoch+1, global_iters), labels=np.concatenate([np.zeros_like(rec_a), np.ones_like(rec_b)]), rec=np.concatenate([rec_a, rec_b]))

    neptune.stop()
    if args.model_path is not None and args.save:
        saved = saver.save(session, args.model_path + "/model", global_step=global_iters)
        print('Saved model to ' + saved)
