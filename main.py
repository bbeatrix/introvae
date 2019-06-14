import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras, keras.backend as K

from keras.layers import Activation, Input, Dense, Flatten
from keras.models import Model

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

#
# Config
#

# set random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

print('Keras version: ', keras.__version__)
print('Tensorflow version: ', tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
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

#if args.dataset == 'cifar10':
#    ds = data.create_cifar10_unsup_dataset(args.batch_size, args.train_size, args.test_size, args.latent_cloud_size, args.normal_class, args.gcnorm, args.augment)
#    train_data, train_placeholder, train_dataset, train_iterator, train_iterator_init_op, train_next = ds[0]
#    test_data, test_placeholder, test_dataset, test_iterator, test_iterator_init_op, test_next = ds[1]
#    fixed_data, fixed_placeholder, fixed_dataset, fixed_iterator, fixed_iterator_init_op, fixed_next = ds[2]
#else:
#    train_dataset, train_iterator, train_iterator_init_op, train_next \
#         = data.create_dataset(os.path.join(data_path, "train/*.npy"), args.batch_size, args.train_size)
#    test_dataset, test_iterator, test_iterator_init_op, test_next \
#         = data.create_dataset(os.path.join(data_path, "test/*.npy"), args.batch_size, args.test_size)
#    fixed_dataset, fixed_iterator, fixed_iterator_init_op, fixed_next \
#         = data.create_dataset(os.path.join(data_path, "train/*.npy"), args.batch_size, args.latent_cloud_size)

train_data, train_iterator, train_iterator_init_op, train_next = data.get_dataset(args.dataset, tfds.Split.TRAIN, args.batch_size, train_size, args.augment, args.normal_class, add_obs_noise=args.add_obs_noise)
fixed_data, fixed_iterator, fixed_iterator_init_op, fixed_next = data.get_dataset(args.dataset, tfds.Split.TRAIN, args.batch_size, args.latent_cloud_size, args.augment, args.normal_class, add_obs_noise=args.add_obs_noise)
test_data_a, test_iterator_a, test_iterator_init_op_a, test_next_a = data.get_dataset(args.test_dataset_a, tfds.Split.TEST, args.batch_size, test_size_a, args.augment, args.normal_class, outliers=False, add_obs_noise=args.add_obs_noise)
test_data_b, test_iterator_b, test_iterator_init_op_b, test_next_b = data.get_dataset(args.test_dataset_b, tfds.Split.TEST, args.batch_size, test_size_b, args.augment, args.normal_class, outliers=True, add_obs_noise=args.add_obs_noise)

if args.neg_dataset is not None:
    neg_train_size = args.neg_train_size
    neg_test_size = args.neg_test_size
    neg_data, neg_iterator, neg_iterator_init_op, neg_next = data.get_dataset(args.neg_dataset, tfds.Split.TRAIN, args.batch_size, neg_train_size, args.augment, add_obs_noise=args.add_obs_noise)
    neg_test_data, neg_test_iterator, neg_test_iterator_init_op, neg_test_next = data.get_dataset(args.neg_dataset, tfds.Split.TEST, args.batch_size, neg_test_size, args.augment, add_obs_noise=args.add_obs_noise)


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
    encoder_layers = model.encoder_layers_baseline_mnist(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.encoder_wd, args.seed)
    generator_layers = model.generator_layers_baseline_mnist(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.generator_wd, args.seed)

elif args.model_architecture == 'dcgan':
    encoder_layers = model.encoder_layers_dcgan(args.shape, args.base_filter_num, args.encoder_use_bn, args.encoder_wd)
    generator_layers = model.generator_layers_dcgan(args.shape, args.base_filter_num, args.encoder_use_bn, args.generator_wd)
elif args.model_architecture == 'dcgan_univ':
    encoder_layers = model.encoder_layers_dcgan_univ(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.encoder_wd)
    generator_layers = model.generator_layers_dcgan_univ(args.shape, args.n_channels, args.base_filter_num, args.encoder_use_bn, args.generator_wd)
else:
    encoder_layers = model.encoder_layers_introvae(args.shape, args.base_filter_num, args.encoder_use_bn)
    generator_layers = model.generator_layers_introvae(args.shape, args.base_filter_num, args.generator_use_bn)


encoder_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='encoder_input')

generator_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='generator_input')
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
for layer in generator_layers:
    generator_output = layer(generator_output)

generator_output = model.add_observable_output(generator_output, args)
z, z_mean, z_log_var = model.add_sampling(encoder_output, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd)

if args.trained_gamma:
    log_gamma = tf.get_variable('log_gamma', [], tf.float32, tf.constant_initializer(value=args.initial_log_gamma))
else:
    log_gamma = tf.constant(args.initial_log_gamma)
gamma = tf.exp(log_gamma)

encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var])
generator = Model(inputs=generator_input, outputs=generator_output)
if args.aux:
    aux_encoder = Model(inputs=aux_input, outputs=[aux_encoder_output])


if args.separate_discriminator:
    discriminator_output = encoder_input
    for layer in encoder_layers:
        discriminator_output = layer(discriminator_output)
    _, zd_mean, zd_log_var = model.add_sampling(discriminator_output, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd)
    discriminator = Model(inputs=encoder_input, outputs=[zd_mean, zd_log_var])
else:
    discriminator = encoder
    zd_mean, zd_log_var = z_mean, z_log_var

xr = generator(z)
reconst_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='reconst_latent_input')
zr_mean, zr_log_var = discriminator(generator(reconst_latent_input))
zr_mean_ng, zr_log_var_ng = discriminator(tf.stop_gradient(generator(reconst_latent_input)))
xr_latent = generator(reconst_latent_input)
if args.neg_dataset is not None:
    zn_mean, zn_log_var = discriminator(neg_input)

sampled_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='sampled_latent_input')
zpp_gen = generator(sampled_latent_input)
zpp_mean, zpp_log_var = discriminator(zpp_gen)
#zpp_mean_ng, zpp_log_var_ng = discriminator(tf.stop_gradient(zpp_gen))
zpp_mean_ng, zpp_log_var_ng = discriminator(tf.stop_gradient(zpp_gen))

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


HALF_LOG_TWO_PI = 0.91893

def reconstruction_loss(x, xr):
    if args.obs_noise_model == 'bernoulli':
        return -tf.reduce_sum(x * tf.log(tf.maximum(xr, 1e-8)) + (1-x) * tf.log(tf.maximum(1-xr, 1e-8)), [1, 2, 3])
    else:
        return tf.reduce_sum(tf.square((x - xr) / gamma) / 2 + log_gamma + HALF_LOG_TWO_PI, [1, 2, 3])

reconst_loss = reconstruction_loss(encoder_input, xr)

#reconst_loss = K.mean(keras.objectives.mean_squared_error(encoder_input, xr), axis=(1,2))

l_ae = tf.reduce_mean(reconstruction_loss(encoder_input, xr))
l_ae2 = tf.reduce_mean(reconstruction_loss(encoder_input, xr_latent))

zpp_gradients = tf.gradients(l_reg_zpp, [zpp_gen])[0]

assert args.gradreg == 0.0, "Not implemented"


if args.margin_inf or args.m < 0.:
    margin_variable = tf.Variable(0., trainable=False)
    margin_update_op = tf.assign(margin_variable, margin_variable + 1/1000)
    margin = 10 * tf.math.log(margin_variable + 1)
    # margin = margin_variable
else:
    margin = tf.Variable(args.m, trainable=False, dtype=tf.float32)
    margin_update_op = tf.assign(margin, margin)

#encoder_l_adv = args.reg_lambda * l_reg_z + args.alpha * K.maximum(0., margin - l_reg_zr_ng) + args.alpha * K.maximum(0., margin - l_reg_zpp_ng)
discriminator_loss = args.reg_lambda * l_reg_zd + args.alpha_reconstructed * K.maximum(0., margin - l_reg_zr_ng) + args.alpha_generated * K.maximum(0., margin - l_reg_zpp_ng)
#discriminator_loss = args.reg_lambda * l_reg_zd + args.alpha_reconstructed * (1.0 / l_reg_zr_ng) + args.alpha_generated * (1.0 / l_reg_zpp_ng)

if args.neg_dataset is not None:
    discriminator_loss +=  args.alpha_neg * K.maximum(0., margin - l_reg_neg)
if args.random_images_as_negative:
    zn_mean, zn_log_var = encoder(tf.clip_by_value(tf.abs(tf.random_normal( [args.batch_size] + list(args.original_shape) )), 0.0, 1.0))
    l_reg_noise = train_reg_loss(zn_mean, zn_log_var)
    discriminator_loss += args.reg_lambda * K.maximum(0., margin - l_reg_noise)

if args.fixed_gen_as_negative:
    fixed_gen_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='fixed_gen_input')
    z_fg_mean, z_fg_log_var = encoder(fixed_gen_input)
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

encoder_loss = encoder_l_adv + args.beta * l_ae #+ args.gradreg * spectreg_loss
encoder1_loss = args.reg_lambda * l_reg_zd  + args.beta * l_ae # + args.gradreg * spectreg_loss
encoder2_loss = args.alpha_reconstructed * K.maximum(0., margin - l_reg_zr_ng) + args.alpha_generated * K.maximum(0., margin - l_reg_zpp_ng) + args.beta * l_ae # + args.gradreg * spectreg_loss


if args.generator_adversarial_loss:
    generator_l_adv = args.alpha_reconstructed * l_reg_zr + args.alpha_generated * l_reg_zpp
    generator_loss = generator_l_adv + args.beta * l_ae2 # + args.gradreg * spectreg_loss
else:
    generator_l_adv = 0.0
    generator_loss = args.beta * l_ae2 # + args.gradreg * spectreg_loss


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
encoder_apply_grads_op = optimizer.apply_gradients(encoder_grads)


#encoder1_grads = encoder_optimizer.compute_gradients(encoder1_loss, var_list=encoder_params)
#encoder1_apply_grads_op = encoder_optimizer.apply_gradients(encoder1_grads)

#encoder2_grads = encoder_optimizer.compute_gradients(encoder2_loss, var_list=encoder_params)
#encoder2_apply_grads_op = encoder_optimizer.apply_gradients(encoder2_grads)


generator_grads = optimizer.compute_gradients(generator_loss, var_list=generator_params)
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

    summary_writer = tf.summary.FileWriter(args.prefix+"/", graph=tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=None)
    if args.model_path is not None and tf.train.checkpoint_exists(args.model_path):
        saver.restore(session, tf.train.latest_checkpoint(args.model_path))
        print('Model restored from ' + args.model_path)
        ckpt = tf.train.get_checkpoint_state(args.model_path)
        global_iters = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        start_epoch = (global_iters * args.batch_size) // args.train_size
    print('Global iters: ', global_iters)

    #if args.oneclass_eval:
        #utils.save_kldiv(session, '_'.join([args.prefix, args.test_dataset_a]), start_epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_a}), OrderedDict({"mean": z_mean, "log_var": z_log_var}), args.test_size)
        #utils.save_kldiv(session, '_'.join([args.prefix, args.test_dataset_b]), start_epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_b}), OrderedDict({"mean": z_mean, "log_var": z_log_var}), args.test_size)
        #utils.oneclass_eval(args.normal_class, "{}_{}_epoch{}_iter{}.npy".format(args.prefix, 'kldiv', start_epoch, global_iters), margin_np)

    if not args.train:
        def search_opt_z(get_next_batch, z_update_iters=20, lr=0.1):
            encoder_zs = []
            new_zs = []

            temp = set(tf.all_variables())
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                new_z = tf.get_variable("z", [args.batch_size, args.latent_dim], dtype=tf.float32, initializer=tf.zeros_initializer)
            assign_encoder_z = tf.assign(new_z, z)
            x_decoded = generator(new_z)
            z_opt_loss = losses.mse_loss(encoder_input, x_decoded, args.original_shape)
            z_optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            z_grads = z_optimizer.compute_gradients(z_opt_loss, var_list=[new_z])
            z_apply_grads_op = z_optimizer.apply_gradients(z_grads)

            # newly created params should be initialized
            uninitialized_vars = set(session.run(tf.report_uninitialized_variables()))
            #print("uninitialized: \n", uninitialized_vars)
            session.run(tf.initialize_variables(set(tf.all_variables()) - temp)) # not the proper way but at least this works

            #list_of_variables = tf.all_variables()
            #uninitialized_vars0 = list(tf.get_variable(name) for name in uninitialized_vars)
            #print("uninitialized: \n", uninitialized_vars0)
            #session.run(tf.initialize_variables(uninitialized_vars0))
            #session.run(tf.initialize_variables([z_optimizer.get_slot(loss, name) for name in optim.get_slot_names()]))
            #uninitialized_vars = [v for v in tf.global_variables() if v.name.split(':')[0] in report_uninitialized_vars]
            #print("uninitialized vars: \n", uninitialized_vars)
            #session.run(tf.variables_initializer(tf.report_uninitialized_variables()))
            #session.run(tf.local_variables_initializer())
            # check tht there's no uninitialized var now
            #uninitialized_vars = set(session.run(tf.report_uninitialized_variables()))
            #print("uninitialized now: \n", uninitialized_vars)

            nb_batches = args.test_size // args.batch_size
            for i in range(nb_batches):
                x_batch = session.run(get_next_batch)
                encoder_z_np = session.run(z, feed_dict={encoder_input: x_batch})
                encoder_zs.extend(encoder_z_np)
                session.run(assign_encoder_z, feed_dict={encoder_input: x_batch})
                for i in range(z_update_iters):
                    session.run(z_apply_grads_op, feed_dict={encoder_input: x_batch})
                new_z_np = session.run(new_z)
                new_zs.extend(new_z_np)
            return encoder_zs, new_zs

        encoder_zs_a, new_zs_a = search_opt_z(test_next_a)
        encoder_zs_b, new_zs_b = search_opt_z(test_next_b)
        auc_old_z = roc_auc_score(np.concatenate([np.zeros_like(encoder_zs_a), np.ones_like(encoder_zs_b)]), np.linalg.norm(np.concatenate([encoder_zs_a, encoder_zs_b]), axis=1, keepdims=True))
        auc_new_z = roc_auc_score(np.concatenate([np.zeros_like(new_zs_a), np.ones_like(new_zs_b)]), np.linalg.norm(np.concatenate([new_zs_a, new_zs_b]), axis=1, keepdims=True))

        print("\n auc_old_z: {} \n".format(auc_old_z))
        print("\n auc_new_z: {} \n".format(auc_new_z))
        raise SystemExit("Exit intentionally before training.")

    for iteration in range(iterations):
        epoch = global_iters * args.batch_size // args.train_size
        global_iters += 1

        x, _, margin_np = session.run([train_next, margin_update_op, margin])
        z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
        z_x, x_r, x_p = session.run([z, xr, generator_output], feed_dict={encoder_input: x, generator_input: z_p})

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
            train_feed_dict[neg_input] = x_n
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
            pass
            #for j in range(1):
            #    #x, _, margin_np = session.run([train_next, margin_update_op, margin])
            #    #z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
            #    #z_x, x_r, x_p = session.run([z, xr, generator_output], feed_dict={encoder_input: x, generator_input: z_p})

            _ = session.run([encoder_apply_grads_op], feed_dict=train_feed_dict)
            #_ = session.run([encoder1_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, aux_input: x_transformed[:args.batch_size], aux_y: aux_y_np[:args.batch_size]})
            #_ = session.run([encoder2_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, aux_input: x_transformed[:args.batch_size], aux_y: aux_y_np[:args.batch_size]})

            #for j in range(1):
            #    #x, _, margin_np = session.run([train_next, margin_update_op, margin])
            #    #z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
            #    #z_x, x_r, x_p = session.run([z, xr, generator_output], feed_dict={encoder_input: x, generator_input: z_p})
            #    #_ = session.run([generator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p}) # , aux_input: x_transformed[:args.batch_size], aux_y: aux_y_np[:args.batch_size]
            _ = session.run([generator_apply_grads_op], feed_dict=train_feed_dict)

        #for  j in range(x.shape[0] // args.batch_size):
        #for j in range(3):
        #    _ = session.run([aux_apply_grads_op], feed_dict={encoder_input: x_transformed[j*args.batch_size:(j+1)*args.batch_size], aux_y: aux_y_np[j*args.batch_size:(j+1)*args.batch_size]})

        if args.separate_discriminator:
            if args.fixed_gen_as_negative:
                _ = session.run([discriminator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
            else:
                _ = session.run([discriminator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})

        if global_iters % args.frequency == 0:
            summary, = session.run([summary_op], feed_dict={encoder_input: x})
            summary_writer.add_summary(summary, global_iters)

        if (global_iters % args.frequency) == 0:
            if args.fixed_gen_as_negative:
                gamma_np, aux_loss_np, enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, l_reg_zd_np, disc_loss_np, l_reg_z_fixed_gen_np = \
                    session.run([gamma, aux_loss, encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp, learning_rate, l_reg_zd, discriminator_loss, l_reg_fixed_gen],
                                feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg, aux_input: x_transformed[:args.batch_size], aux_y: aux_y_np[:args.batch_size]})
                neptune.send_metric('l_reg_fixed_gen', x=global_iters, y=l_reg_z_fixed_gen_np)
            else:
                gamma_np, aux_loss_np, enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, l_reg_zd_np, disc_loss_np = \
                    session.run([gamma, aux_loss, encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp, learning_rate, l_reg_zd, discriminator_loss],
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

            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch+1, args.nb_epoch, iteration+1, iterations))
            print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}, lr={}'.format(enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, lr_np))
            print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}, lr={}'.format(generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np))
            print(' Disc_loss: {}, l_reg_zd: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}'.format(disc_loss_np, l_reg_zd_np, l_reg_zr_ng_np, l_reg_zpp_ng_np))

        if ((global_iters % iterations_per_epoch == 0) and args.save_latent):
            _ = session.run([test_iterator_init_op_a, test_iterator_init_op_b])
            _ = utils.save_output(session, '_'.join([args.prefix, args.dataset]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: fixed_next}), OrderedDict({"train_mean": z_mean, "train_log_var": z_log_var}), args.latent_cloud_size)
            a_result_dict = utils.save_output(session, '_'.join([args.prefix, args.test_dataset_a]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_a}), OrderedDict({"test_mean": z_mean, "test_log_var": z_log_var, "test_reconstloss": reconst_loss}), test_size_a, args.augment_avg_at_test, args.original_shape)
            b_result_dict = utils.save_output(session, '_'.join([args.prefix, args.test_dataset_b]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_b}), OrderedDict({"test_mean": z_mean, "test_log_var": z_log_var, "test_reconstloss": reconst_loss}), test_size_b, args.augment_avg_at_test, args.original_shape)
            if args.neg_dataset is not None:
                neg_result_dict = utils.save_output(session, '_'.join([args.prefix, args.neg_dataset]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: neg_test_next}), OrderedDict({"test_mean": z_mean, "test_log_var": z_log_var, "test_reconstloss": reconst_loss}), neg_test_size, args.augment_avg_at_test, args.original_shape)

        if (global_iters % iterations_per_epoch == 0) and args.save_fixed_gen and ((epoch+1 <= 10) or ((epoch+1)%10 == 0)):
            z_fixed_gen = tf.random_normal([args.batch_size, args.latent_dim])
            _ = utils.save_output(session, '_'.join([args.prefix, args.dataset]), epoch, global_iters, args.batch_size, OrderedDict({generator_input: z_fixed_gen}), OrderedDict({"fixed_gen": generator_output}), args.fixed_gen_num)

        if ((global_iters % iterations_per_epoch == 0) and args.save_latent and (epoch + 1) % 10 == 0):

            _ = session.run([test_iterator_init_op_a, test_iterator_init_op_b])
            xt_a, xt_b = session.run([test_next_a, test_next_b])
            xt_a_r, = session.run([xr], feed_dict={encoder_input: xt_a})
            xt_b_r, = session.run([xr], feed_dict={encoder_input: xt_b})

            def make_observations(data):
                return data

            n_x = 5
            n_y = min(args.batch_size // n_x, 50)
            print('Save original images.')

            orig_img = utils.plot_images(np.transpose(make_observations(x), (0, 2, 3, 1)), n_x, n_y, "{}_original_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save generated images.')
            gen_img = utils.plot_images(np.transpose(make_observations(x_p), (0, 2, 3, 1)), n_x, n_y, "{}_sampled_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save reconstructed images.')
            rec_img = utils.plot_images(np.transpose(make_observations(x_r), (0, 2, 3, 1)), n_x, n_y, "{}_reconstructed_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save A test images.')
            test_a_img = utils.plot_images(np.transpose(make_observations(xt_a_r), (0, 2, 3, 1)), n_x, n_y, "{}_test_a_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save B test images.')
            test_b_img = utils.plot_images(np.transpose(make_observations(xt_b_r), (0, 2, 3, 1)), n_x, n_y, "{}_test_b_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)

            neptune.send_image('original', orig_img)
            neptune.send_image('generated', gen_img)
            neptune.send_image('reconstruction', rec_img)
            neptune.send_image('test_a', test_a_img)
            neptune.send_image('test_b', test_b_img)

            if args.fixed_gen_as_negative:
                print('Save fixed generated images used as negative samples.')
                fixed_gen_as_neg = utils.plot_images(np.transpose(x_fg, (0, 2, 3, 1)), n_x, n_y, "{}_fixed_gen_as_negatives_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
                neptune.send_image('fixed_gen_as_negatives', fixed_gen_as_neg)


        if False and ((global_iters % iterations_per_epoch == 0) and ((epoch + 1) % 10 == 0)):
            if args.model_path is not None:
                saved = saver.save(session, args.model_path + "/model", global_step=global_iters)
                print('Saved model to ' + saved)

        if ((global_iters % iterations_per_epoch == 0) and args.oneclass_eval):

            def kldiv(mean, log_var):
                return 0.5 * np.sum( - log_var + np.square(mean) + np.exp(log_var) - 1, axis=-1)

            def compare(a_result_dict, b_result_dict, a_name, b_name, postfix=""):
                kl_a = kldiv( np.array(a_result_dict['test_mean']), np.array(a_result_dict['test_log_var']))
                kl_b = kldiv( np.array(b_result_dict['test_mean']), np.array(b_result_dict['test_log_var']))
                mean_a = np.mean(a_result_dict['test_mean'], axis=1)
                mean_b = np.mean(b_result_dict['test_mean'], axis=1)
                rec_a = a_result_dict['test_reconstloss']
                rec_b = b_result_dict['test_reconstloss']
                l2_mean_a = np.linalg.norm(a_result_dict['test_mean'], axis=1)
                l2_mean_b = np.linalg.norm(b_result_dict['test_mean'], axis=1)
                l2_var_a = np.linalg.norm(a_result_dict['test_log_var'], axis=1)
                l2_var_b = np.linalg.norm(b_result_dict['test_log_var'], axis=1)
                nll_a = kl_a + rec_a
                nll_b = kl_b + rec_b
                nllwb_a = np.float32(args.beta) * kl_a + rec_a
                nllwb_b = np.float32(args.beta) * kl_b + rec_b

                original_dim = np.float32(np.prod(args.original_shape))
                bpd_a = nll_a / original_dim
                bpd_b = nll_b / original_dim
                bpdwb_a = nllwb_a / original_dim
                bpdwb_b = nllwb_b / original_dim

                normed_nll_a = kl_a + (rec_a / original_dim)
                normed_nll_b = kl_b + (rec_b / original_dim)

                neptune.send_metric('test_mean_a{}'.format(postfix), x=global_iters, y=np.mean(mean_a))
                neptune.send_metric('test_mean_b{}'.format(postfix), x=global_iters, y=np.mean(mean_b))
                neptune.send_metric('test_var_a{}'.format(postfix), x=global_iters, y=np.mean(np.exp(a_result_dict['test_log_var']), axis=(0,1)))
                neptune.send_metric('test_var_b{}'.format(postfix), x=global_iters, y=np.mean(np.exp(b_result_dict['test_log_var']), axis=(0,1)))
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
                auc_normed_nll = roc_auc_score(np.concatenate([np.zeros_like(normed_nll_a), np.ones_like(normed_nll_a)]), np.concatenate([normed_nll_a, normed_nll_b]))
                auc_nllwb = roc_auc_score(np.concatenate([np.zeros_like(nllwb_a), np.ones_like(nllwb_a)]), np.concatenate([nllwb_a, nllwb_b]))

                neptune.send_metric('auc_kl_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_kl)
                neptune.send_metric('auc_mean_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_mean)
                neptune.send_metric('auc_rec_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_rec)
                neptune.send_metric('auc_l2_mean_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_l2_mean)
                neptune.send_metric('auc_l2_var_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_l2_var)
                neptune.send_metric('auc_neglog_likelihood_{}_vs_{}{}'.format(args.test_dataset_a, args.test_dataset_b, postfix), x=global_iters, y=auc_nll)
                neptune.send_metric('auc_bpd{}'.format(postfix), x=global_iters, y=auc_nll)
                neptune.send_metric('auc_normed_nll{}'.format(postfix), x=global_iters, y=auc_normed_nll)
                neptune.send_metric('auc_nllwb{}'.format(postfix), x=global_iters, y=auc_nllwb)

                neptune.send_metric('test_bpd_a{}'.format(postfix), x=global_iters, y=np.mean(bpd_a))
                neptune.send_metric('test_bpd_b{}'.format(postfix), x=global_iters, y=np.mean(bpd_b))


                if postfix == "":
                    neptune.send_metric('auc', x=global_iters, y=auc_nll)
                return kl_a, kl_b, rec_a, rec_b

            kl_a, kl_b, rec_a, rec_b = compare(a_result_dict, b_result_dict, args.test_dataset_a, args.test_dataset_b, "")
            if args.neg_dataset is not None:
                compare(a_result_dict, neg_result_dict, args.test_dataset_a, args.neg_dataset, "_neg")

        if (global_iters % iterations_per_epoch == 0) and ((epoch + 1) % 10 == 0):
            np.savez("{}_kl_epoch{}_iter{}".format(args.prefix, epoch+1, global_iters), labels=np.concatenate([np.zeros_like(kl_a), np.ones_like(kl_b)]), kl=np.concatenate([kl_a, kl_b]))
            np.savez("{}_rec_epoch{}_iter{}".format(args.prefix, epoch+1, global_iters), labels=np.concatenate([np.zeros_like(rec_a), np.ones_like(rec_b)]), rec=np.concatenate([rec_a, rec_b]))

    neptune.stop()
    if args.model_path is not None:
        saved = saver.save(session, args.model_path + "/model", global_step=global_iters)
        print('Saved model to ' + saved)

