import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras, keras.backend as K

from keras.layers import Input
from keras.models import Model

import os, sys, time
from collections import OrderedDict

import model, params, losses, utils, data

from sklearn.metrics import roc_auc_score

import neptune

args = params.getArgs()
print(args)

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

train_data, train_iterator, train_iterator_init_op, train_next = data.get_dataset(args.dataset, tfds.Split.TRAIN, args.batch_size, args.train_size)
fixed_data, fixed_iterator, fixed_iterator_init_op, fixed_next = data.get_dataset(args.dataset, tfds.Split.TRAIN, args.batch_size, args.latent_cloud_size)
test_data_a, test_iterator_a, test_iterator_init_op_a, test_next_a = data.get_dataset(args.test_dataset_a, tfds.Split.TEST, args.batch_size, args.test_size)
test_data_b, test_iterator_b, test_iterator_init_op_b, test_next_b = data.get_dataset(args.test_dataset_b, tfds.Split.TEST, args.batch_size, args.test_size)

args.n_channels = 3 if args.color else 1
args.original_shape = (args.n_channels, ) + args.shape

#
# Build networks
#

if args.model_architecture == 'deepsvdd':
    encoder_layers = model.encoder_layers_deepsvdd(args.shape, args.base_filter_num, args.encoder_use_bn)
    generator_layers = model.generator_layers_deepsvdd(args.shape, args.base_filter_num, args.generator_use_bn)
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

encoder_output = encoder_input
for layer in encoder_layers:
    encoder_output = layer(encoder_output)

generator_output = generator_input
for layer in generator_layers:
    generator_output = layer(generator_output)

z, z_mean, z_log_var = model.add_sampling(encoder_output, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd)

encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var])
generator = Model(inputs=generator_input, outputs=generator_output)

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

sampled_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='sampled_latent_input')
zpp_mean, zpp_log_var = discriminator(generator(sampled_latent_input))
zpp_mean_ng, zpp_log_var_ng = discriminator(tf.stop_gradient(generator(sampled_latent_input)))

global_step = tf.Variable(0, trainable=False)

starter_learning_rate = args.lr
if args.lr_schedule == 'exponential':
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.96, staircase=True)
else:
    learning_rate = tf.constant(args.lr)

encoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

l_reg_z = train_reg_loss(z_mean, z_log_var)
l_reg_zr_ng = train_reg_loss(zr_mean_ng, zr_log_var_ng)
l_reg_zpp_ng = train_reg_loss(zpp_mean_ng, zpp_log_var_ng)
l_reg_zd = train_reg_loss(zd_mean, zd_log_var)

reconst_loss = K.mean(keras.objectives.mean_squared_error(encoder_input, xr), axis=(1,2))

l_ae = losses.mse_loss(encoder_input, xr, args.original_shape)
l_ae2 = losses.mse_loss(encoder_input, xr_latent, args.original_shape)

z_mean_gradients = tf.gradients(z_mean * tf.random_normal((args.latent_dim,)), [encoder_input])[0]
z_log_var_gradients = tf.gradients(z_log_var * tf.random_normal((args.latent_dim,)), [encoder_input])[0]

spectreg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z_mean_gradients), axis=1))
spectreg_loss += tf.reduce_mean(tf.reduce_sum(tf.square(z_log_var_gradients), axis=1))
#spectreg_loss = tf.reduce_mean(spectreg_loss, axis=-1)

if args.margin_inf or args.m < 0.:
    margin_variable = tf.Variable(0., trainable=False)
    margin_update_op = tf.assign(margin_variable, margin_variable + 1/1000)
    margin = 10 * tf.math.log(margin_variable + 1)
    # margin = margin_variable
else:
    margin = tf.Variable(args.m, trainable=False, dtype=tf.float32)
    margin_update_op = tf.assign(margin, margin)

#encoder_l_adv = args.reg_lambda * l_reg_z + args.alpha * K.maximum(0., margin - l_reg_zr_ng) + args.alpha * K.maximum(0., margin - l_reg_zpp_ng)
discriminator_loss = args.reg_lambda * l_reg_zd + args.alpha * K.maximum(0., margin - l_reg_zr_ng) + args.alpha * K.maximum(0., margin - l_reg_zpp_ng)

if args.random_images_as_negative:
    zn_mean, zn_log_var = encoder(tf.clip_by_value(tf.abs(tf.random_normal( [args.batch_size] + list(args.original_shape) )), 0.0, 1.0))
    l_reg_noise = train_reg_loss(zn_mean, zn_log_var)
    discriminator_loss += args.reg_lambda * K.maximum(0., margin - l_reg_noise)

if args.fixed_gen_as_negative:
    fixed_gen_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='fixed_gen_input')
    z_fg_mean, z_fg_log_var = encoder(fixed_gen_input)
    l_reg_fixed_gen = train_reg_loss(z_fg_mean, z_fg_log_var)
    discriminator_loss += args.reg_lambda * K.maximum(0., margin - l_reg_fixed_gen)
    fixed_gen_np = np.zeros([args.fixed_gen_num] + list(args.original_shape))
    fixed_gen_index = 0

if args.separate_discriminator:
    encoder_l_adv = args.reg_lambda * l_reg_z
else:
    encoder_l_adv = discriminator_loss

encoder_loss = encoder_l_adv + args.beta * l_ae + args.gradreg * spectreg_loss

l_reg_zr = train_reg_loss(zr_mean, zr_log_var)
l_reg_zpp = train_reg_loss(zpp_mean, zpp_log_var)

if args.generator_adversarial_loss:
    generator_l_adv = args.alpha * l_reg_zr + args.alpha * l_reg_zpp
    generator_loss = generator_l_adv + args.beta * l_ae2 + args.gradreg * spectreg_loss
else:
    generator_loss = args.beta * l_ae2 + args.gradreg * spectreg_loss

#
# Define training step operations
#

encoder_params = encoder.trainable_weights
generator_params = generator.trainable_weights

if args.separate_discriminator:
    discriminator_params = discriminator.trainable_weights
    discriminator_grads = discriminator_optimizer.compute_gradients(discriminator_loss, var_list=discriminator_params)
    discriminator_apply_grads_op = discriminator_optimizer.apply_gradients(discriminator_grads)
    for v in discriminator_params:
        tf.summary.histogram(v.name, v)

encoder_grads = encoder_optimizer.compute_gradients(encoder_loss, var_list=encoder_params)
encoder_apply_grads_op = encoder_optimizer.apply_gradients(encoder_grads)

generator_grads = generator_optimizer.compute_gradients(generator_loss, var_list=generator_params)
generator_apply_grads_op = generator_optimizer.apply_gradients(generator_grads, global_step=global_step)

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

    for iteration in range(iterations):
        epoch = global_iters * args.batch_size // args.train_size
        global_iters += 1

        x, _, margin_np = session.run([train_next, margin_update_op, margin])
        z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
        z_x, x_r, x_p = session.run([z, xr, generator_output], feed_dict={encoder_input: x, generator_input: z_p})

        if args.fixed_gen_as_negative:
            if fixed_gen_index + args.batch_size > args.fixed_gen_num:
                fixed_gen_index = 0
            if epoch <= args.fixed_gen_max_epoch:
                z_fg = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
                x_fg = session.run([generator_output], feed_dict={generator_input: z_fg})[0]
                fixed_gen_np[fixed_gen_index:(fixed_gen_index+args.batch_size), :, :, :] = x_fg
            else:
                x_fg = fixed_gen_np[fixed_gen_index:(fixed_gen_index+args.batch_size), :, :, :]
            fixed_gen_index += args.batch_size
            _ = session.run([encoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
            _ = session.run([generator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
        else:
            _ = session.run([encoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
            _ = session.run([generator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})

        if args.separate_discriminator:
            if args.fixed_gen_as_negative:
                _ = session.run([discriminator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
            else:
                _ = session.run([discriminator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})

        if global_iters % 10 == 0:
            summary, = session.run([summary_op], feed_dict={encoder_input: x})
            summary_writer.add_summary(summary, global_iters)

        if (global_iters % args.frequency) == 0:
            if args.fixed_gen_as_negative:
                enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, l_reg_zd_np, disc_loss_np = \
                    session.run([encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp, learning_rate, l_reg_zd, discriminator_loss],
                                feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p, fixed_gen_input: x_fg})
            else:
                enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, l_reg_zd_np, disc_loss_np = \
                    session.run([encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp, learning_rate, l_reg_zd, discriminator_loss],
                                feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})

            neptune.send_metric('disc_loss', x=global_iters, y=disc_loss_np)
            neptune.send_metric('l_reg_zd', x=global_iters, y=l_reg_zd_np)
            neptune.send_metric('enc_loss', x=global_iters, y=enc_loss_np)
            neptune.send_metric('l_ae', x=global_iters, y=enc_l_ae_np)
            neptune.send_metric('l_reg_z', x=global_iters, y=l_reg_z_np)
            neptune.send_metric('l_reg_zr_ng', x=global_iters, y=l_reg_zr_ng_np)
            neptune.send_metric('l_reg_zpp_ng', x=global_iters, y=l_reg_zpp_ng_np)
            neptune.send_metric('generator_loss', x=global_iters, y=generator_loss_np)
            neptune.send_metric('dec_l_ae', x=global_iters, y=dec_l_ae_np)
            neptune.send_metric('l_reg_zr', x=global_iters, y=l_reg_zr_np)
            neptune.send_metric('l_reg_zpp', x=global_iters, y=l_reg_zpp_np)
            neptune.send_metric('lr', x=global_iters, y=lr_np)
            neptune.send_metric('margin', x=global_iters, y=margin_np)

            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch+1, args.nb_epoch, iteration+1, iterations))
            print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}, lr={}'.format(enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, lr_np))
            print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}, lr={}'.format(generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np))
            print(' Disc_loss: {}, l_reg_zd: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}'.format(disc_loss_np, l_reg_zd_np, l_reg_zr_ng_np, l_reg_zpp_ng_np))

        if ((global_iters % iterations_per_epoch == 0) and args.save_latent):
            _ = session.run([test_iterator_init_op_a, test_iterator_init_op_b])
            _ = utils.save_output(session, '_'.join([args.prefix, args.dataset]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: fixed_next}), OrderedDict({"train_mean": z_mean, "train_log_var": z_log_var}), args.latent_cloud_size)
            a_result_dict = utils.save_output(session, '_'.join([args.prefix, args.test_dataset_a]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_a}), OrderedDict({"test_mean": z_mean, "test_log_var": z_log_var, "test_reconstloss": reconst_loss}), args.test_size)
            b_result_dict = utils.save_output(session, '_'.join([args.prefix, args.test_dataset_b]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_b}), OrderedDict({"test_mean": z_mean, "test_log_var": z_log_var, "test_reconstloss": reconst_loss}), args.test_size)

        if (global_iters % iterations_per_epoch == 0) and args.save_fixed_gen and ((epoch + 1)  % 10 == 0):
            z_fixed_gen = tf.random_normal([args.batch_size, args.latent_dim])
            _ = utils.save_output(session, '_'.join([args.prefix, args.dataset]), epoch, global_iters, args.batch_size, OrderedDict({generator_input: z_fixed_gen}), OrderedDict({"fixed_gen": generator_output}), args.fixed_gen_num)

        if ((global_iters % iterations_per_epoch == 0) and args.save_latent and (epoch + 1) % 10 == 0):

            _ = session.run([test_iterator_init_op_a, test_iterator_init_op_b])
            xt_a, xt_b = session.run([test_next_a, test_next_b])
            xt_a_r, = session.run([xr], feed_dict={encoder_input: xt_a})
            xt_b_r, = session.run([xr], feed_dict={encoder_input: xt_b})

            n_x = 5
            n_y = min(args.batch_size // n_x, 50)
            print('Save original images.')
            orig_img = utils.plot_images(np.transpose(x, (0, 2, 3, 1)), n_x, n_y, "{}_original_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save generated images.')
            gen_img = utils.plot_images(np.transpose(x_p, (0, 2, 3, 1)), n_x, n_y, "{}_sampled_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save reconstructed images.')
            rec_img = utils.plot_images(np.transpose(x_r, (0, 2, 3, 1)), n_x, n_y, "{}_reconstructed_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save A test images.')
            test_a_img = utils.plot_images(np.transpose(xt_a_r, (0, 2, 3, 1)), n_x, n_y, "{}_test_a_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save B test images.')
            test_b_img = utils.plot_images(np.transpose(xt_b_r, (0, 2, 3, 1)), n_x, n_y, "{}_test_b_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)

            neptune.send_image('original', orig_img)
            neptune.send_image('generated', gen_img)
            neptune.send_image('reconstruction', rec_img)
            neptune.send_image('test_a', test_a_img)
            neptune.send_image('test_b', test_b_img)


        if False and ((global_iters % iterations_per_epoch == 0) and ((epoch + 1) % 10 == 0)):
            if args.model_path is not None:
                saved = saver.save(session, args.model_path + "/model", global_step=global_iters)
                print('Saved model to ' + saved)

        if ((global_iters % iterations_per_epoch == 0) and args.oneclass_eval):
            utils.oneclass_eval(args.normal_class, "{}_{}_epoch{}_iter{}.npy".format(args.prefix, 'kldiv', epoch, global_iters), margin_np)
            #kl_a = utils.save_kldiv(session, '_'.join([args.prefix, args.test_dataset_a]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_a}), OrderedDict({"mean": z_mean, "log_var": z_log_var}), args.test_size)
            #kl_b = utils.save_kldiv(session, '_'.join([args.prefix, args.test_dataset_b]), epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next_b}), OrderedDict({"mean": z_mean, "log_var": z_log_var}), args.test_size)

            def kldiv(mean, log_var):
                return 0.5 * np.sum(- 1 - log_var + np.square(mean) + np.exp(log_var), axis=-1)

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
            likelihood_a = kl_a + rec_a
            likelihood_b = kl_b + rec_b

            neptune.send_metric('test_mean_a', np.mean(mean_a))
            neptune.send_metric('test_mean_b', np.mean(mean_b))
            neptune.send_metric('test_var_a', np.mean(np.exp(a_result_dict['test_log_var']), axis=(0,1)))
            neptune.send_metric('test_var_b', np.mean(np.exp(b_result_dict['test_log_var']), axis=(0,1)))
            neptune.send_metric('test_rec_a', np.mean(rec_a))
            neptune.send_metric('test_rec_b', np.mean(rec_b))

            auc_kl = roc_auc_score(np.concatenate([np.zeros_like(kl_a), np.ones_like(kl_b)]), np.concatenate([kl_a, kl_b]))
            auc_mean = roc_auc_score(np.concatenate([np.zeros_like(mean_a), np.ones_like(mean_b)]), np.concatenate([mean_a, mean_b]))
            auc_rec = roc_auc_score(np.concatenate([np.zeros_like(rec_a), np.ones_like(rec_b)]), np.concatenate([rec_a, rec_b]))
            auc_l2_mean = roc_auc_score(np.concatenate([np.zeros_like(l2_mean_a), np.ones_like(l2_mean_b)]), np.concatenate([l2_mean_a, l2_mean_b]))
            auc_l2_var = roc_auc_score(np.concatenate([np.zeros_like(l2_var_a), np.ones_like(l2_var_b)]), np.concatenate([l2_var_a, l2_var_b]))
            auc_likelihood = roc_auc_score(np.concatenate([np.zeros_like(likelihood_a), np.ones_like(likelihood_b)]), np.concatenate([likelihood_a, likelihood_b]))

            neptune.send_metric('auc_kl_{}_vs_{}'.format(args.test_dataset_a, args.test_dataset_b), x=global_iters, y=auc_kl)
            neptune.send_metric('auc_mean_{}_vs_{}'.format(args.test_dataset_a, args.test_dataset_b), x=global_iters, y=auc_mean)
            neptune.send_metric('auc_rec_{}_vs_{}'.format(args.test_dataset_a, args.test_dataset_b), x=global_iters, y=auc_rec)
            neptune.send_metric('auc_l2_mean_{}_vs_{}'.format(args.test_dataset_a, args.test_dataset_b), x=global_iters, y=auc_l2_mean)
            neptune.send_metric('auc_l2_var_{}_vs_{}'.format(args.test_dataset_a, args.test_dataset_b), x=global_iters, y=auc_l2_var)
            neptune.send_metric('auc_likelihood_{}_vs_{}'.format(args.test_dataset_a, args.test_dataset_b), x=global_iters, y=auc_likelihood)

            neptune.send_metric('auc', x=global_iters, y=auc_likelihood)

neptune.stop()
