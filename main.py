# import comet_ml in the top of your file
from comet_ml import Experiment, Optimizer


import numpy as np
import tensorflow as tf
import keras, keras.backend as K

from keras.layers import Input
from keras.models import Model

import os, sys, time
from collections import OrderedDict

import model, params, losses, utils, data

#
# Config
#

args = params.getArgs()
print(args)

# set random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

print('Keras version: ', keras.__version__)
print('Tensorflow version: ', tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
sess = tf.Session(config=config)
set_session(sess)


#
# Datasets
#

K.set_image_data_format('channels_first')

data_path = os.path.join(args.datasets_dir, args.dataset)

iterations = args.nb_epoch * args.train_size // args.batch_size
iterations_per_epoch = args.train_size // args.batch_size

if args.dataset == 'cifar10':
    ds = data.create_cifar10_unsup_dataset(args.batch_size, args.train_size, args.test_size, args.latent_cloud_size, args.normal_class, args.gcnorm)
    train_data, train_placeholder, train_dataset, train_iterator, train_iterator_init_op, train_next = ds[0]
    test_data, test_placeholder, test_dataset, test_iterator, test_iterator_init_op, test_next = ds[1]
    fixed_data, fixed_placeholder, fixed_dataset, fixed_iterator, fixed_iterator_init_op, fixed_next = ds[2]
else:
    train_dataset, train_iterator, train_iterator_init_op, train_next \
         = data.create_dataset(os.path.join(data_path, "train/*.npy"), args.batch_size, args.train_size)
    test_dataset, test_iterator, test_iterator_init_op, test_next \
         = data.create_dataset(os.path.join(data_path, "test/*.npy"), args.batch_size, args.test_size)
    fixed_dataset, fixed_iterator, fixed_iterator_init_op, fixed_next \
         = data.create_dataset(os.path.join(data_path, "train/*.npy"), args.batch_size, args.latent_cloud_size)

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

xr = generator(z)
reconst_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='reconst_latent_input')
zr_mean, zr_log_var = encoder(generator(reconst_latent_input))
zr_sg = tf.stop_gradient(generator(reconst_latent_input))
zr_mean_ng, zr_log_var_ng = encoder(zr_sg)
xr_latent = generator(reconst_latent_input)

sampled_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='sampled_latent_input')
zpp_mean, zpp_log_var = encoder(generator(sampled_latent_input))
zpp_sg = tf.stop_gradient(generator(sampled_latent_input))
zpp_mean_ng, zpp_log_var_ng = encoder(zpp_sg)

global_step = tf.Variable(0, trainable=False)

starter_learning_rate = args.lr
if args.lr_schedule == 'exponential':
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 8250, 0.1, staircase=True)
else:
    learning_rate = tf.constant(args.lr)

encoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)

print('Encoder')
encoder.summary()
print('Generator')
generator.summary()

#
# Define losses
#
optimizer = Optimizer("UixmeRWMTNgxw3Od6lPNjQxsI")
params = """
alpha real [0, 10] [0.75]
beta real [0, 10] [0.25]
m real [0, 100] [40.0]
"""
optimizer.set_params(params)

while True:
    suggestion = optimizer.get_suggestion()
    # Add the following code anywhere in your machine learning file
    #args.m = suggestion['m']
    #args.alpha = suggestion['alpha']
    #args.beta = suggestion['beta']
    #args.gradreg = suggestion['gradreg']
    experiment = Experiment(api_key="UixmeRWMTNgxw3Od6lPNjQxsI", project_name="introvae-oneclass-exps", workspace="csadrian", log_graph=False)
    print("Starting new experiment")
    print(args)


    l_reg_z = losses.reg_loss(z_mean, z_log_var)
    l_reg_zr_ng = losses.reg_loss(zr_mean_ng, zr_log_var_ng)
    l_reg_zpp_ng = losses.reg_loss(zpp_mean_ng, zpp_log_var_ng)

    l_ae = losses.mse_loss(encoder_input, xr, args.original_shape)
    l_ae2 = losses.mse_loss(encoder_input, xr_latent, args.original_shape)


    z_mean_gradients = tf.gradients(z_mean * tf.random_normal((args.latent_dim,)), [encoder_input])[0]
    z_log_var_gradients = tf.gradients(z_log_var * tf.random_normal((args.latent_dim,)), [encoder_input])[0]

    zr_mean_gradients = tf.gradients(zr_mean_ng * tf.random_normal((args.latent_dim,)), [zr_sg])[0]
    zr_log_var_gradients = tf.gradients(zr_log_var_ng * tf.random_normal((args.latent_dim,)), [zr_sg])[0]

    zpp_mean_gradients = tf.gradients(zpp_mean_ng * tf.random_normal((args.latent_dim,)), [zpp_sg])[0]
    zpp_log_var_gradients = tf.gradients(zpp_log_var_ng * tf.random_normal((args.latent_dim,)), [zpp_sg])[0]

    spectreg_loss_z = tf.reduce_mean((1.0-tf.reduce_sum(tf.square(z_mean_gradients), axis=1))**2)
    #spectreg_loss += tf.reduce_mean(1.0-tf.reduce_sum(tf.square(z_log_var_gradients), axis=1))
    #spectreg_loss_r = tf.reduce_mean((1.0-tf.reduce_sum(tf.square(zr_mean_gradients), axis=1))**2)
    #spectreg_loss += tf.reduce_mean(1.0-tf.reduce_sum(tf.square(zr_log_var_gradients), axis=1))
    spectreg_loss_zpp = tf.reduce_mean((1.0-tf.reduce_sum(tf.square(zpp_mean_gradients), axis=1))**2)
    #spectreg_loss += tf.reduce_mean(1.0-tf.reduce_sum(tf.square(zpp_log_var_gradients), axis=1))

    #spectreg_loss = tf.reduce_mean(spectreg_loss, axis=-1)
    spectreg_loss = spectreg_loss_z# + spectreg_loss_zpp# + spectreg_loss_r

    encoder_l_adv = l_reg_z + args.alpha * K.maximum(0., args.m - l_reg_zr_ng) + args.alpha * K.maximum(0., args.m - l_reg_zpp_ng)
    encoder_loss = encoder_l_adv + args.beta * l_ae + args.gradreg * spectreg_loss

    l_reg_zr = losses.reg_loss(zr_mean, zr_log_var)
    l_reg_zpp = losses.reg_loss(zpp_mean, zpp_log_var)

    generator_l_adv = args.alpha * l_reg_zr + args.alpha * l_reg_zpp
    generator_loss = generator_l_adv + args.beta * l_ae2# + args.gradreg * spectreg_loss


    #
    # Define training step operations
    #

    encoder_params = encoder.trainable_weights
    generator_params = generator.trainable_weights

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

    session = sess
    if True:
        init = tf.global_variables_initializer()
        session.run([init, train_iterator_init_op, test_iterator_init_op, fixed_iterator_init_op],
                    feed_dict={train_placeholder: train_data, test_placeholder: test_data, fixed_placeholder: fixed_data})
        summary_writer = tf.summary.FileWriter(args.prefix+"/", graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=None)
        if False and args.model_path is not None and tf.train.checkpoint_exists(args.model_path):
            saver.restore(session, tf.train.latest_checkpoint(args.model_path))
            print('Model restored from ' + args.model_path)
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            global_iters = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            start_epoch = (global_iters * args.batch_size) // args.train_size
        print('Global iters: ', global_iters)

        if args.oneclass_eval:
            utils.save_kldiv(session, args.prefix, start_epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next}), OrderedDict({"mean": z_mean, "log_var": z_log_var}), args.test_size)
            utils.oneclass_eval(args.normal_class, "{}_{}_epoch{}_iter{}.npy".format(args.prefix, 'kldiv', start_epoch, global_iters), args.m)

        for iteration in range(iterations):
            epoch = global_iters * args.batch_size // args.train_size
            global_iters += 1

            x = session.run(train_next)
            z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
            z_x, x_r, x_p = session.run([z, xr, generator_output], feed_dict={encoder_input: x, generator_input: z_p})

            _ = session.run([encoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
            _ = session.run([generator_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})

            if global_iters % 10 == 0:
                summary, = session.run([summary_op], feed_dict={encoder_input: x})
                summary_writer.add_summary(summary, global_iters)

            if (global_iters % args.frequency) == 0:
                enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, global_step_np = \
                 session.run([encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, generator_loss, l_ae2, l_reg_zr, l_reg_zpp, learning_rate, global_step],
                             feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
                print('Epoch: {}/{}, iteration: {}/{}'.format(epoch+1, args.nb_epoch, iteration+1, iterations))
                print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}, lr: {}'.format(enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, lr_np))
                print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}, lr: {}, global_step: {}'.format(generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np, lr_np, global_step_np))

            if ((global_iters % iterations_per_epoch == 0) and args.save_latent):
                #utils.save_output(session, args.prefix, epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next}), OrderedDict({"test_mean": z_mean, "test_log_var": z_log_var}), args.test_size)
                #utils.save_output(session, args.prefix, epoch, global_iters, args.batch_size, OrderedDict({encoder_input: fixed_next}), OrderedDict({"train_mean": z_mean, "train_log_var": z_log_var}), args.latent_cloud_size)

                n_x = 5
                n_y = args.batch_size // n_x
                print('Save original images.')
                utils.plot_images(np.transpose(x, (0, 2, 3, 1)), n_x, n_y, "{}_original_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
                print('Save generated images.')
                utils.plot_images(np.transpose(x_p, (0, 2, 3, 1)), n_x, n_y, "{}_sampled_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
                print('Save reconstructed images.')
                utils.plot_images(np.transpose(x_r, (0, 2, 3, 1)), n_x, n_y, "{}_reconstructed_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)

            if False and ((global_iters % iterations_per_epoch == 0) and ((epoch + 1) % 10 == 0)):
                if args.model_path is not None:
                    saved = saver.save(session, args.model_path + "/model", global_step=global_iters)
                    print('Saved model to ' + saved)

            if ((global_iters % iterations_per_epoch == 0) and args.oneclass_eval):
                utils.save_kldiv(session, args.prefix, epoch, global_iters, args.batch_size, OrderedDict({encoder_input: test_next}), OrderedDict({"mean": z_mean, "log_var": z_log_var}), args.test_size)
                auc = utils.oneclass_eval(args.normal_class, "{}_{}_epoch{}_iter{}.npy".format(args.prefix, 'kldiv', epoch, global_iters), args.m)
                experiment.log_metric('auc', auc, step=global_step_np)
    quit()
    suggestion.report_score('auc', auc)