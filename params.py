import os
import argparse

def str2bool(v):
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('--name', dest="name", default="", type=str, help="Name of the experiment.")
parser.add_argument('--dataset', dest="dataset", default="celeba-128x128", type=str, help="Dataset (possible values: celeba-128x128, cifar10)")
parser.add_argument('--datasets_dir', dest="datasets_dir", default="datasets", type=str, help="Location of datasets.")
parser.add_argument('--color', dest="color", default=True, type=str2bool, help="Color (True/False)")
parser.add_argument('--train_size', dest="train_size", type=int, default=29000, help="Train set size.")
parser.add_argument('--test_size', dest="test_size", type=int, default=1000, help="Test set size.")
parser.add_argument('--shape', dest="shape", default="128,128", help="Image shape.")
parser.add_argument('--normal_class', dest="normal_class", type=int, default="-1", help="Normal class for oneclass classification (0-9, -1 means all classes)")
parser.add_argument('--augment', dest='augment', default=False, type=str2bool, help="Use dataset augmentation (True/False)")
parser.add_argument('--test_dataset_a', dest='test_dataset_a', default='cifar10', type=str, help="Test dataset a - for comparison on different test datasets")
parser.add_argument('--test_dataset_b', dest='test_dataset_b', default='svhn_cropped', type=str, help="Test dataset b - for comparison on different test datasets")
parser.add_argument('--num_classes', dest="num_classes", default=10, type=int, help="Number of classes in dataset.")
parser.add_argument('--neg_dataset', dest='neg_dataset', default=None, type=str, help="Negative samples dataset")
parser.add_argument('--obs_noise_model', dest='obs_noise_model', default='gaussian', type=str, help="Noise model in the observable space. Can be: bernoulli, gaussian")
parser.add_argument('--add_obs_noise', dest='add_obs_noise', default=False, type=str2bool, help="Adds additive [0,1) uniform noise to input.")
parser.add_argument('--add_iso_noise_to_neg', dest='add_iso_noise_to_neg', default=False, type=str2bool, help="Adds isotropic gaussian noise to the negative dataset.")
parser.add_argument('--neg_train_size', dest="neg_train_size", type=int, default=50000, help="Train set size for negative set.")
parser.add_argument('--neg_test_size', dest="neg_test_size", type=int, default=10000, help="Test set size for negative set.")


# model hyperparameters
parser.add_argument('--mml', dest="mml", type=str2bool, default=False, help="If True, the likelihood of negatives is minimized, likelihood of normals is maximized")
parser.add_argument('--margin_inf', dest="margin_inf", type=str2bool, default=False, help="If True, the margin is increasing intead of being fixed value")
parser.add_argument('--m', dest="m", type=int, default=120, help="Value of model hyperparameter m.")
parser.add_argument('--alpha', dest="alpha", type=float, default=None, help="Value of model hyperparameter alpha.")
parser.add_argument('--alpha_reconstructed', dest="alpha_reconstructed", type=float, default=0.25, help="Value of model hyperparameter alpha searately for loss terms from reconstructed images.")
parser.add_argument('--alpha_generated', dest="alpha_generated", type=float, default=0.25, help="Value of model hyperparameter alpha separately for loss terms from generated images.")
parser.add_argument('--alpha_neg', dest="alpha_neg", type=float, default=0.0, help="Alpha value for negative samples.")
parser.add_argument('--alpha_fixed_gen', dest="alpha_fixed_gen", type=float, default=1.0, help="Weight of loss from fixed generated images as negatives")
parser.add_argument('--beta', dest="beta", type=float, default=0.05, help="Value of model hyperparameter beta.")
parser.add_argument('--beta_neg', dest="beta_neg", type=float, default=0.0, help="Weight of reconstruction loss of negative samples.")
parser.add_argument('--reg_lambda', dest="reg_lambda", type=float, default=1.0, help="Weight of vae loss.")
parser.add_argument('--use_augmented_variance_loss', dest="use_augmented_variance_loss", type=str2bool, default=False, help="If true, augmented variance loss is used instead of standard vae loss.")
parser.add_argument('--random_images_as_negative', dest="random_images_as_negative", default=False, type=str2bool, help="Use random images as negative samples.")
parser.add_argument('--generator_adversarial_loss', dest="generator_adversarial_loss", type=str2bool, default=True, help="If False, the generator loss is just the ae loss, otherwise the adversarial loss is added to it.")
parser.add_argument('--fixed_gen_as_negative', dest="fixed_gen_as_negative", type=str2bool, default=False, help="Use generated images as negative samples.")
parser.add_argument('--fixed_gen_max_epoch', dest="fixed_gen_max_epoch", type=int, default=10, help="Number of epochs after which generated negative samples are not updated anymore.")
parser.add_argument('--fixed_gen_num', dest="fixed_gen_num", type=int, default=10000, help="Number of generated images used as negative samples.")
parser.add_argument('--fixed_negatives_npy', dest="fixed_negatives_npy", type=str, default=None, help="Use images from npy file as fixed negative samples.")
parser.add_argument('--augment_avg_at_test', dest="augment_avg_at_test", type=str2bool, default=False, help="If True, likelihood is averaged over batch size number of images at test time.")
parser.add_argument('--neg_prior', dest="neg_prior", type=str2bool, default=False, help="Use different prior for negative samples if True.")
parser.add_argument('--neg_prior_mean_coeff', dest="neg_prior_mean_coeff", type=float, default=10, help="Coeff of mean if the negative prior is shifted Gaussian.")
parser.add_argument('--mmd_lambda', dest="mmd_lambda", type=float, default=0.0, help="Weight of MMD loss term in encoder loss.")
parser.add_argument('--priors_means_same_coords', dest="priors_means_same_coords", type=int, default=0, help="Number of same coordinates of means of the two priors.")
parser.add_argument('--eubo_lambda', dest="eubo_lambda", type=float, default=0.0, help="EUBO weight.")
parser.add_argument('--eubo_neg_lambda', dest="eubo_neg_lambda", type=float, default=0.0, help="EUBO weight on negatives.")
parser.add_argument('--eubo_gen_lambda', dest="eubo_gen_lambda", type=float, default=0.0, help="EUBO weight on generated samples as negatives.")
parser.add_argument('--z_num_samples', dest="z_num_samples", type=int, default=1, help="Number of samples from the posterior.")
parser.add_argument('--phi', dest="phi", type=float, default=1.0, help="Weight of reconst_loss term of weight w in EUBO.")
parser.add_argument('--chi', dest="chi", type=float, default=1.0, help="Weight of log_p_z term of weight w in EUBO.")
parser.add_argument('--psi', dest="psi", type=float, default=1.0, help="Weight of log_q_z term of weight w in EUBO.")
parser.add_argument('--cubo', dest="cubo", type=str2bool, default=False, help="Use CUBO instead of EUBO if True.")


# training
parser.add_argument('--train', dest="train", default=True, type=str2bool, help="Skip train loop if set to False.")
parser.add_argument('--lr', dest="lr", default="0.001", type=float, help="Learning rate for the optimizer.")
parser.add_argument('--batch_size', dest="batch_size", default=200, type=int, help="Batch size.")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=200, help="Number of epochs.")
parser.add_argument('--verbose', dest="verbose", type=int, default=2, help="Logging verbosity: 0-silent, 1-verbose, 2-perEpoch (default).")
parser.add_argument('--aux', dest="aux", type=str2bool, default=False, help="Use auxiliary training objective of predicting geometric transformations.")
parser.add_argument('--optimizer', dest="optimizer", type=str, default='rmsprop', help="Optimizer.")
parser.add_argument('--joint_training', dest="joint_training", type=str2bool, default=False, help="To train the encoder and decoder jointly or separately.")
parser.add_argument('--gradient_clipping', dest="gradient_clipping", type=str2bool, default=False, help="If True, use gradient clipping.")

# architecture
parser.add_argument('--sampling', dest="sampling", type=str2bool, default=True, help="Use sampling.")
parser.add_argument('--sampling_std', dest="sampling_std", type=float, default=-1.0, help="Sampling std, if < 0, then we learn std.")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension.")
parser.add_argument('--base_filter_num', dest="base_filter_num", default=32, type=int, help="Initial number of filter in the conv model.")
parser.add_argument('--resnet_wideness', dest="resnet_wideness", default=1, help="Wideness of resnet model (1-wide first block has 16 filters).")
parser.add_argument('--model_architecture', dest="model_architecture", type=str, default="introvae", help="Model architecture (introvae/deepsvdd/dcgan)")
parser.add_argument('--separate_discriminator', dest="separate_discriminator", type=str2bool, default=False, help='If True, separate discriminator is used instead of the encoder.')
parser.add_argument('--trained_gamma', dest="trained_gamma", type=str2bool, default=False, help="If True, variance on generator is trained.")
parser.add_argument('--initial_log_gamma', dest="initial_log_gamma", type=float, default=0.0, help="Initial values of log gamma.")

# encoder
parser.add_argument('--encoder_use_sn', dest="encoder_use_sn", type=str2bool, default=False, help="Use spectral normalization in encoder.")
parser.add_argument('--encoder_use_bn', dest="encoder_use_bn", type=str2bool, default=False, help="Use batch normalization in encoder.")
parser.add_argument('--encoder_wd', dest="encoder_wd", type=float, default=0.0, help="Weight decay param for the encoder.")

# generator
parser.add_argument('--generator_use_bn', dest="generator_use_bn", type=str2bool, default=False, help="Use batch normalization in generator.")
parser.add_argument('--generator_wd', dest="generator_wd", type=float, default=0.0, help="Weight decay param for generator.")

# locations
parser.add_argument('--prefix', dest="prefix", default="trash", help="File prefix for the output visualizations and models.")
parser.add_argument('--model_path', dest="model_path", default=None, help="Path to saved networks. If None, build networks from scratch.")

# micellaneous
parser.add_argument('--tags', dest="tags", type=str, default="junk", help="Tags for the experiments, comma separated.")
parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.95, help="Fraction of memory that can be allocated to this process.")
parser.add_argument('--frequency', dest="frequency", type=int, default=20, help="Image saving frequency.")
parser.add_argument('--save_latent', dest="save_latent", type=str2bool, default=False, help="If True, then save latent pointcloud.")
parser.add_argument('--latent_cloud_size', dest="latent_cloud_size", type=int, default=10000, help="Size of latent cloud.")
parser.add_argument('--save_fixed_gen', dest="save_fixed_gen", type=str2bool, default=False, help='If True, then save fixed_gen_num of generated images.')
parser.add_argument('--oneclass_eval', dest="oneclass_eval", type=str2bool, default=False, help="If True, then eval oneclass classification with AUC")
parser.add_argument('--seed', dest="seed", type=int, default=10, help="Random seed" )

parser.add_argument('--gradreg', dest="gradreg", type=float, default=0.0, help="Gradreg (spectreg).")
parser.add_argument('--lr_schedule', dest="lr_schedule", type=str, default="constant", help="Learning rate scheduling mode, can be: constant, exponential.")

parser.add_argument('--gcnorm', dest='gcnorm', type=str, default="l1", help="Global contrast normalization type (can be: l1/std/None)" )

args = parser.parse_args()

if args.neg_dataset == "None":
    args.neg_dataset = None

def getArgs():
    # put output files in a separate directory
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    prefix_parts = args.prefix.split("/")
    prefix_parts.append(prefix_parts[-1])
    args.prefix = "/".join(prefix_parts)

    args.shape = tuple(map(int, str(args.shape).split(",")))
    if args.alpha is not None:
        args.alpha_reconstructed, args.alpha_generated = args.alpha, args.alpha
    return args
