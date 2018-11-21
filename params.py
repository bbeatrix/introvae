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
parser.add_argument('--dataset', dest="dataset", default="celeba-128x128", type=str, help="Dataset (possible values: celeba-128x128)")
parser.add_argument('--datasets_dir', dest="datasets_dir", default="datasets", type=str, help="Location of datasets.")
parser.add_argument('--color', dest="color", default=True, type=str2bool, help="Color (True/False)")
parser.add_argument('--train_size', dest="train_size", type=int, default=29000, help="Train set size.")
parser.add_argument('--test_size', dest="test_size", type=int, default=1000, help="Test set size.")
parser.add_argument('--shape', dest="shape", default="128,128", help="Image shape.")

# model hyperparameters
parser.add_argument('--m', dest="m", type=int, default=120, help="Value of model hyperparameter m.")
parser.add_argument('--alpha', dest="alpha", type=float, default=0.25, help="Value of model hyperparameter alpha.")
parser.add_argument('--beta', dest="beta", type=float, default=0.05, help="Value of model hyperparameter beta.")

# training
parser.add_argument('--lr', dest="lr", default="0.001", type=float, help="Learning rate for the optimizer.")
parser.add_argument('--batch_size', dest="batch_size", default=200, type=int, help="Batch size.")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=200, help="Number of epochs.")
parser.add_argument('--verbose', dest="verbose", type=int, default=2, help="Logging verbosity: 0-silent, 1-verbose, 2-perEpoch (default).")

# architecture
parser.add_argument('--sampling', dest="sampling", type=str2bool, default=True, help="Use sampling.")
parser.add_argument('--sampling_std', dest="sampling_std", type=float, default=-1.0, help="Sampling std, if < 0, then we learn std.")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension.")
parser.add_argument('--base_filter_num', dest="base_filter_num", default=32, type=int, help="Initial number of filter in the conv model.")
parser.add_argument('--resnet_wideness', dest="resnet_wideness", default=1, help="Wideness of resnet model (1-wide first block has 16 filters).")

# encoder
parser.add_argument('--encoder_use_bn', dest="encoder_use_bn", type=str2bool, default=False, help="Use batch normalization in encoder.")
parser.add_argument('--encoder_wd', dest="encoder_wd", type=float, default=0.0, help="Weight decay param for the encoder.")

# generator
parser.add_argument('--generator_use_bn', dest="generator_use_bn", type=str2bool, default=False, help="Use batch normalization in generator.")
parser.add_argument('--generator_wd', dest="generator_wd", type=float, default=0.0, help="Weight decay param for generator.")

# locations
parser.add_argument('--prefix', dest="prefix", default="trash", help="File prefix for the output visualizations and models.")
parser.add_argument('--model_path', dest="model_path", default=None, help="Path to saved networks. If None, build networks from scratch.")

# micellaneous
parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.95, help="Fraction of memory that can be allocated to this process.")
parser.add_argument('--frequency', dest="frequency", type=int, default=20, help="Image saving frequency.")
parser.add_argument('--save_latent', dest="save_latent", type=str2bool, default=False, help="If True, then save latent pointcloud.")
parser.add_argument('--latent_cloud_size', dest="latent_cloud_size", type=int, default=10000, help="Size of latent cloud.")

args = parser.parse_args()

def getArgs():
    # put output files in a separate directory
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    prefix_parts = args.prefix.split("/")
    prefix_parts.append(prefix_parts[-1])
    args.prefix = "/".join(prefix_parts)

    args.shape = tuple(map(int, str(args.shape).split(",")))
    return args
