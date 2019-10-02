import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-deep')

path = '/home/bbea/introvae/pictures/2019-10-02-16:06:57_FINAL_neg_letters_bernoulli_1_mean_coeff_8_0.0_fashion_mnist_vs_mnist_baseline_mnist_alpha_gen=0.0_alpha_neg=1.0_neg_dataset=emnist-letters_nc=-1_seed=10'

files = os.listdir(path)

save_dir = '/home/bbea/introvae/latent_plots/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

a_means_files = [item for item in files if 'test_a_mean' in item]

print(len(a_means_files))
