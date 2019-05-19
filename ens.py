import numpy as np
from keras.datasets import cifar10

from sklearn.metrics import roc_auc_score


def oneclass_eval(normal_class, kldiv, margin, t):
    if normal_class == -1:
        pass
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_true = [1 if item == normal_class else 0 for item in y_test]
        y_scores = kldiv
        auc = roc_auc_score(t, y_scores)
        print('AUC: ', auc)


nc = 6

dirs = []
dirs.append("gr_final_nowd_nobn_normed_cifar10_vs_cifar10_dcgan_univ_aug=False_alpha_reconstructed=1.0_alpha_generated=1.0_m=30_beta=0.1_nc=3_seed=0")
#dirs.append("final_nowd_nobn_normed_cifar10_vs_cifar10_dcgan_univ_aug=False_alpha_reconstructed=1.0_alpha_generated=1.0_m=40_beta=0.1_nc="+str(nc)+"_seed=0")
#dirs.append("final_nowd_nobn_normed_cifar10_vs_cifar10_dcgan_univ_aug=False_alpha_reconstructed=1.0_alpha_generated=1.0_m=40_beta=0.1_nc="+str(nc)+"_seed=1")
#dirs.append("final_nowd_nobn_normed_cifar10_vs_cifar10_dcgan_univ_aug=False_alpha_reconstructed=1.0_alpha_generated=1.0_m=40_beta=0.1_nc="+str(nc)+"_seed=2")

import glob

nls = []

for dir in dirs:

    files = [f for f in glob.glob("pictures/" + dir + "/*neglog_*epoch*.npz", recursive=True)]

    for f in files:
        print(f)
        a = np.load(f)
        nl = a['neglog_likelihoods']
        print(nl[:10])

        labels = a['labels']
        print(nl[-10:])
        nls.append(nl)
        oneclass_eval(nc, nl, 0, labels)

nls_np = np.array(nls)
nl = nls_np.mean(axis=0)
oneclass_eval(nc, nl, 0, labels)
