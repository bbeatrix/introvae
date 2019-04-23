import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-deep')


def plothist(dir_path, test_a, test_b, prefix, plot_type='kldiv', epoch_range=range(0,21,1), limits=[0, 100]):
    path = dir_path + prefix + '/'
    save_dir = './pictures/hist_test_a_vs_test_b/'
    files = os.listdir(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in epoch_range:
        x_mean_file = [item for item in files if prefix + '_{}_{}_epoch{}_'.format(test_a, 'test_mean', epoch) in item]
        y_mean_file = [item for item in files if prefix + '_{}_{}_epoch{}_'.format(test_b, 'test_mean', epoch) in item]

        x_log_var_file = [item for item in files if prefix + '_{}_{}_epoch{}_'.format(test_a, 'test_log_var', epoch) in item]
        y_log_var_file = [item for item in files if prefix + '_{}_{}_epoch{}_'.format(test_b, 'test_log_var', epoch) in item]

        x_rec_file = [item for item in files if prefix + '_{}_{}_epoch{}_'.format(test_a, 'test_reconstloss', epoch) in item]
        y_rec_file = [item for item in files if prefix + '_{}_{}_epoch{}_'.format(test_b, 'test_reconstloss', epoch) in item]

        def kldiv(mean, log_var):
            return 0.5 * np.sum(- 1 - log_var + np.square(mean) + np.exp(log_var), axis=-1)

        if x_mean_file and y_mean_file and x_log_var_file and y_log_var_file and x_rec_file and y_rec_file:
            x_mean = np.load(path + x_mean_file[0])
            y_mean = np.load(path + y_mean_file[0])
            x_logvar = np.load(path + x_log_var_file[0])
            y_logvar = np.load(path + y_log_var_file[0])
            x_rec = np.load(path + x_rec_file[0])
            y_rec = np.load(path + y_rec_file[0])

            x_kldiv = kldiv(x_mean, x_logvar)
            y_kldiv = kldiv(y_mean, y_logvar)

            if x_mean.ndim > 1:
                x_mean = np.mean(x_mean, axis=1)
            if y_mean.ndim > 1:
                y_mean = np.mean(y_mean, axis=1)

            bins = np.linspace(limits[0], limits[1], 100)

            if plot_type == 'test_mean':
                _, x_bins, _ = plt.hist(x_mean, bins, alpha=0.5, label=test_a)
                plt.hist(y_mean, bins=x_bins, alpha=0.5, label=test_b)
            elif plot_type == 'test_likelihood':
                #calc likelihood
                x_likelihood = x_kldiv + x_rec
                y_likelihood = y_kldiv + y_rec

                _, x_bins, _ = plt.hist(x_likelihood, bins, alpha=0.5, label=test_a)
                plt.hist(y_likelihood, bins=x_bins, alpha=0.5, label=test_b)
            plt.legend(loc='upper right')
            plt.title('{}_hist epoch_{}'.format(plot_type, epoch), loc='center')

            plt.savefig('{}/{}_{}_hist_epoch{}.png'.format(save_dir, prefix, plot_type, epoch))
            print('Saved hist to {}/{}_{}_hist_epoch{}.png'.format(save_dir, prefix, plot_type, epoch))
            plt.clf()


if __name__ == "__main__":
    dir_path = '/home/csadrian/introvae/pictures/'
    #dir_path = './pictures/'
    test_a = 'fashion_mnist'
    test_b = 'mnist'
    m = 30
    for alpha in [0.0, 0.5]:
        prefix = "{}_vs_{}_aug=False_alpha={}_m={}_seed=0".format(test_a, test_b, alpha, m)
        plothist(dir_path, test_a, test_b, prefix, 'test_likelihood', range(199,200,1), [0, 20])
        plothist(dir_path, test_a, test_b, prefix, 'test_mean', range(199,200,1), [-1, 1])
