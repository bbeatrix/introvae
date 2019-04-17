import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plothist(plot_type='kldiv', prefix='cifar10_vs_svhn', epoch_range=range(0,21,1), limits=[0, 100]):
    path = './pictures/' + prefix + '/'
    files = os.listdir(path)
    save_dir = 'cifar10_vs_svhn/{}'.format(prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in epoch_range:
        x_file = [item for item in files if prefix + '_cifar10_{}_epoch{}_'.format(plot_type, epoch) in item]
        y_file = [item for item in files if prefix + '_svhn_{}_epoch{}_'.format(plot_type, epoch) in item]

        if x_file and y_file:
            x = np.load(path + x_file[0])
            y = np.load(path + y_file[0])

            #if plot_type == 'test_reconstloss':
                #print(x.shape)

            if x.ndim > 1:
                x = np.mean(x, axis=1)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            bins = np.linspace(limits[0], limits[1], 100)

            plt.hist(x, bins, alpha=0.5, label='cifar10', color='blue')
            plt.hist(y, bins, alpha=0.5, label='svhn', color='red')
            plt.legend(loc='upper right')
            plt.title('{} epoch_{}'.format(plot_type, epoch), loc='center')

            plt.savefig('{}/cifarvssvhn_{}_epoch{}.png'.format(save_dir, plot_type, epoch))
            print('Saved hist to cifarvssvhn_{}_epoch{}.png'.format(plot_type, epoch))
            plt.clf()


if __name__ == "__main__":
    plothist('kldiv', 'cifar10_vs_svhn_m=40_alpha=0.25', range(0,21,1), [0, 100])
    plothist('kldiv', 'cifar10_vs_svhn_m=40_alpha=0.25', range(9,199,10), [0, 100])
    plothist('test_mean', 'cifar10_vs_svhn_m=40_alpha=0.25', range(0,21,1), [-1, 1])
    plothist('test_mean', 'cifar10_vs_svhn_m=40_alpha=0.25', range(9,199,10), [-1, 1])
    plothist('test_reconstloss', 'cifar10_vs_svhn_m=40_alpha=0.25', range(0,21,1), [0, 200])
    plothist('test_reconstloss', 'cifar10_vs_svhn_m=40_alpha=0.25', range(9,199,10), [0, 200])