from sklearn import decomposition
from sklearn import datasets
from sklearn.manifold import TSNE

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
#plt.style.use('seaborn-deep')

sns.set()


def main():
    np.random.seed(0)

    #exp = 'neg_letters'
    #exp = 'baseline'
    exp = 'neg_adv'

    epochs = [i for i in range(1, 101, 10)]
    #epochs = [1, 5, 10, 50, 100]
    
    limit = 300

    coords = [2, 3]

    path = '/home/bbea/projects/2ndprior/latents/{}/'.format(exp)


    save_dir = '/home/bbea/projects/2ndprior/latents/latent_plots/'

    
    files = os.listdir(path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_types = ['test_a_mean', 'test_b_mean']
    if exp == 'neg_letters':
        file_types.append('test_neg_mean')

    files_dict = {}

    for file_type in file_types:
        files_dict[file_type] = [item for item in files if file_type in item]

    for epoch in epochs:
        #print('epoch: ', epoch)

        data = {}
        for key, value in files_dict.items():
            file =  [item for item in files_dict[key] if '_epoch{}_'.format(epoch) in item][0]
            data[key] = np.load(path + file)[:limit,:]

        if exp == 'neg_letters':
            plot_latent_means(exp, save_dir, epoch, data['test_a_mean'], data['test_b_mean'], coords, neg_means=data['test_neg_mean']) 
        #elif exp == 'neg_adv':
        else:
            plot_latent_means(exp, save_dir, epoch, data['test_a_mean'], data['test_b_mean'], coords)


def plot_latent_means(exp, save_dir, epoch, a_means, b_means, coords, **kwargs):
    coord1, coord2 = coords[0], coords[1]

    plt.figure(figsize=(10,10))
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)

    plt.hlines(0, -5, 15, colors='k', linestyles='solid', alpha=0.25) 
    plt.vlines(0, -5, 15, colors='k', linestyles='solid', alpha=0.25) 

    plt.scatter(a_means[:,coord1], a_means[:,coord2], c="b", alpha=1.0, marker='o', label="Fashion-MNIST (inliers) test")
    
    if 'neg_means' in kwargs.keys():
        neg_means = kwargs['neg_means']
        plt.scatter(neg_means[:,coord1], neg_means[:,coord2], c="k", alpha=1.0, marker='v', label="EMNIST-Letters (negatives) test")
    
    plt.scatter(b_means[:,coord1], b_means[:,coord2], c="r", alpha=1.0, marker='x', label="MNIST (OOD) test")

    plt.xlabel("$z_{}$".format(coord1), fontsize=18)
    plt.ylabel("$z_{}$".format(coord2), fontsize=18)
    plt.legend(loc='upper left', prop={'size': 18})
    #plt.title('Epoch: {}'.format(epoch), loc='center')
    plt.tight_layout()
    #plt.savefig('{}/{}_latent_epoch{}.pdf'.format(save_dir, exp, epoch))
    #print('Saved latent plot as {}_latent_epoch{}.pdf'.format(exp, epoch))
    plt.savefig('{}/{}_latent_coords{}-{}_epoch{}.png'.format(save_dir, exp, coord1, coord2, epoch))
    print('Saved latent plot as {}_latent_coords{}-{}_epoch{}.png'.format(exp, coord1, coord2, epoch))
    plt.close()
    #plt.show()


def visualize_with_ellipse(plot_name, z_mean, z_logvar, labels=['test_a', 'test_b', 'test_neg']):
    n = z_mean.shape[0]
    limit = 300

    ells = [Ellipse(xy = z_mean[i], width = 2 * np.exp(z_logvar[i][0]), height = 2 * np.exp(z_logvar[i][1])) for i in range(inum)]

    fig, ax = plt.subplots(subplot_kw = {'aspect' : 'equal'})

    b, r, k = ([], ) * 3
    color_dict = {'test_a': 'blue', 'test_b': 'red', 'test_neg': 'black'}

    for i in range(n):
        ax.add_artist(ells[i])
        ells[i].set_clip_box(ax.bbox)
        ells[i].set_alpha(0.5)

        ells[i].set_facecolor(color_dict[label[i]])
        ax.legend((b[0], r[0], k[0]), (0, 1, 2), loc="best")

    plt.scatter(z_mean[:limit, 0], z_mean[:limit, 1], s = 1, c="black")

    plt.xlim(-20,20)
    plt.ylim(-20,20)

    plt.savefig(plot_name + '.png')


if __name__ == "__main__":
    main()
