import statistics
import pandas as pd

import matplotlib.pyplot as plt

HALF_LOG_TWO_PI = 0.91893

from neptunelib.session import Session
session = Session()
project = session.get_projects('csadrian')['csadrian/oneclass']


tags = ['baseline_fashion_mnist_vs_mnist',\
       'adv_fashion_mnist_vs_mnist',\
       'baseline_mnist_vs_fashion_mnist',\
       'adv_mnist_vs_fashion_mnist',\
       'neg_letters_fashion_mnist_vs_mnist',\
       'neg_letters_half_fashion_mnist_vs_mnist',\
       'neg_kmnist_fashion_mnist_vs_mnist',\
       'neg_fashmnist_fashion_mnist_vs_mnist',\
       'adv_genonly_fashion_mnist_vs_mnist',\
       ]

fig, ax = plt.subplots()

#plt.plot(baseline['x'], ['EpRewMean'], color='tab:cyan', label='With curriculum learning')

for tag in tags:
    exps = project.get_experiments(tag=[tag])

    auc_bpd_lasts = []
    test_rec_a_lasts = []
    test_rec_b_lasts = []

    results = []
    for exp in exps:
        res = exp.get_numeric_channels_values('auc_bpd', 'test_rec_a', 'test_rec_b')
        
        if res["x"].iloc[-1] < 100000:
            continue

        if res["x"].iloc[-1] == 100000 and tag == '':
            continue

        results.append(res)

        auc_bpd_lasts.append(res["auc_bpd"].iloc[-1])
        test_rec_a_lasts.append(res["test_rec_a"].iloc[-1] - 28*28*HALF_LOG_TWO_PI)
        test_rec_b_lasts.append(res["test_rec_b"].iloc[-1] - 28*28*HALF_LOG_TWO_PI)
    if len(results) <= 1:
        continue
    auc_bpd_std = statistics.stdev(auc_bpd_lasts)
    auc_bpd_mean = statistics.mean(auc_bpd_lasts)
    test_rec_a_std = statistics.stdev(test_rec_a_lasts)
    test_rec_a_mean = statistics.mean(test_rec_a_lasts)
    test_rec_b_std = statistics.stdev(test_rec_b_lasts)
    test_rec_b_mean = statistics.mean(test_rec_b_lasts)

    print(tag, 'auc_bpd_mean:', auc_bpd_mean, 'auc_bpd_std:', auc_bpd_std, 'test_rec_a_mean:', test_rec_a_mean, 'test_rec_a_std:', test_rec_a_std, "cnt: ", len(auc_bpd_lasts))
    
    results_df = pd.concat(results)
    results_by_row = results_df.groupby(results_df.x)
    means = results_by_row.mean()
    #print(means)
    if tag == 'baseline_fashion_mnist_vs_mnist':
        plt.plot(means.index.values, means['auc_bpd'], color='tab:brown', label='Baseline VAE')#, linestyle='--')
    elif tag == 'adv_fashion_mnist_vs_mnist':
        plt.plot(means.index.values, means['auc_bpd'], color='tab:cyan', label='With adversarial negative sampling')#, linestyle='--')
    elif tag == 'neg_letters_fashion_mnist_vs_mnist':
        plt.plot(means.index.values, means['auc_bpd'], color='tab:olive', label='EMNIST/Letters as negative samples', linestyle='--')

plt.xlabel('Steps')
plt.ylabel('Mean AUC (bits per dimension)')
plt.legend()
plt.savefig('auc_bpd.pdf')

asd

plt.plot(wo_curr_ep_rew_mean['x'], wo_curr_ep_rew_mean['EpRewMean'], color='tab:brown', label='Without curriculum learning', linestyle='--')

plt.ylim(0.0, 1.1)
plt.axhline(y=1.0, color='black', linestyle=':')
ax.annotate('Max reward (1.0 when proof found)', xy=(0, 1.02))
plt.title('Comparing episode rewards\n Curriculum learning vs pure exploration')
plt.xlabel('Steps')
plt.ylabel('Mean reward per episode')
plt.legend()
plt.savefig('curr_vs_nocurr_episode_reward_mean.pdf')

   


