import matplotlib.pyplot as plt
from collections import defaultdict
from neptune import Session
import numpy as np

session = Session()
project = session.get_projects('csadrian')['csadrian/oneclass']

def check_crit(params, crit):
  for k, v in crit.items():
    if params[k] != v:
      return False
  return True

crits = {}
#crits['szepkep_baseline'] ={'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['szepkep_letters'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'emnist-letters', 'neg_prior_mean_coeff': 8 }
#crits['szepkep_kmnist'] =  {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'kmnist', 'neg_prior_mean_coeff': 8 }
#crits['szepkep_noise'] =   {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'uniform-noise', 'neg_prior_mean_coeff': 8 }
#crits['szepkep_adv'] =     {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'neg_prior_mean_coeff': 8, 'alpha_generated': 1.0 }
crits['szepkep_isonoise'] =   {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'fashion_mnist', 'add_iso_noise_to_neg': 'True', 'neg_prior_mean_coeff': 8 }

#crits['fashion_mnist_vs_mnist_baseline_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['fashion_mnist_vs_mnist_baseline_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0 }

#crits['mnist_vs_fashion_mnist_baseline_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['mnist_vs_fashion_mnist_baseline_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0 }

#crits['cifar10_vs_svhn_baseline_bernoulli'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['cifar10_vs_svhn_baseline_gaussian'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'model_architecture': 'dcgan_univ', 'trained_gamma': 'False', 'reg_lambda': 1.0, 'optimizer': 'rmsprop', 'encoder_use_bn': 'False'}

#crits['svhn_vs_cifar10_baseline_bernoulli'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['svhn_vs_cifar10_baseline_gaussian'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0 }


channels = ['auc_bpd', 'auc_kl', 'auc_mean', 'test_bpd_a', 'test_bpd_b']
exps = project.get_experiments()
for exp in exps:
  exp._my_params = exp.get_parameters()

for key, crit in crits.items():
  res = defaultdict(list)
  for exp in exps:
    params = exp._my_params
    if not check_crit(params, crit):
      continue
    else:
      print(exp.id)
    vals = exp.get_logs()
    for channel in channels:
      if channel in vals:
        res[channel].append(float(vals[channel]['y']))
    postfix = params['test_dataset_a'] + '_vs_' + params['test_dataset_b']
    res['auc_kl'].append(float(vals['auc_kl_'+postfix]['y']))
    res['auc_mean'].append(float(vals['auc_mean_'+postfix]['y']))

  for channel in channels:
    v = np.array(res[channel])
    if v.shape[0] > 5:
      print("{}: Warning: more than 5 exeriments: {} using only 5 (no order assumed)".format(key, v.shape[0]))
    v = v[:5]
    mean = np.mean(v)
    std = np.std(v)
    cnt = v.shape[0]
    print("{: <25} {} mean: {:.2f}, std: {:.2f}, cnt: {}".format(key, channel, mean, std, cnt))
