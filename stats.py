import matplotlib.pyplot as plt
from collections import defaultdict
from neptune import Session
import numpy as np
import sys
import pandas as pd

session = Session()
project = session.get_projects('csadrian')['csadrian/oneclass']

def check_crit(params, crit):
  for k, v in crit.items():
    if k not in params.keys():
      return False
    if isinstance(v, list):
      if params[k] not in v:
        return False
    if isinstance(params[k], list):
      if v not in params[k]:
        return False
    elif params[k] != v:
        return False
  return True

crits = {}
#crits['szepkep_baseline'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'base_filter_num': 32, 'beta': 1.0, 'latent_dim': 10, 'fixed_gen_as_negative': 'False', 'seed': [1, 2, 3, 4, 5]}
#crits['szepkep_letters'] = {'exp_id': ['ON-1619', 'ON-1628', 'ON-1637', 'ON-1643', 'ON-1647']} # ], 'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'emnist-letters', 'neg_prior_mean_coeff': 8, 'base_filter_num': 32, 'beta': 1.0, 'beta_neg': 0.0, 'latent_dim': 10, 'generator_adversarial_loss': 'True', 'fixed_gen_as_negative': 'False', 'seed': [1, 2, 3, 4, 5] }
#crits['szepkep_kmnist'] =  {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'kmnist', 'neg_prior_mean_coeff': 8, 'seed': [1, 2, 3, 4, 5] }
#crits['szepkep_noise'] =   {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'uniform-noise', 'neg_prior_mean_coeff': 8 }
#crits['szepkep_adv'] =   {'exp_id': ['ON-1617', 'ON-1626', 'ON-1633', 'ON-1639', 'ON-1650']} #{'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'neg_prior_mean_coeff': 8, 'alpha_generated': 1.0, 'seed': [1, 2, 3, 4, 5], 'base_filter_num': 32, 'latent_dim': 10, 'generator_adversarial_loss': 'True'}
#crits['szepkep_isonoise'] =   {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'fashion_mnist', 'add_iso_noise_to_neg': 'True', 'neg_prior_mean_coeff': 8 }

#crits['fashion_mnist_vs_mnist_baseline_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['fashion_mnist_vs_mnist_baseline_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0 }

#crits['mnist_vs_fashion_mnist_baseline_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['mnist_vs_fashion_mnist_baseline_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0 }

#crits['cifar10_vs_svhn_baseline_bernoulli'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['cifar10_vs_svhn_baseline_gaussian'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'model_architecture': 'dcgan_univ', 'trained_gamma': 'False', 'reg_lambda': 1.0, 'optimizer': 'rmsprop', 'encoder_use_bn': 'False'}

#crits['svhn_vs_cifar10_baseline_bernoulli'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0 }
#crits['svhn_vs_cifar10_baseline_gaussian'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0 }

#crits['check_fashion_mnist_vs_mnist_basseline'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'seed': [6, 7, 8, 9, 10], 'neg_dataset':'None', 'alpha_generated': 1.0}
#crits['svhn_vs_cifar10_bernoulli_bn_adam'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'seed': [6, 7, 8, 9, 10], 'neg_dataset':'None', 'alpha_generated': 0.0}

#crits['ablation_fashion_mnist_bernoulli_gen_adv_off_neg_gen'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'False' }
#crits['ablation_fashion_mnist_bernoulli_gen_adv_off_neg_fixed_gen'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True' }
#crits['ablation_fashion_mnist_gaussian_gen_adv_off_neg_gen'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'False' }
#crits['ablation_fashion_mnist_gaussian_gen_adv_off_neg_fixed_gen'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True' }

#crits['ablation_cifar10_bernoulli_gen_adv_off_neg_gen'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'False' }
#crits['ablation_cifar10_bernoulli_gen_adv_off_neg_fixed_gen'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'generator_adversarial_loss': 'False' }
#crits['ablation_cifar10_gaussian_gen_adv_off_neg_gen'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'False' }
#crits['ablation_cifar10_gaussian_gen_adv_off_neg_fixed_gen'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'generator_adversarial_loss': 'False' }

#crits['baseline_mnist_vs_letters_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'bernoulli'}
#crits['baseline_mnist_vs_letters_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'gaussian'}

#crits['baseline_letters_vs_mnist_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli'}
#crits['baseline_letters_vs_mnist_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian'}

#crits['baseline_fashion_mnist_vs_letters_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'bernoulli'}
#crits['baseline_fashion_mnist_vs_letters_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'gaussian'}

#crits['baseline_letters_vs_fashion_mnist_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli'}
#crits['baseline_letters_vs_fashion_mnist_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian'}

#crits['neg_fashion_mnist_mnist_vs_letters_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'fashion_mnist'}
#crits['neg_fashion_mnist_mnist_vs_letters_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'gaussian', 'neg_dataset': 'fashion_mnist'}

#crits['neg_fashion_mnist_letters_vs_mnist_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'fashion_mnist'}
#crits['neg_fashion_mnist_letters_vs_mnist_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'fashion_mnist'}

#crits['neg_adv_gen_fashion_mnist_vs_letters_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'bernoulli', 'alpha_generated': 1.0}
#crits['neg_adv_gen_fashion_mnist_vs_letters_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'gaussian', 'alpha_generated': 1.0}

#crits['neg_adv_gen_letters_vs_fashion_mnist_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'alpha_generated': 1.0}
#crits['neg_adv_gen_letters_vs_fashion_mnist_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'alpha_generated': 1.0}

#crits['neg_adv_gen_mnist_vs_fashion_mnist_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'alpha_generated': 1.0}
#crits['neg_adv_gen_mnist_vs_fashion_mnist_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'alpha_generated': 1.0}

#crits['neg_adv_gen_mnist_vs_letters_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'bernoulli', 'alpha_generated': 1.0}
#crits['neg_adv_gen_mnist_vs_letters_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'gaussian', 'alpha_generated': 1.0}

#crits['neg_adv_gen_letters_vs_mnist_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'alpha_generated': 1.0}
#crits['neg_adv_gen_letters_vs_mnist_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'alpha_generated': 1.0}

#crits['beta_neg_fashion_mnist_vs_mnist_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'beta_neg': 1.0}

#crits['neg_mnist_fashion_mnist_vs_letters_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'mnist'}
#crits['neg_mnist_fashion_mnist_vs_letters_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'gaussian', 'neg_dataset': 'mnist'}

#crits['neg_mnist_letters_vs_fashion_mnist_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'mnist'}
#crits['neg_mnist_letters_vs_fashion_mnist_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'mnist'}

#crits['beta_neg_letters_vs_mnist_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'beta_neg': 1.0, 'seed': [1,3,4,5]}

#crits['mnist_vs_fashion_mnist_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'emnist-letters', 'beta_neg': 0.0} # 'neg_prior_mean_coeff': 8}
#crits['mnist_vs_fashion_mnist_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'emnist-letters', 'beta_neg': 0.0} #'neg_prior_mean_coeff': 8}

#crits['cifar10_vs_svhn_neg_adv_bernoulli'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True', 'seed': [1,2,3,4,5]}
#crits['cifar10_vs_svhn_neg_adv_gaussian'] = {'beta': 1.0, 'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True', 'seed': [1,2,3,4,5], 'add_obs_noise': 'False'}
#crits['cifar10_vs_svhn_neg_adv_quantizedgaussian'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True', 'seed': [1,2,3,4,5], 'add_obs_noise': 'True'}
#crits['svhn_vs_cifar10_neg_adv_bernoulli'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True', 'seed': [1,2,3,4,5]}
#crits['svhn_vs_cifar10_neg_adv_gaussian'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True', 'seed': [1,2,3,4,5], 'add_obs_noise': 'False'}

#crits['svhn_vs_cifar10_neg_adv_quantizedgaussian'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None','alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True', 'seed': [1,2,3,4,5],'add_obs_noise': 'True'}

#crits['svhn_vs_cifar10_neg_imagenet_bernoulli'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'imagenet', 'alpha_generated': 0.0, 'encoder_use_sn': 'False', 'seed': [1,2,3,4,5]}
#crits['svhn_vs_cifar10_neg_imagenet_gaussian'] = {'test_dataset_a': 'svhn_cropped', 'test_dataset_b': 'cifar10', 'obs_noise_model': 'gaussian', 'neg_dataset': 'imagenet', 'alpha_generated': 0.0, 'encoder_use_sn': 'False', 'seed': [1,2,3,4,5]}

#crits['cifar_vs_svhn_neg_adv_bernoulli_sn_genadvlossoff'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}
#crits['cifar_vs_svhn_neg_adv_gaussian_sn_genadvlossoff'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}

#crits['cifar_vs_svhn_neg_adv_bernoulli_sn_genadvlossoff_fixedgen'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True'}
#crits['cifar_vs_svhn_neg_adv_gaussian_sn_genadvlossoff_fixedgen'] = {'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True'}

#crits['fashion_mnist_vs_mnist_neg_adv_bernoulli_sn'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True'}
#crits['fashion_mnist_vs_mnist_neg_adv_gaussian_sn'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'True'}

#crits['fashion_mnist_vs_mnist_neg_adv_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'False', 'generator_adversarial_loss': 'True'}
#crits['fashion_mnist_vs_mnist_neg_adv_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'False', 'generator_adversarial_loss': 'True'}

#crits['fashion_mnist_vs_mnist_neg_adv_bernoulli_sn_genadvlossoff'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}
#crits['fashion_mnist_vs_mnist_neg_adv_gaussian_sn_genadvlossoff'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}

#crits['fashion_mnist_vs_mnist_neg_adv_bernoulli_sn_genadvlossoff_fixedgen'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True'}
#crits['fashion_mnist_vs_mnist_neg_adv_gaussian_sn_genadvlossoff_fixedgen'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True'}

#crits['letters_vs_mnist_neg_adv_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}
#crits['letters_vs_mnist_neg_adv_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}

#crits['mnist_vs_letters_neg_adv_bernoulli'] = {'test_dataset_b': 'emnist-letters', 'test_dataset_a': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}
#crits['mnist_vs_letters_neg_adv_gaussian'] = {'test_dataset_b': 'emnist-letters', 'test_dataset_a': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}

#crits['letters_vs_mnist_neg_adv_bernoulli_genadvlossoff'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'False', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}
#crits['letters_vs_mnist_neg_adv_gaussian_genadvlossoff'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'False', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}

#crits['letters_vs_mnist_neg_adv_bernoulli_genadvlossoff_fixedgen'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'fixed_gen_as_negative': 'True'}
#crits['letters_vs_mnist_neg_adv_gaussian_genadvlossoff_fixedgen'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'fixed_gen_as_negative': 'True'}

#crits['beta_neg_grid'] = { 'beta_neg': 0.9, 'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'emnist-letters', 'neg_prior_mean_coeff': 8, 'base_filter_num': 32, 'beta': 1.0, 'latent_dim': 10, 'generator_adversarial_loss': 'True', 'fixed_gen_as_negative': 'False', 'seed': [1, 2, 3, 4, 5] }

##############################################################################################

#crits['beta_neg_0.9'] = {'exp_id': ['ON-1919']}
#crits['beta_neg_0.8'] = {'exp_id': ['ON-1914']}
#crits['beta_neg_0.7'] = {'exp_id': ['ON-1905']}
#crits['beta_neg_0.6'] = {'exp_id': ['ON-1900']}
#crits['beta_neg_0.5'] = {'exp_id': ['ON-1895']}
#crits['beta_neg_0.4'] = {'exp_id': ['ON-1893']}
#crits['beta_neg_0.3'] = {'exp_id': ['ON-1891']}
#crits['beta_neg_0.2'] = {'exp_id': ['ON-1889']}
#crits['beta_neg_0.1'] = {'exp_id': ['ON-1887']}
#crits['beta_neg_0.0'] = {'exp_id': ['ON-1885']}

#crits['ld=50'] = {'exp_id': ['ON-2137']}
#crits['ld=100'] = {'exp_id': ['ON-2327']}
#crits['ld=250'] = {'exp_id': ['ON-2126']}
#crits['ld=500'] = {'exp_id': ['ON-2109']}

#crits['svhn_vs_cifar_quantized_negadv'] = {'exp_id': ['ON-2395', 'ON-2400', 'ON-2402', 'ON-2407', 'ON-2409']}

#crits['letters_vs_mnist_bernoulli_negadv'] = {'exp_id': ['ON-1960', 'ON-1964', 'ON-1968', 'ON-1972', 'ON-1976', 'ON-2674', 'ON-2678', 'ON-2684', 'ON-2690', 'ON-2696']}

#exps = project.get_experiments(crits[list(crits.keys())[0]]['exp_id'])

################################################################################################

#permutations
#baseline

#negaux

#negadv
#crits['fashionmnist_vs_mnist_negadv_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'False', 'generator_adversarial_loss': 'True'}
#crits['fashionmnist_vs_mnist_negadv_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'False', 'generator_adversarial_loss': 'True'}

#crits['fashionmnist_vs_letters_negadv_bernoulli'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None',  'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}
#crits['fashionmnist_vs_letters_negadv_gaussian'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'emnist-letters', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}

#crits['mnist_vs_fashionmnist_negadv_bernoulli'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None',  'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}
#crits['mnist_vs_fashionmnist_negadv_gaussian'] = {'test_dataset_a': 'mnist', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None',  'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}

#crits['mnist_vs_letters_negadv_bernoulli'] = {'test_dataset_b': 'emnist-letters', 'test_dataset_a': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}
#crits['mnist_vs_letters_negadv_gaussian'] = {'test_dataset_b': 'emnist-letters', 'test_dataset_a': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}

#crits['letters_vs_fashionmnist_negadv_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}
#crits['letters_vs_fashionmnist_negadv_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'fashion_mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}

#crits['letters_vs_mnist_negadv_bernoulli'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}
#crits['letters_vs_mnist_negadv_gaussian'] = {'test_dataset_a': 'emnist-letters', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'True'}


#adversarial_loss_ablation

crits['ablation_fashion_mnist_bernoulli_gen_adv_off_neg_gen'] = {'tags': 'FIN_genadvlossablation', 'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False' }
#crits['ablation_fashion_mnist_bernoulli_gen_adv_off_neg_fixed_gen'] = {'tags': ['FIN_genadvlossablation'], 'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True' }
#crits['ablation_fashion_mnist_gaussian_gen_adv_off_neg_gen'] = {'tags': ['FIN_genadvlossablation'], 'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'generator_adversarial_loss': 'False' }
#crits['ablation_fashion_mnist_gaussian_gen_adv_off_neg_fixed_gen'] = {'tags': ['FIN_genadvlossablation'], 'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 0.0, 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True' }

#crits['ablation_cifar_vs_svhn_neg_adv_bernoulli_sn_genadvlossoff'] = {'tags': ['FIN_genadvlossablation'], 'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}
#crits['ablation_cifar_vs_svhn_neg_adv_gaussian_sn_genadvlossoff'] = {'tags': ['FIN_genadvlossablation'], 'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'False'}
#crits['ablation_cifar_vs_svhn_neg_adv_bernoulli_sn_genadvlossoff_fixedgen'] = {'tags': ['FIN_genadvlossablation'], 'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True'}
#crits['ablation_cifar_vs_svhn_neg_adv_gaussian_sn_genadvlossoff_fixedgen'] = {'tags': ['FIN_genadvlossablation'], 'test_dataset_a': 'cifar10', 'test_dataset_b': 'svhn_cropped', 'obs_noise_model': 'gaussian', 'neg_dataset': 'None', 'alpha_generated': 1.0, 'encoder_use_sn': 'True', 'generator_adversarial_loss': 'False', 'fixed_gen_as_negative': 'True'}




channels = ['auc_bpd', 'auc_kl'] #,'test_bpd_a', 'test_bpd_b']

exps = project.get_experiments()


for exp in exps:
  exp._my_params = exp.get_parameters()
  exp._my_tags = exp.get_tags()

for row in [99]:
  for key, crit in crits.items():
    res = defaultdict(list)
    for exp in exps:
      params = exp._my_params       #{}
      params['exp_id'] = exp._id
      params['tags'] = exp._my_tags
      if not check_crit(params, crit):
        continue
      else:
        print(exp._id)
        print('Tags: {} \n'.format(params['tags']))
        #vals = exp.get_logs()
        #for channel in channels:
          #if channel in vals:
            #res[channel].append(float(vals[channel]['y']))

        #postfix = params['test_dataset_a'] + '_vs_' + params['test_dataset_b']
        #res['auc_kl'].append(float(vals['auc_kl_'+postfix]['y']))
        #res['auc_mean'].append(float(vals['auc_mean_'+postfix]['y']))

        postfix = '_' + params['test_dataset_a'] + '_vs_' + params['test_dataset_b']
        try:
          auc_df = exp.get_numeric_channels_values('auc_bpd', 'auc_kl'+postfix)  #, 'test_bpd_a', 'test_bpd_b')

          if len(auc_df) > row and not pd.isna(auc_df.loc[row, 'auc_bpd']) and not pd.isna(auc_df.loc[row, 'auc_kl'+postfix]):
            last = row
          else:
            last = len(auc_df) - 1
            print('Exp does not contain {} data points, taking the last one: {}.'.format(row, last))

          auc_bpd = auc_df.loc[last, 'auc_bpd']
          auc_kl = auc_df.loc[last, 'auc_kl'+postfix]
          res['auc_bpd'].append(float(auc_bpd))
          res['auc_kl'].append(float(auc_kl))
        except KeyError:
          print('KeyError, skipping this exp.')

    for channel in channels:
      v = np.array(res[channel])
      if v.shape[0] > 10:
        print("{}: Warning: more than 10 exeriments: {} using only 10 (no order assumed)".format(key, v.shape[0]))
      v = v[:10]
      print('{}   {}'.format(channel, v))
      #print('before: {}'.format(v))
      #v = v/np.log(2)
      #print('after dividing with log(2): {}'.format(v))
      mean = np.mean(v)
      std = np.std(v)
      cnt = v.shape[0]
      print("{}   {: <25} {} mean: {:.2f}, std: {:.2f}, cnt: {}".format(row, key, channel, mean, std, cnt))
    print('\n')