import matplotlib.pyplot as plt
from collections import defaultdict
from neptune import Session
import numpy as np

session = Session()
project = session.get_projects('csadrian')['csadrian/oneclass']

def check_crit(exp, crit):
  params = exp.get_parameters()
  for k, v in crit.items():
    if params[k] != v:
      return False
  return True

crits = {}
crits['szepkep_letters'] = {'test_dataset_a': 'fashion_mnist', 'test_dataset_b': 'mnist', 'obs_noise_model': 'bernoulli', 'neg_dataset': 'emnist-letters' }

channels = ['auc_bpd']
exps = project.get_experiments()
for key, crit in crits.items():
  res = defaultdict(list)
  for exp in exps:
    if not check_crit(exp, crit):
      continue
    vals = exp.get_logs()
    for channel in channels:
        res[channel].append(float(vals[channel]['y']))
  for channel in channels:
    v = np.array(res[channel])
    mean = np.mean(v)
    std = np.std(v)
    cnt = v.shape[0]
    print(key, "mean: ", mean, ", std:", std, ", cnt:", cnt)
