import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
import os
from collections import defaultdict

plt.style.use('seaborn-deep')
sns.set()

def check_crit(params, crit):
    for k, v in crit.items():
        if k not in params.keys():
            return False
        if isinstance(v, list):
            if params[k] not in v:
                return False
        elif isinstance(params[k], list):
            if v not in params[k]:
                return False
        elif params[k] != v:
            return False
    return True


def get_exp_results(project, crits, channels):
    print('Get exps of {} project...'.format(project))
    exps = project.get_experiments()
    print('Number of exps: ', len(exps))

    print('Checking crits...')
    all_results = defaultdict(list)

    for key, crit in crits.items():
        results = defaultdict(list)
        for exp in exps:
            print('Checking exp ', exp._id)
            params = exp.get_parameters()
            params.update(exp.get_system_properties())
            params['state'] = exp.state
            if not check_crit(params, crit):
                continue
            else:
                print(exp._id)

            for channel in channels: 
                if channel in params.keys():
                    result = params[channel]
                else:
                    result_df = exp.get_numeric_channels_values(channel)
                    result = float(result_df.loc[len(result_df) - 1, channel])
                results[channel].append(result)
        all_results[key] = results

    return all_results

def plot_heatmap(data, x_axis_label, y_axis_label, value_label, query_name):
    df = pd.DataFrame.from_dict(data).drop_duplicates(subset=[x_axis_label, y_axis_label], keep='last')
    df = df.pivot(y_axis_label, x_axis_label, value_label)

    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(df, annot=True, linewidths=.5, vmin=0, vmax=1)
    plt.title(query_name, pad=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(query_name + '.png', bbox_inches='tight')
    plt.clf()


def define_crits():
    crits = {}
    crits['ns_negadv_1'] = {'eubo_gen_lambda': 1.0, 'alpha_adv_gen': [0.1, 1.0, 10], 'beta_adv_gen': [0.0, 0.1, 1.0, 10], 'state': 'succeeded'}
    crits['ns_negadv_2'] = {'eubo_gen_lambda': 0.1, 'alpha_adv_gen': [0.1, 1.0, 10], 'beta_adv_gen': [0.0, 0.1, 1.0, 10], 'state': 'succeeded'}
    crits['ns_negadv_3'] = {'eubo_gen_lambda': 10.0, 'alpha_adv_gen': [0.1, 1.0, 10], 'beta_adv_gen': [0.0, 0.1, 1.0, 10], 'state': 'succeeded'}
    return crits


def main():
    session = neptune.Session.with_default_backend()
    project = session.get_project('csadrian/oneclass') 
    crits = define_crits()
    channels = ['alpha_adv_gen', 'beta_adv_gen', 'auc_bpd']
    results = get_exp_results(project, crits, channels)
    for query_name, query_result in results.items():
        plot_heatmap(query_result,
                     *channels,
                     query_name=str(crits[query_name]))
    print('Fin')


if __name__ == "__main__":
    main()
