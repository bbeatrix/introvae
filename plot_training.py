import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

couts = [
    {'filename': 'cifar10_seed=0_class=0_m=60.cout', 'class': 0, 'sota': 0.671},
    {'filename': 'cifar10_seed=0_class=1_m=60.cout', 'class': 1, 'sota': 0.659},
    {'filename': 'cifar10_seed=0_class=2_m=60.cout', 'class': 2, 'sota': 0.529},
    {'filename': 'cifar10_seed=0_class=3_m=60.cout', 'class': 3, 'sota': 0.591},
    {'filename': 'cifar10_seed=0_class=4_m=60.cout', 'class': 4, 'sota': 0.662},
    {'filename': 'cifar10_seed=0_class=5_m=60.cout', 'class': 5, 'sota': 0.657},
    {'filename': 'cifar10_seed=0_class=6_m=60.cout', 'class': 6, 'sota': 0.749},
    {'filename': 'cifar10_seed=0_class=7_m=60.cout', 'class': 7, 'sota': 0.673},
    {'filename': 'cifar10_seed=0_class=8_m=60.cout', 'class': 8, 'sota': 0.768},
    {'filename': 'cifar10_seed=0_class=9_m=60.cout', 'class': 9, 'sota': 0.76}
]

for cout in couts:
    xs = []
    ys = []
    with open(cout['filename']) as f:
        x=0
        plt.clf()
        fig, ax = plt.subplots()
        for line in f:
            if 'AUC' in line:
                parts = line.split()
                x += 1
                y = float(parts[1])
                xs.append(x)
                ys.append(y)
        ax.plot(xs, ys)
        ax.hlines(y=cout['sota'], xmin=1, xmax=200)
        plt.savefig(cout['filename']+".png")

