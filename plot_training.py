import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sotas = {
    0: 0.671,
    1: 0.659,
    2: 0.529,
    3: 0.591,
    4: 0.662,
    5: 0.657,
    6: 0.749,
    7: 0.673,
    8: 0.768,
    9: 0.76
}


import json
from os import listdir
from os.path import isfile, join

path = "/mnt/g2big/oneclass/dcgan/"

onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.cout')]

couts = []
for f in onlyfiles:
    rec = { 'filename': f, 'filepath': join(path, f) }
    parts = f.split('_')
    for part in parts:
        parts2 = part.split('=')
        if len(parts2) == 2:
            rec[parts2[0]] = parts2[1]
    rec['class'] = rec['normclass']
    f = open(rec['filepath'])
    lines = f.readlines()
    if len(lines) == 0:
      continue
    print(len(lines))
    #rec = eval(lines[0])
    couts.append(rec)
print(couts)

"""
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
"""

def get_color(rec):
  if rec['m'] == '40':
   return 'red'
  elif rec['m'] == '50' and rec['bf'] == '64':
   return 'orange'
  elif rec['m'] == '50' and rec['bf'] == '32':
   return 'yellow'
  else:
   return 'blue'

def byclass():
    for i in range(10):
        res = []
        for cout in couts:
            if int(cout['class']) == i:
                res.append(cout)
        yield res

for q in byclass():
  plt.clf()
  fig, ax = plt.subplots()

  for cout in q:
    xs = []
    ys = []
    with open(cout['filepath']) as f:
        x=0
        for line in f:
            if 'AUC' in line:
                parts = line.split()
                x += 1
                y = float(parts[1])
                xs.append(x)
                ys.append(y)
        #ax.plot(xs, ys, label=( cout['lrs'] if 'lrs' in cout.keys()   else  'constant')+cout['m']+cout['bf'], color=get_color(cout))
        ax.plot(xs, ys, label=cout['filename'], color=get_color(cout))
        ax.hlines(y=sotas[int(cout['class'])], xmin=1, xmax=100)
  ax.legend()
  plt.savefig(str(q[0]['class'])+".png")

