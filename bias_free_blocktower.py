"""
COPHY=/storage/Datasets/CoPhy/CoPhy_224 # root location of CoPhy benchmark
python bias_free_blocktower.py --dir $COPHY/blocktowerCF/3 # for blocktower compose dof 3 blocks
python bias_free_blocktower.py --dir $COPHY/blocktowerCF/4 # and for blocktower compose dof 4 blocks

Totla number of examples:
* 33k for H=3
* 40k for H=4
"""

import os
import time
import ipdb
import numpy as np
from tqdm import *
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Stats for making sure that the dataset if highly-balanced.')
parser.add_argument('--dir',
                    default='/storage/Datasets/CoPhy/CoPhy_224/blocktowerCF/3',
                    type=str,
                    help='loc of files')

COLORS = ['red', 'green', 'blue', 'yellow']

def main(args):
  # Retrieve examples
  num_ex = 0
  list_dir = [x for x in os.listdir(args.dir) if
              os.path.isdir(os.path.join(args.dir, x))]
  list_masses_stab, list_masses_unstab = [], []
  list_frictions_stab, list_frictions_unstab = [], []
  list_gravities_stab, list_gravities_unstab = [], []
  list_stab_abcd = []
  list_stab = []
  for x in tqdm(list_dir):

    # gravities (env)
    with open(os.path.join(args.dir, x, 'gravity.txt'), 'r') as f1:
      gravities = f1.readline().strip()

    # confounders according to color ordering
    confounders = np.load(os.path.join(args.dir, x, 'confounders.npy'))

    # colors from bottom to top
    with open(os.path.join(args.dir, x, 'cd', 'colors.txt'), 'r') as f:
      colors = f.readlines() # from bottom to top
    colors = [x.strip().split('color=')[-1] for x in colors[1:]]

    # re-arrange
    masses, frictions = [],  []
    for col in colors:
      idx_col = COLORS.index(col)
      masses.append(confounders[idx_col, 0])
      frictions.append(confounders[idx_col, 1])
    masses = '-'.join([str(x) for x in masses])
    frictions = '-'.join([str(x) for x in frictions])

    # stab
    states_ab = np.load(os.path.join(args.dir, x, 'ab', 'states.npy'))
    states_cd = np.load(os.path.join(args.dir, x, 'cd', 'states.npy'))
    stab_ab = int(np.sum(np.abs(states_ab[0] - states_ab[-1])) < 0.05)
    stab_cd =  int(np.sum(np.abs(states_cd[0] - states_cd[-1])) < 0.05) # 1=stab, 0=unstab

    if stab_cd == 1:
      list_masses_stab.append(masses)
      list_frictions_stab.append(frictions)
      list_gravities_stab.append(gravities)
    else:
      list_masses_unstab.append(masses)
      list_frictions_unstab.append(frictions)
      list_gravities_unstab.append(gravities)

    list_stab_abcd.append(str(stab_ab) + str(stab_cd))
    list_stab.append(stab_cd)
    num_ex += 1

  print("Num examples in total: ", num_ex)

  # Stats per confounders - should be around ~50%
  print("*** Mass of each block (from bottom to top) ***")
  get_stats_masses = compute_stats(list_masses_stab, list_masses_unstab)
  show_dict(get_stats_masses)

  print("*** Friction coefficient of each block (from bottom to top) ***")
  get_stats_frictions = compute_stats(list_frictions_stab,
                                      list_frictions_unstab)
  show_dict(get_stats_frictions)

  print("*** X an Y gravity of the scene ***")
  get_stats_gravities = compute_stats(list_gravities_stab,
                                      list_gravities_unstab)
  show_dict(get_stats_gravities)

  print("*** Repartition of (A,B)->(C,D) scenarios (e.g. '00' means that both (A,B) and (C,D) are unstable) ***")
  for x in list(set(list_stab_abcd)):
    print("{} -> {:.2f}".format(x, 100. * list_stab_abcd.count(x) / len(
      list_stab_abcd)))

  print(f"\nPerc stab = {np.sum(list_stab)/len(list_stab)}")


def compute_stats(list_stab, list_unstab):
  assert len(set(list_stab)) == len(set(list_unstab))

  possible_v = list(set(list_stab))
  _dict = {}

  for v in possible_v:
    num_stab = list_stab.count(v)
    num_unstab = list_unstab.count(v)
    stat = num_stab / (num_stab + num_unstab)
    _dict[v] = stat
  return _dict


def show_dict(mydict):
  print()
  for k, v in mydict.items():
    print("{}: {:.2f}".format(k, v * 100.))


if __name__ == "__main__":
  # Args
  args = parser.parse_args()
  print(args)
  print("")

  start = time.time()
  main(args)
