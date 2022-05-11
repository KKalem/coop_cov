#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
import pandas as pd

from tqdm import tqdm

import yaml, sys, time, uuid, glob, os

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from experiments import Experiment


results_dir = 'experiments/'
yaml_files = glob.glob(results_dir+"*.yaml")

labels = []
exps = []

for yaml in tqdm(yaml_files):
    e = Experiment()
    e.load(yaml)
    labels, row = e.as_row()
    exps.append(row)

exps = pd.DataFrame(data = exps,
                    columns = labels)

print(exps['runtime'].describe())


def efficiencies(d):
    return d['intended_covered_area'] / d['total_travel']


def num_holes(d):
    def len_none(x):
        try:
            return len(x)
        except:
            return 0
    return d['missed_lenwidths'].apply(len_none)


def max_min_holes(d):
    def maxmin(x):
        try:
            return max([min(h) for h in x])
        except:
            return 0
    return d['missed_lenwidths'].apply(maxmin)


plt.ion()

plt.figure()
num_agents = 6

x_col = 'intended_area'
d = exps.query(f'plan_type == 0 & num_agents == {num_agents}')
plt.scatter(d[x_col], efficiencies(d), alpha=0.1, label=f'simple', marker='x')

for ku in np.unique(exps['kept_uncertainty']):
    d = exps.query(f'plan_type == 1 & num_agents == {num_agents} & kept_uncertainty == {ku}')
    plt.scatter(d[x_col], efficiencies(d), alpha=0.1, label=f'dubins-{ku}')

plt.xlabel(x_col)
plt.ylabel('eff $[m^2/m]$')
plt.legend()





