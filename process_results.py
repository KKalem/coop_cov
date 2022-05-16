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


if __name__ == '__main__':
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

    # remove the failed exps
    exps = exps.query('experiment_finished == True')

    print(exps['runtime'].describe())

    # add some new columns
    exps.insert(0, 'coverage_ratio', exps['intended_covered_area']/exps['intended_area'])
    exps.insert(0, 'efficiency', exps['intended_covered_area']/exps['total_travel'])
    # eff over coverage ratio, call it performance
    exps.insert(0, 'performance', exps['efficiency']/exps['coverage_ratio'])



    variables = ['area_width',
                 'area_height',
                 'comm_range',
                 'num_agents',
                 'num_landmarks',
                 'plan_type',
                 'k']

    plt.ion()
    plt.figure()
    markers = ['x', '^', '+', 'v', 'o', '.', '-']

    ###########################################################################################
    # "Taking k into consideration increases the performance through dubins"
    ###########################################################################################
    query = [
        # f"intended_area == {np.unique(exps['intended_area'])[2]}",
        f'area_width == 600',
        f'area_height == 600',
        f"num_agents == 3",
        f"num_landmarks == 0",
        # f"coverage_ratio > 0.5"
            ]
    query_str = ' & '.join(query)
    common = exps.query(query_str)

    x_col = 'k'
    y_col = 'performance'

    labels = ['simple, no-comm', 'simple, coop', 'dubins, no-comm', 'dubins, coop']
    offsets = [0.002, 0.004, -0.002, -0.004]

    for plan_type, comm_range, label, marker, offset in zip([0,0,1,1], [0,50,0,50], labels, markers, offsets):
        d = common.query(f'plan_type == {plan_type} & comm_range == {comm_range}')

        if plan_type == 1:
            for ku in np.unique(d['kept_uncertainty']):
                dd = d.query(f'kept_uncertainty == {ku}')
                plt.scatter(dd[x_col]+offset, dd[y_col], marker=marker, alpha=0.5, label=label+f' {ku}')
        else:
            plt.scatter(d[x_col]+offset, d[y_col], marker=marker, alpha=0.5, label=label)

    plt.xlabel(x_col)
    if len(np.unique(exps[x_col])) < 10:
        plt.xticks(ticks=np.unique(exps[x_col]))

    plt.ylabel(y_col)
    if len(np.unique(exps[y_col])) < 10:
        plt.xticks(ticks=np.unique(exps[y_col]))

    plt.title(query_str)
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ###########################################################################################





