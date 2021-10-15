#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection
from matplotlib.textpath import TextPath
from descartes import PolygonPatch
plt.rcParams['pdf.fonttype'] = 42

from multiprocessing import Pool
import json
from itertools import product
import sys

import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon
from tqdm import tqdm

from toolbox import geometry as geom
from mission_plan import MissionPlan
from pose_graph import PoseGraph, PGO_VertexIdStore
from auv import AUV
from mission_plan import construct_lawnmower_paths
from auv_agent import Agent


class SimConfig:
    def __init__(self,
                 communication,
                 summarize_pg,
                 num_auvs,
                 num_hooks,
                 hook_len,
                 gap_between_rows,
                 overlap_between_lanes,
                 swath,
                 beam_radius,
                 target_threshold,
                 comm_dist,
                 dt,
                 seed,
                 std_shift,
                 drift_mag,
                 interact_period_ticks,
                 max_ticks):

        # just read everything into object vars
        self.args = locals()
        for k,v in self.args.items():
            self.__dict__[k] = v

    def save_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.args, f)



def construct_sim_objects(config):
    id_store = PGO_VertexIdStore()

    paths = construct_lawnmower_paths(num_agents = config.num_auvs,
                                      num_hooks = config.num_hooks,
                                      hook_len = config.hook_len,
                                      swath = config.swath,
                                      gap_between_rows = config.gap_between_rows,
                                      overlap_between_lanes = config.overlap_between_lanes,
                                      double_sided=False)


    init_poss = [p[0]-[config.swath,0] for p in paths]


    auvs = []
    pgs = []
    agents = []
    for i,path in enumerate(paths):
        auv = AUV(auv_id=i,
                  init_pos = init_poss[i],
                  init_heading = 0.,
                  target_threshold = config.target_threshold)


        pg = PoseGraph(pg_id = i,
                       id_store = id_store)

        agent = Agent(real_auv = auv,
                      pose_graph = pg,
                      waypoints = path)


        auvs.append(auv)
        pgs.append(pg)
        agents.append(agent)

    return auvs, pgs, agents, paths



def run_sim(config, agents, consistent_drifts, paths, communication=True):
    done = False
    t = 0
    print("Running...")
    with tqdm(total=paths.shape[0]*paths.shape[1]) as pbar:
        while True:
            t += 1
            pbar.desc = f'Tick:{t}'
            # move first
            for i,agent in enumerate(agents):
                # for the first WP, assume surface travel
                # this just makes the plots look better
                if agent.current_wp_idx == 0:
                    drift_x = 0.
                    drift_y = 0.
                # same for the LAST WP too
                elif agent.current_wp_idx == len(agent.waypoints)-1:
                    drift_x = 0.
                    drift_y = 0.
                else:
                    # enviromental drift
                    drift_x = np.random.normal(0, config.std_shift) + consistent_drifts[i][0]
                    drift_y = np.random.normal(0, config.std_shift) + consistent_drifts[i][1]

                agent.update(config.dt,
                             drift_x = drift_x,
                             drift_y = drift_y)

                if agent.internal_auv.reached_target:
                    pbar.update(1)

            if communication:
                # interact second
                if t%5 == 0:
                    for agent in agents:
                        agent.communicate(agents,
                                          config.comm_dist,
                                          config.summarize_pg)

            # check if done
            paths_done = [agent.waypoints_exhausted for agent in agents]
            if all(paths_done):
                print("Paths done")
                break

            if t >= config.max_ticks:
                print("Max ticks reached")
                break

    print("...Done")
    errs = [a.distance_traveled_error() for a in agents]
    print(f"Distance traveled errors: {errs}")

    return errs, t


def plot_coverage(ax,
                  config,
                  paths,
                  agents,
                  pgs,
                  auvs,
                  consistent_drifts,
                  coverage_polies,
                  plot_pg=False):
    print("Plotting...")
    try:
        __IPYTHON__
        plt.ion()
    except:
        pass
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']

    # plot the plan
    # for path in paths:
        # p = np.array(path)
        # ax.plot(p[:,0], p[:,1], c='k', alpha=0.2, linestyle=':')

    for c, polies in zip(colors, coverage_polies):
        for poly in polies:
            ax.add_artist(PolygonPatch(poly, alpha=0.08, fc=c, ec=c))

    for c, agent, pg, auv in zip(colors, agents, pgs, auvs):

        ax.plot(auv.pose_trace[:,0], auv.pose_trace[:,1], c=c, alpha=0.5)
        ax.plot(agent.internal_auv.pose_trace[:,0], agent.internal_auv.pose_trace[:,1],
                c=c, alpha=0.5, linestyle=':')


        t = pg.odom_pose_trace
        # ax.scatter(t[:,0], t[:,1], alpha=0.2, marker='.', s=5, c=c)
        if plot_pg:
            for p in pg.fixed_poses:
                ax.text(p[0], p[1], 'F', c=c)

            for edge in pg.measured_edges:
                p1 = edge.parent_pose
                p2 = edge.child_pose
                diff = p2-p1
                ax.arrow(p1[0], p1[1], diff[0], diff[1],
                          alpha=0.3, color=c, head_width=0.5, shape='left',
                          length_includes_head=True)

            for edge in pg.self_odom_edges:
                p1 = edge.parent_pose
                p2 = edge.child_pose
                diff = p2-p1
                ax.arrow(p1[0], p1[1], diff[0], diff[1],
                          alpha=0.2, color=c, head_width=0.3, length_includes_head=True)

            for edge in pg.self_summary_edges:
                p1 = edge.parent_pose
                p2 = edge.child_pose
                diff = p2-p1
                plt.arrow(p1[0], p1[1], diff[0], diff[1],
                          alpha=0.1, color='k', head_width=7, length_includes_head=True)

            pg_marker = ' '*pg.pg_id + 'f'
            for p in pg.foreign_poses:
                tp = TextPath(p[:2], pg_marker, size=0.1)
                ax.add_patch(PathPatch(tp, color=c))

            for edge in pg.foreign_edges:
                p1 = edge.parent_pose
                p2 = edge.child_pose
                p = (p2+p1)/2
                tp = TextPath(p[:2], pg_marker, size=0.1)
                ax.add_patch(PathPatch(tp, color=c))


    for agent, path, d, c in zip(agents, paths, consistent_drifts, colors):
        _, nd = geom.vec_normalize(d)
        nd *= config.swath*0.6
        # x,y = agent.internal_auv.pose_trace[0][:2]
        x,y = path[0,0], np.mean(path[:,1])
        x -= (1.5 * config.swath)
        ax.arrow(x, y, nd[0], nd[1],
                  color = c,
                  length_includes_head = True,
                  width = config.swath/20)
        ax.scatter(x- (1.5*config.swath) ,y, alpha=0)


def run_try(config, plot=True, show_plot=False, save_plot=True):
    try:
        return run(config, plot, show_plot, save_plot)
    except:
        print(">>>>>>>>   RUN FAILED")
        print("Config:")
        print(config.__dict__)
        print("<<<<<<<<")
        return None



def run(config, plot=True, show_plot=False, save_plot=True):
    np.random.seed(config.seed)
    print(f"Running with config {config.__dict__}")

    consistent_drifts = []
    for i in range(config.num_auvs):
        _, d = geom.vec_normalize((np.random.random(2)-0.5)*2)
        consistent_drifts.append(d * config.drift_mag)


    # run the sim
    auvs, pgs, agents, paths = construct_sim_objects(config)
    errs, ticks = run_sim(config = config,
                          agents = agents,
                          consistent_drifts = consistent_drifts,
                          paths = paths,
                          communication = config.communication)

    # collect the communication done between auvs
    verts_received = {}
    edges_received = {}
    for auv, agent in zip(auvs, agents):
        verts_received[auv.auv_id] = agent.received_data['verts']
        edges_received[auv.auv_id] = agent.received_data['edges']


    # create the polygons for the coveages and such
    coverage_polies = []
    for auv in auvs:
        polies = auv.coverage_polygon(config.swath,
                                      shapely = True,
                                      beam_radius = config.beam_radius)
        coverage_polies.append(polies)
    flat_coverage_polies = [item for sublist in coverage_polies for item in sublist]

    predicted_polies = []
    for auv,pg in zip(auvs, pgs):
        polies = auv.coverage_polygon(swath = config.swath,
                                      pg = pg,
                                      shapely = True,
                                      beam_radius = config.beam_radius)
        predicted_polies.append(polies)
    flat_predicted_polies = [item for sublist in predicted_polies for item in sublist]


    true_positive_percents = []
    for reals, predicteds in zip(coverage_polies, predicted_polies):
        real_poly = unary_union(reals)
        predicted_poly = unary_union(predicteds)
        tp_poly = real_poly & predicted_poly
        tp_area = tp_poly.area
        pred_area = predicted_poly.area
        percent = 100* (tp_area / pred_area)
        if percent > 100:
            true_positive_percents.append(None)
            print(f"\n\n>>>> Weird percentage, {config.__dict__}, \n{tp_area}, {pred_area} \n\n")
        else:
            true_positive_percents.append(percent)


    intended_coverages = []
    for path in paths:
        # ignore the last point because its a "get out" point, no coverage there
        minx, miny = np.min(path[:-1], axis=0)
        maxx, maxy = np.max(path[:-1], axis=0)
        # not just the path, but _around_ the path
        # assuming the coverage is done east to west
        minx -= config.swath/2
        maxx += config.swath/2
        rect = Polygon(shell=[(minx, miny),
                              (minx, maxy),
                              (maxx, maxy),
                              (maxx, miny)])
        intended_coverages.append(rect)

    # calculate some stats for the mission
    intended_coverage = unary_union(intended_coverages)
    actually_covered = unary_union(flat_coverage_polies)
    predicted_coverage = unary_union(flat_predicted_polies)
    missed_area = intended_coverage - actually_covered
    unplanned_area = actually_covered - intended_coverage
    coverage_within_plan = actually_covered & intended_coverage
    planned_coverage_percent = 100* coverage_within_plan.area / intended_coverage.area
    travels = [a.distance_traveled for a in auvs]
    total_travel = sum(travels)
    # accuracy of predicted coverage?
    true_positive_covered = actually_covered & predicted_coverage
    # false_positive_covered = predicted_coverage.difference(actually_covered)
    true_positive_percent = 100* (true_positive_covered.area / predicted_coverage.area)
    print(f"Missed = {missed_area.area} m2")
    print(f"Covered = {planned_coverage_percent}% of the planned area")
    print(f"Total travel = {total_travel} m")
    print(f"True positive percent = {true_positive_percent}%")


    results = {
        "missed":missed_area.area,
        "covered_percent":planned_coverage_percent,
        "total_travel":total_travel,
        "covered_inside":coverage_within_plan.area,
        "final_distance_traveled_errs":errs,
        "true_positive_percent":true_positive_percent,
        "true_positive_percents":true_positive_percents,
        "travels":travels,
        "verts_received":verts_received,
        "edges_received":edges_received,
        "num_hooks":config.num_hooks,
        "hook_len":config.hook_len
    }


    if plot:
        # FIGURES WOOO
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        fig.set_size_inches(10,5)

        # axs[1].add_artist(PolygonPatch(intended_coverage, alpha=0.5, fc='grey', ec='grey'))
        axs[1].add_artist(PolygonPatch(actually_covered, alpha=0.5, fc='green', ec='green'))
        axs[1].add_artist(PolygonPatch(missed_area, alpha=0.5, fc='red', ec='red'))
        axs[1].add_artist(PolygonPatch(unplanned_area, alpha=0.5, fc='blue', ec='blue'))
        # axs[1].add_artist(PolygonPatch(false_positive_covered, alpha=0.5, hatch='x'))
        for agent, pg, auv in zip(agents, pgs, auvs):
            axs[1].plot(auv.pose_trace[:,0], auv.pose_trace[:,1], c='black', alpha=0.5)


        plot_coverage(axs[0],
                      config=config,
                      paths=paths,
                      agents=agents,
                      pgs=pgs,
                      auvs=auvs,
                      coverage_polies=coverage_polies,
                      consistent_drifts=consistent_drifts,
                      plot_pg=False)


        axs[0].set_xlabel('x[m]')
        axs[1].set_xlabel('x[m]')
        axs[0].set_ylabel('y[m]')

        try:
            __IPYTHON__
            plt.ion()
            plt.show()
        except:
            pass

        if save_plot:
            fig.savefig(f"seed={config.seed}_comm={config.communication}.pdf", bbox_inches='tight')

    return results


def make_config(seed,
                comm,
                summarize_pg = True,
                num_auvs = 6,
                num_hooks = 5,
                hook_len = 100,
                gap_between_rows = -5,
                overlap_between_lanes = 3,
                max_ticks = None):

    max_ticks = num_hooks * hook_len * 50

    config = SimConfig(
        num_auvs = num_auvs,
        num_hooks = num_hooks,
        hook_len = hook_len,
        gap_between_rows = gap_between_rows,
        overlap_between_lanes = overlap_between_lanes,
        swath = 50,
        beam_radius = 1,
        target_threshold = 3,
        comm_dist = 50,
        dt = 0.5,
        seed = seed,
        std_shift = 0.4,
        drift_mag = 0.02,
        interact_period_ticks = 5,
        max_ticks = max_ticks,
        communication = comm,
        summarize_pg = summarize_pg
    )
    return config

def singlify_config(config):
    sconfig = SimConfig(
        num_auvs = 1,
        num_hooks = config.num_hooks,
        hook_len = config.num_auvs * config.hook_len,
        gap_between_rows = config.gap_between_rows,
        overlap_between_lanes = config.overlap_between_lanes,
        swath = config.swath,
        beam_radius = config.beam_radius,
        target_threshold = config.target_threshold,
        comm_dist = config.comm_dist,
        dt = config.dt,
        seed = config.seed,
        std_shift = config.std_shift,
        drift_mag = config.drift_mag,
        interact_period_ticks = config.interact_period_ticks,
        max_ticks = config.max_ticks,
        communication = config.communication,
        summarize_pg = config.summarize_pg
    )
    return sconfig


def plot_violins(comm_results, nocomm_results, single_results):
    comm_percents = [r['covered_percent'] for r in comm_results]
    nocomm_percents = [r['covered_percent'] for r in nocomm_results]
    plt.figure()
    plt.violinplot([comm_percents, nocomm_percents], showmedians=True)
    plt.ylabel('Percent Area Covered')
    plt.xticks(ticks=[1.0, 2.0], labels=['Cooperative', 'Non-cooperative'])
    # plt.ylim(top=100.)
    plt.savefig('PercentAreaCovered.pdf', dpi=150, bbox_inches='tight')

    comm_tps = [r['true_positive_percent'] for r in comm_results]
    nocomm_tps = [r['true_positive_percent'] for r in nocomm_results]
    plt.figure()
    plt.violinplot([comm_tps, nocomm_tps], showmedians=True)
    plt.ylabel('True Positive Percent')
    plt.xticks(ticks=[1.0, 2.0], labels=['Cooperative', 'Non-cooperative'])
    # plt.ylim(top=100.)
    plt.savefig('TruePositives.pdf', dpi=150, bbox_inches='tight')


def plot_violins_with_singles(comm_results, nocomm_results, single_results):
    comm_percents = [r['covered_percent'] for r in comm_results]
    nocomm_percents = [r['covered_percent'] for r in nocomm_results]
    single_percents = [r['covered_percent'] for r in single_results]
    plt.figure()
    plt.violinplot([comm_percents, nocomm_percents, single_percents], showmedians=True)
    plt.ylabel('Percent Area Covered')
    plt.xticks(ticks=[1.0, 2.0, 3.0], labels=['Comm', 'No Comm', 'Single'])
    plt.ylim(top=100.)
    plt.savefig('PercentAreaCovered.pdf', dpi=150, bbox_inches='tight')

    # comm_errs = [r['final_distance_traveled_errs'] for r in comm_results]
    # comm_errs = [item*100 for sublist in comm_errs for item in sublist]
    # nocomm_errs = [r['final_distance_traveled_errs'] for r in nocomm_results]
    # nocomm_errs = [item*100 for sublist in nocomm_errs for item in sublist]
    # single_errs = [r['final_distance_traveled_errs'] for r in single_results]
    # single_errs = [item*100 for sublist in single_errs for item in sublist]
    # plt.figure()
    # plt.violinplot([comm_errs, nocomm_errs, single_errs], showmedians=True)
    # plt.ylabel('Final Error (% of distance traveled)')
    # plt.xticks(ticks=[1.0, 2.0, 3.0], labels=['Comm', 'No Comm', 'Single'])
    # plt.ylim(top=100.)
    # plt.savefig('DistanceTraveledError.pdf', dpi=150, bbox_inches='tight')

    comm_tps = [r['true_positive_percent'] for r in comm_results]
    nocomm_tps = [r['true_positive_percent'] for r in nocomm_results]
    single_tps = [r['true_positive_percent'] for r in single_results]
    plt.figure()
    plt.violinplot([comm_tps, nocomm_tps, single_tps], showmedians=True)
    plt.ylabel('True Positive Percent')
    plt.xticks(ticks=[1.0, 2.0, 3.0], labels=['Comm', 'No Comm', 'Single'])
    plt.ylim(top=100.)
    plt.savefig('TruePositives.pdf', dpi=150, bbox_inches='tight')



def run_with_configs(comm_configs, nocomm_configs, single_configs, yes=None):

    if yes is None:
        answer = input(f"Run {len(comm_configs)} + {len(nocomm_configs)} + {len(single_configs)} sims? [y/N]:")
        if answer != 'y':
            sys.exit()
    print("Running~~~~")

    # def run(config, plot=True, show_plot=False, save_plot=True):
    comm_arg_lists =   [ [config, False, False, False] for config in comm_configs ]
    nocomm_arg_lists = [ [config, False, False, False] for config in nocomm_configs ]
    single_arg_lists = [ [config, False, False, False] for config in single_configs ]

    print("Comm runs")
    with Pool(processes=12) as p:
        comm_results = p.starmap(run, comm_arg_lists)

    print("NO-Comm runs")
    with Pool(processes=12) as p:
        nocomm_results = p.starmap(run, nocomm_arg_lists)

    print("Single runs")
    with Pool(processes=12) as p:
        single_results = p.starmap(run, single_arg_lists)

    return comm_results, nocomm_results, single_results


def run_same_distances(min_seed, max_seed):
    seeds = list(range(min_seed, max_seed))
    comm_configs = [make_config(s,True) for s in seeds]
    nocomm_configs = [make_config(s,False) for s in seeds]
    single_configs = [singlify_config(c) for c in comm_configs]

    comm_results, nocomm_results, single_results = run_with_configs(comm_configs,
                                                                    nocomm_configs,
                                                                    single_configs)


    with open(f'same_dist_comm_{min_seed}_{max_seed}.json', 'w') as f:
        json.dump(comm_results, f)
    with open(f'same_dist_nocomm_{min_seed}_{max_seed}.json', 'w') as f:
        json.dump(nocomm_results, f)
    with open(f'same_dist_single_{min_seed}_{max_seed}.json', 'w') as f:
        json.dump(single_results, f)

    plot_violins(comm_results, nocomm_results, single_results)





def run_multiple_distances(min_seed, max_seed,
                           min_hooks, max_hooks,
                           min_hooklen, max_hooklen, hooklen_step):
    seeds = list(range(min_seed, max_seed))
    hooks = list(range(min_hooks, max_hooks))
    hooklens = list(range(min_hooklen, max_hooklen, hooklen_step))
    run_multiple_distances_listed(seeds, hooks, hooklens)

def run_multiple_distances_listed(seeds, hooks, hooklens, yes=None):
    prod = list(product(seeds, hooks, hooklens))

    comm_configs =   [make_config(s, True,  num_auvs=6, num_hooks=h, hook_len=l) for s,h,l in prod]
    nocomm_configs = [make_config(s, False, num_auvs=6, num_hooks=h, hook_len=l) for s,h,l in prod]
    single_configs = [make_config(s, False, num_auvs=1, num_hooks=h, hook_len=l) for s,h,l in prod]

    comm_results, nocomm_results, single_results = run_with_configs(comm_configs,
                                                                    nocomm_configs,
                                                                    single_configs,
                                                                    yes)


    min_seed = min(seeds)
    max_seed = max(seeds)
    min_hooks = min(hooks)
    max_hooks = max(hooks)
    min_hooklen = min(hooklens)
    max_hooklen = max(hooklens)
    s = f'{min_seed}_{max_seed}_{min_hooks}_{max_hooks}_{min_hooklen}_{max_hooklen}'
    with open(f'comm_{s}.json', 'w') as f:
        json.dump(comm_results, f)
    with open(f'nocomm_{s}.json', 'w') as f:
        json.dump(nocomm_results, f)
    with open(f'single_{s}.json', 'w') as f:
        json.dump(single_results, f)


    try:
        plot_violins(comm_results, nocomm_results, single_results)
    except:
        print("No screen I'm guessing, so no violins for you")
        pass



def mean_std_min_max(a):
    means = []
    stds = []
    mins = []
    maxs = []

    for r in a:
        mean = np.median(r)
        std = np.std(r)
        # minv = np.min(a, axis=1)
        # maxv = np.max(a, axis=1)
        minv = np.quantile(r, 0.10)
        maxv = np.quantile(r, 0.90)

        means.append(mean)
        stds.append(std)
        mins.append(minv)
        maxs.append(maxv)

    return np.array((means, stds, mins, maxs))


def stats_from_json(filenames, num_hooks_filter=None, hook_len_filter=None):
    listed = []
    for filename in filenames:
        print(f"Reading {filename}")
        with open(filename, 'r') as f:
            data = json.load(f)


        for c in data:
            n = c['num_hooks']
            l = c['hook_len']
            cp = c['covered_percent']
            tp = c['true_positive_percent']
            if num_hooks_filter is not None and n == num_hooks_filter:
                listed.append((n, l, cp, tp))
            elif hook_len_filter is not None and l == hook_len_filter:
                listed.append((n, l, cp, tp))
            elif num_hooks_filter is None and hook_len_filter is None:
                listed.append((n, l, cp, tp))
        print("Done")

    assert len(listed)>0, f"No data in json! num_hooks_filter:{num_hooks_filter}, hook_len_filter:{hook_len_filter}"
    a = np.array(listed)

    stats = {'by_num_hooks':{'values':None, 'covered_percent':None, 'true_positive_percent':None},
             'by_hook_len':{'values':None, 'covered_percent':None, 'true_positive_percent':None}}


    by_num_hooks = a[a[:, 0].argsort()]
    unique_hooks, indices = np.unique(by_num_hooks[:, 0], return_index=True)
    stats['by_num_hooks']['values'] = unique_hooks

    covered_percent_by_num_hooks = np.split(a[:,2], indices[1:])
    stats['by_num_hooks']['covered_percent'] = mean_std_min_max(covered_percent_by_num_hooks)

    true_positive_percent_by_num_hooks = np.split(a[:,3], indices[1:])
    stats['by_num_hooks']['true_positive_percent'] = mean_std_min_max(true_positive_percent_by_num_hooks)


    by_hook_len = a[a[:, 1].argsort()]
    unqiue_lens, indices = np.unique(by_hook_len[:, 1], return_index=True)
    stats['by_hook_len']['values'] = unqiue_lens

    covered_percent_by_hook_len = np.split(a[:,2], indices[1:])
    stats['by_hook_len']['covered_percent'] = mean_std_min_max(covered_percent_by_hook_len)

    true_positive_percent_by_hook_len = np.split(a[:,3], indices[1:])
    stats['by_hook_len']['true_positive_percent'] = mean_std_min_max(true_positive_percent_by_hook_len)

    return stats


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_over_marginals(comm_stats, nocomm_stats, num_hooks_filter=None, hook_len_filter=None, smoothing=0.2):
    # get stats from stats_from_json

    # x2 beacuse we wanna talk in LINES not HOOKS, each HOOK is TWO LINES
    hooks_values = 2*np.array(comm_stats['by_num_hooks']['values'])
    lens_values = comm_stats['by_hook_len']['values']

    lines = ['solid', '--']
    colors = ['r', 'b']
    labels = ['Collaborative', 'Non-Collab.']
    statss = [comm_stats, nocomm_stats]

    if num_hooks_filter is not None:
        title = f"Number of lines = {2*num_hooks_filter}"
    elif hook_len_filter is not None:
        title = f"Line distance = {hook_len_filter}m"
    else:
        title = "No filter"

    for yaxis_type, ylabel in zip(['covered_percent', 'true_positive_percent'], ['Coverage[%]', 'True Positive[%]']):
        for xs, xaxis_type, xlabel in zip([hooks_values, lens_values], ['by_num_hooks', 'by_hook_len'], ['Number of lines', 'Line distance[m]']):

            if num_hooks_filter is not None and xaxis_type == 'by_num_hooks':
                continue
            if hook_len_filter is not None and xaxis_type == 'by_hook_len':
                continue

            fig, ax = plt.subplots(1,1)
            fig.set_size_inches(3,3)
            ax.set_xticks(xs)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(40, 95)
            # if hook_len_filter is not None:
                # ax.set_xlim(10, 100)
            # if num_hooks_filter is not None:
                # ax.set_xlim(100, 1000)

            for stats, linestyle, color, label in zip(statss, lines, colors, labels):
                ys = stats[xaxis_type][yaxis_type]
                ax.plot(xs, smooth(ys[0], smoothing), ls=linestyle, c=color, label=label)
                ax.fill_between(xs, smooth(ys[2], smoothing), smooth(ys[3], smoothing), color=color, alpha=0.15)

            ax.legend(loc='lower left')
            ax.set_title(title)

            fig.savefig(f'{yaxis_type}_{xlabel}_{title}.pdf', dpi=150, bbox_inches='tight')



if __name__ == "__main__":
    # config = make_config(seed=88,
                         # comm=False,
                         # summarize_pg=True,
                         # num_auvs=6,
                         # num_hooks=90,
                         # hook_len=100,
                         # overlap_between_lanes=3,
                         # gap_between_rows=-5,
                         # max_ticks = 20000)
    # # config = singlify_config(config)
    # results = run(config, plot=True, show_plot=True, save_plot=False)


# run_same_distances(40,140)
    # run_same_distances(40,42)

    # run_multiple_distances_listed(seeds = list(range(40,90)),
                                  # hooks = [5, 10, 15, 20, 25],
                                  # hooklens = [50, 100, 150, 200, 250])

    # for h in range(200, 401, 10):
        # print(f"\n\n\n\nNUM HOOKS {h}\n\n\n\n")

        # run_multiple_distances_listed(seeds = list(range(91, 201)),
                                      # hooks = [h],
                                      # hooklens = [100],
                                      # yes = 'yes')

    # sys.exit(0)

    try:
        __IPYTHON__
        plt.ion()
    except:
        pass


    # # s = f'{min_seed}_{max_seed}_{min_hooks}_{max_hooks}_{min_hooklen}_{max_hooklen}'
    comm_files = [
        'small_lens_and_hooks/comm_40_89_5_25_50_250.json',
        'seed_40_90_extremes/comm_40_89_50_100_100_100.json',
        'seed_40_90_extremes/comm_40_89_5_5_300_1000.json',
        'seed_40_90_really_extremes/comm_40_89_250_250_100_100.json',
        'seed_40_90_really_extremes/comm_40_89_500_500_100_100.json']

    nocomm_files = [
        'small_lens_and_hooks/nocomm_40_89_5_25_50_250.json',
        'seed_40_90_extremes/nocomm_40_89_50_100_100_100.json',
        'seed_40_90_extremes/nocomm_40_89_5_5_300_1000.json',
        'seed_40_90_really_extremes/nocomm_40_89_250_250_100_100.json',
        'seed_40_90_really_extremes/nocomm_40_89_500_500_100_100.json']


    comm_files.extend([
        "seed_160_180_extremes/comm_160_179_50_100_100_100.json",
        "seed_160_180_extremes/comm_160_179_5_5_300_1000.json"])

    nocomm_files.extend([
        "seed_160_180_extremes/nocomm_160_179_50_100_100_100.json",
        "seed_160_180_extremes/nocomm_160_179_5_5_300_1000.json"])

    # comm_files.extend([
     # "comm_40_89_110_110_100_100.json",
     # "comm_40_89_120_120_100_100.json",
     # "comm_40_89_130_130_100_100.json",
     # "comm_40_89_140_140_100_100.json",
     # "comm_40_89_150_150_100_100.json",
     # "comm_40_89_60_60_100_100.json",
     # "comm_40_89_70_70_100_100.json",
     # "comm_40_89_80_80_100_100.json",
     # "comm_40_89_90_90_100_100.json"])

    # nocomm_files.extend([
     # "nocomm_40_89_110_110_100_100.json",
     # "nocomm_40_89_120_120_100_100.json",
     # "nocomm_40_89_130_130_100_100.json",
     # "nocomm_40_89_140_140_100_100.json",
     # "nocomm_40_89_150_150_100_100.json",
     # "nocomm_40_89_60_60_100_100.json",
     # "nocomm_40_89_70_70_100_100.json",
     # "nocomm_40_89_80_80_100_100.json",
     # "nocomm_40_89_90_90_100_100.json"])


    # comm_files.extend([
        # "comm_91_200_100_100_100_100.json",
        # "comm_91_200_110_110_100_100.json",
        # "comm_91_200_120_120_100_100.json",
        # "comm_91_200_130_130_100_100.json",
        # "comm_91_200_140_140_100_100.json",
        # "comm_91_200_150_150_100_100.json",
        # "comm_91_200_160_160_100_100.json",
        # "comm_91_200_170_170_100_100.json",
        # "comm_91_200_180_180_100_100.json",
        # "comm_91_200_190_190_100_100.json",
        # "comm_91_200_30_30_100_100.json",
        # "comm_91_200_40_40_100_100.json",
        # "comm_91_200_50_50_100_100.json",
        # "comm_91_200_60_60_100_100.json",
        # "comm_91_200_70_70_100_100.json",
        # "comm_91_200_80_80_100_100.json",
        # "comm_91_200_90_90_100_100.json"])

    # nocomm_files.extend([
     # "nocomm_91_200_100_100_100_100.json",
     # "nocomm_91_200_110_110_100_100.json",
     # "nocomm_91_200_120_120_100_100.json",
     # "nocomm_91_200_130_130_100_100.json",
     # "nocomm_91_200_140_140_100_100.json",
     # "nocomm_91_200_150_150_100_100.json",
     # "nocomm_91_200_160_160_100_100.json",
     # "nocomm_91_200_170_170_100_100.json",
     # "nocomm_91_200_180_180_100_100.json",
     # "nocomm_91_200_190_190_100_100.json",
     # "nocomm_91_200_30_30_100_100.json",
     # "nocomm_91_200_40_40_100_100.json",
     # "nocomm_91_200_50_50_100_100.json",
     # "nocomm_91_200_60_60_100_100.json",
     # "nocomm_91_200_70_70_100_100.json",
     # "nocomm_91_200_80_80_100_100.json",
     # "nocomm_91_200_90_90_100_100.json"])

    import glob
    comm_files.extend(glob.glob("moar_data/comm_*.json"))
    comm_files.extend(glob.glob("moar_data/nocomm_*.json"))


    # # num_hooks_filter = 5
    # # hook_len_filter = None
    # # comm_stats = stats_from_json(comm_files, num_hooks_filter, hook_len_filter)
    # # nocomm_stats = stats_from_json(nocomm_files, num_hooks_filter, hook_len_filter)
    # # plot_over_marginals(comm_files, nocomm_files,
                        # # num_hooks_filter=5,
                        # # hook_len_filter=None,
                        # # smoothing=0.1)

    num_hooks_filter = None
    hook_len_filter = 100
    comm_stats = stats_from_json(comm_files, num_hooks_filter, hook_len_filter)
    nocomm_stats = stats_from_json(nocomm_files, num_hooks_filter, hook_len_filter)

    plot_over_marginals(comm_stats, nocomm_stats,
                        num_hooks_filter=num_hooks_filter,
                        hook_len_filter=hook_len_filter,
                        smoothing=0.0)











