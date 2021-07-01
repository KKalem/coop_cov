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
                 num_auvs,
                 num_hooks,
                 hook_len,
                 gap_between_rows,
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
        with open(filename) as f:
            json.dump(self.args, f)



def construct_sim_objects(config):
    id_store = PGO_VertexIdStore()

    paths = construct_lawnmower_paths(num_agents = config.num_auvs,
                                      num_hooks = config.num_hooks,
                                      hook_len = config.hook_len,
                                      swath = config.swath,
                                      gap_between_rows = config.gap_between_rows,
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
                                          config.comm_dist)

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
    for path in paths:
        p = np.array(path)
        ax.plot(p[:,0], p[:,1], c='k', alpha=0.2, linestyle=':')

    for c, polies in zip(colors, coverage_polies):
        for poly in polies:
            ax.add_artist(PolygonPatch(poly, alpha=0.1, fc=c, ec=c))

    for c, agent, pg, auv in zip(colors, agents, pgs, auvs):

        ax.plot(auv.pose_trace[:,0], auv.pose_trace[:,1], c=c, alpha=0.5)
        ax.plot(agent.internal_auv.pose_trace[:,0], agent.internal_auv.pose_trace[:,1],
                c=c, alpha=0.5, linestyle='--')


        # for lines instead
        # polies = auv.coverage_polygon(config.swath, shapely=False)
        # for poly in polies:
            # mid = int(len(poly)/2)
            # s1 = poly[:mid]
            # s2 = poly[mid:]
            # for i in range(mid):
                # xs = [s1[i][0], s2[-i-1][0]]
                # ys = [s1[i][1], s2[-i-1][1]]
                # plt.plot(xs, ys, c=c, alpha=0.2)

        t = pg.odom_pose_trace
        ax.scatter(t[:,0], t[:,1], alpha=0.2, marker='.', s=5, c=c)
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


    for agent, d, c in zip(agents, consistent_drifts, colors):
        _, nd = geom.vec_normalize(d)
        nd *= config.swath
        x,y = agent.internal_auv.pose_trace[0][:2]
        x -= (1.5 * config.swath)
        ax.arrow(x, y, nd[0], nd[1],
                  color = c,
                  length_includes_head = True,
                  width = config.swath/20)
        ax.scatter(x- (1.5*config.swath) ,y, alpha=0)


def run(config, plot=True, show_plot=False, save_plot=True):
    np.random.seed(config.seed)

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
    true_positive_percent = 100* ((actually_covered & predicted_coverage).area / predicted_coverage.area)
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
        "travels":travels
    }


    if plot:
        # FIGURES WOOO
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
        fig.set_size_inches(10,5)

        # axs[1].add_artist(PolygonPatch(intended_coverage, alpha=0.5, fc='grey', ec='grey'))
        axs[1].add_artist(PolygonPatch(actually_covered, alpha=0.5, fc='green', ec='green'))
        axs[1].add_artist(PolygonPatch(missed_area, alpha=0.5, fc='red', ec='red'))
        axs[1].add_artist(PolygonPatch(unplanned_area, alpha=0.5, fc='blue', ec='blue'))
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
                num_auvs = 6,
                num_hooks = 5,
                hook_len = 100):
    config = SimConfig(
        num_auvs = num_auvs,
        num_hooks = num_hooks,
        hook_len = hook_len,
        gap_between_rows = 2,
        swath = 50,
        beam_radius = 1,
        target_threshold = 3,
        comm_dist = 20,
        dt = 0.5,
        seed = seed,
        std_shift = 0.4,
        drift_mag = 0.02,
        interact_period_ticks = 5,
        max_ticks = 5000,
        communication = comm
    )
    return config

def singlify_config(config):
    sconfig = SimConfig(
        num_auvs = 1,
        num_hooks = config.num_hooks,
        hook_len = config.num_auvs * config.hook_len,
        gap_between_rows = config.gap_between_rows,
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
        communication = config.communication
    )
    return sconfig

def plot_violins(comm_results, nocomm_results, single_results):
    comm_percents = [r['covered_percent'] for r in comm_results]
    nocomm_percents = [r['covered_percent'] for r in nocomm_results]
    single_percents = [r['covered_percent'] for r in single_results]
    plt.figure()
    plt.violinplot([comm_percents, nocomm_percents, single_percents], showmedians=True)
    plt.ylabel('Percent Area Covered')
    plt.xticks(ticks=[1.0, 2.0, 3.0], labels=['Comm', 'No Comm', 'Single'])
    plt.savefig('PercentAreaCovered.pdf', dpi=150, bbox_inches='tight')

    comm_errs = [r['final_distance_traveled_errs'] for r in comm_results]
    comm_errs = [item*100 for sublist in comm_errs for item in sublist]
    nocomm_errs = [r['final_distance_traveled_errs'] for r in nocomm_results]
    nocomm_errs = [item*100 for sublist in nocomm_errs for item in sublist]
    single_errs = [r['final_distance_traveled_errs'] for r in single_results]
    single_errs = [item*100 for sublist in single_errs for item in sublist]
    plt.figure()
    plt.violinplot([comm_errs, nocomm_errs, single_errs], showmedians=True)
    plt.ylabel('Final Error (% of distance traveled)')
    plt.xticks(ticks=[1.0, 2.0, 3.0], labels=['Comm', 'No Comm', 'Single'])
    plt.savefig('DistanceTraveledError.pdf', dpi=150, bbox_inches='tight')

    comm_tps = [r['true_positive_percent'] for r in comm_results]
    nocomm_tps = [r['true_positive_percent'] for r in nocomm_results]
    single_tps = [r['true_positive_percent'] for r in single_results]
    plt.figure()
    plt.violinplot([comm_tps, nocomm_tps, single_tps], showmedians=True)
    plt.ylabel('True Positive (% of correctly estimated coverage)')
    plt.xticks(ticks=[1.0, 2.0, 3.0], labels=['Comm', 'No Comm', 'Single'])
    plt.savefig('TruePositives.pdf', dpi=150, bbox_inches='tight')


def run_with_configs(comm_configs, nocomm_configs, single_configs):

    answer = input(f"Run {len(comm_configs)} + {len(nocomm_configs)} + {len(single_configs)} sims? [y/N]:")
    if answer != 'y':
        sys.exit()

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


    plot_violins(comm_results, nocomm_results, single_results)


def run_multiple_distances(min_seed, max_seed,
                           min_hooks, max_hooks,
                           min_hooklen, max_hooklen, hooklen_step):
    seeds = list(range(min_seed, max_seed))
    hooks = list(range(min_hooks, max_hooks))
    hooklens = list(range(min_hooklen, max_hooklen, hooklen_step))
    prod = list(product(seeds, hooks, hooklens))

    comm_configs =   [make_config(s, True,  num_auvs=6, num_hooks=h, hook_len=l) for s,h,l in prod]
    nocomm_configs = [make_config(s, False, num_auvs=6, num_hooks=h, hook_len=l) for s,h,l in prod]
    single_configs = [make_config(s, False, num_auvs=1, num_hooks=h, hook_len=l) for s,h,l in prod]

    comm_results, nocomm_results, single_results = run_with_configs(comm_configs,
                                                                    nocomm_configs,
                                                                    single_configs)


    plot_violins(comm_results, nocomm_results, single_results)

    colors = ['r', 'g', 'b']
    shapes = ['x', 'o', '^']
    labels = ['Comm', 'No Comm', 'Single']
    all_results = [comm_results, nocomm_results, single_results]
    plt.figure()
    for results, color, shape, label in zip(all_results, colors, shapes, labels):
        for i,res in enumerate(results):
            travels = np.array(res['travels'])
            tps = np.array(res['true_positive_percents'])
            picked = tps != None
            tps = np.array(tps[picked], dtype=float)
            travels = np.array(travels[picked], dtype=float)
            if i == 0:
                plt.scatter(travels, tps, marker=shape, c=color, alpha=0.3, label=label)
            else:
                plt.scatter(travels, tps, marker=shape, c=color, alpha=0.3)
    plt.xlabel('Travel Distance [m]')
    plt.ylabel('True Positive (% of correctly estimated coverage)')
    plt.ylim(top=100., bottom=0.)
    plt.legend()
    plt.savefig('TravelsTPs.pdf')








if __name__ == "__main__":
    # config = make_config(seed=40, comm=True)
    # run(config, plot=True, show_plot=True, save_plot=False)

    # run_same_distances(40,140)
    run_multiple_distances(40,90,
                           5,10,
                           50,201,20)
    # 60 run example
    # run_multiple_distances(40,42,
                           # 5,10,
                           # 50,201,100)





