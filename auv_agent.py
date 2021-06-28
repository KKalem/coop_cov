#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import json

import numpy as np
from toolbox import geometry as geom

from mission_plan import MissionPlan
from pose_graph import PoseGraph
from auv import AUV
from pose_graph import PGO_VertexIdStore
from mission_plan import construct_lawnmower_paths
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection
from matplotlib.textpath import TextPath
from descartes import PolygonPatch
plt.rcParams['pdf.fonttype'] = 42

from shapely.ops import unary_union
from shapely.geometry import Polygon



class Agent(object):
    def __init__(self,
                 real_auv,
                 pose_graph,
                 waypoints):

        # a reference to the actual physical auv
        # for ceonvenience
        self._real_auv = real_auv

        self.pg = pose_graph

        # this auv model will be used to create the pose graph from
        # noisy measurements of the real auv
        self.internal_auv = AUV(auv_id = real_auv.auv_id,
                                init_pos = real_auv.pose[:2],
                                init_heading = real_auv.pose[2],
                                target_threshold = real_auv.target_threshold,
                                forward_speed = real_auv.forward_speed,
                                auv_length = real_auv.auv_length,
                                max_turn_angle = real_auv.max_turn_angle)


        self.current_wp_idx = 0
        self.waypoints = waypoints

        self.connection_trace = []

    @property
    def waypoints_exhausted(self):
        if self.current_wp_idx >= len(self.waypoints):
            return True
        return False

    def log(self, *args):
        if len(args) == 1:
            args = args[0]
        print(f'[AGNT:{self.pg.pg_id}]\t{args}')


    def update(self,
               dt,
               drift_x = 0.,
               drift_y = 0.,
               drift_heading = 0.):
        # update internal auv
        # apply the same control to real auv, with enviromental noise
        # measure real auv (heading?), apply onto internal auv
        # update pose graph with internal auv


        if self.internal_auv.reached_target and not self.waypoints_exhausted:
            self.current_wp_idx += 1

        if self.waypoints_exhausted:
            return


        current_wp = self.waypoints[self.current_wp_idx]
        cover = False
        # where are we within a hook?
        # each hook is 5 wps.
        if self.current_wp_idx%5 in [1,3]:
            cover=True

        self.internal_auv.set_target(current_wp, cover=cover)

        # control real auv with what the internal one thinks
        td, ta = self.internal_auv.update(dt)
        self._real_auv.update(dt,
                              turn_direction = td,
                              turn_amount = ta,
                              drift_x = drift_x,
                              drift_y = drift_y,
                              drift_heading = drift_heading,
                              cover = cover)

        # compass is good
        self.internal_auv.set_heading(self._real_auv.heading)

        # finally update the pose graph with the internal auv
        self.pg.append_odom_pose(self.internal_auv.apose)



    def communicate(self,
                    all_agents,
                    comm_dist):

        recorded = False

        for agent in all_agents:
            # skip self
            if agent.pg.pg_id == self.pg.pg_id:
                continue

            dist = geom.euclid_distance(self._real_auv.pose[:2], agent._real_auv.pose[:2])
            if dist <= comm_dist:
                self.pg.measure_tip_to_tip(self_real_pose = self._real_auv.pose,
                                           other_real_pose = agent._real_auv.pose,
                                           other_pg = agent.pg)

                self.pg.fill_in_since_last_interaction(agent.pg)

                # was not connected, just connected
                if not recorded:
                    self.connection_trace.append(True)
                    recorded = True

        # is not connected to anyone
        if not recorded:
            self.connection_trace.append(False)


        # if the connection status has changed, optimize the pose graph etc.
        if len(self.connection_trace) > 2:
            if self.connection_trace[-1] != self.connection_trace[-2]:
                success = self.pg.optimize(save_before=False)
                if success:
                    self.internal_auv.set_pose(self.pg.odom_tip_vertex.pose)



    def distance_traveled_error(self):
        # from the GT auv, find distance traveled
        travel = self._real_auv.distance_traveled
        final_error = geom.euclid_distance(self._real_auv.apose, self.internal_auv.apose)
        error = final_error / travel
        self.log(f"Travel: {travel}, err:{final_error}, percent:{error*100}")
        return error



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
    flattened = [item for sublist in coverage_polies for item in sublist]


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
    actually_covered = unary_union(flattened)
    missed_area = intended_coverage - actually_covered
    unplanned_area = actually_covered - intended_coverage
    coverage_within_plan = actually_covered & intended_coverage
    planned_coverage_percent = 100* coverage_within_plan.area / intended_coverage.area
    total_travel = sum([a.distance_traveled for a in auvs])
    print(f"Missed = {missed_area.area} m2")
    print(f"Covered = {planned_coverage_percent}% of the planned area")
    print(f"Total travel = {total_travel} m")

    results = {
        "missed":missed_area.area,
        "covered_percent":planned_coverage_percent,
        "total_travel":total_travel
    }


    if plot:
        # FIGURES WOOO
        fig, axs = plt.subplots(1,2, sharex=False, sharey=True)
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


if __name__ == "__main__":
    from multiprocessing import Pool

    def make_config(seed, comm):
        config = SimConfig(
            num_auvs = 3,
            num_hooks = 5,
            hook_len = 150,
            gap_between_rows = 2,
            swath = 50,
            beam_radius = 1,
            target_threshold = 2,
            comm_dist = 20,
            dt = 0.5,
            seed = seed,
            std_shift = 0.4,
            drift_mag = 0.02,
            interact_period_ticks = 5,
            max_ticks = 15000,
            communication = comm
        )
        return config

    seeds = list(range(40,140))
    comm_configs = [make_config(s,True) for s in seeds]
    nocomm_configs = [make_config(s,False) for s in seeds]
    configs = comm_configs + nocomm_configs

    comm_arg_lists = [ [config, True, False, True] for config in comm_configs ]
    nocomm_arg_lists = [ [config, True, False, True] for config in nocomm_configs ]

    with Pool(processes=12) as p:
        comm_results = p.starmap(run, comm_arg_lists)

    with Pool(processes=12) as p:
        nocomm_results = p.starmap(run, nocomm_arg_lists)

    comm_percents = [r['covered_percent'] for r in comm_results]
    nocomm_percents = [r['covered_percent'] for r in nocomm_results]

    plt.violinplot([comm_percents, nocomm_percents], showmedians=True)
    plt.show()










