#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#


import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np

import json
from itertools import product
import sys

from toolbox import geometry as geom
from mission_plan import MissionPlan
from pose_graph import PoseGraph, PGO_VertexIdStore
from auv import AUV
from mission_plan import construct_lawnmower_paths
from auv_agent import Agent


from main import construct_sim_objects, make_config

import dubins

def dubins_path(pose1, pose2, turning_rad):
    """
    pose = (x,y,heading)
    """
    path = dubins.shortest_path((pose1[0], pose1[1], pose1[2]),
                                (pose2[0], pose2[1], pose2[2]),
                                turning_rad)
    return path




if __name__ == "__main__":

    try:
        __IPYTHON__
        plt.ion()
    except:
        pass

    config = make_config(seed=42,
                         comm=False,
                         summarize_pg=False,
                         num_auvs=1,
                         num_hooks=5,
                         hook_len=100,
                         overlap_between_lanes=0,
                         gap_between_rows=0,
                         max_ticks = 1000)

    np.random.seed(config.seed)

    auvs, pgs, agents, paths = construct_sim_objects(config)

    auv = auvs[0]
    pg = pgs[0]
    agent = agents[0]
    path = paths[0]

    fig, ax = plt.subplots(1,1)
    ax.plot(path[:,0], path[:,1], c='b', alpha=0.2, linestyle=':')
    ax.scatter(auv.pos[0], auv.pos[1], c='g', marker='^')
    ax.axis('equal')


    speed = 1.5
    time_to_next_wp = 100
    turning_rad = 5

    for _ in agent.waypoints:
        time_to_next_wp -= 2
        if agent.current_wp_idx > len(agent.waypoints) or agent.current_wp_idx+1 > len(agent.waypoints):
            break

        current_wp = agent.waypoints[agent.current_wp_idx]
        next_wp = agent.waypoints[agent.current_wp_idx+1]

        if agent.current_wp_idx%2 == 0:
            # this is the first point of a line in a lawnmower
            # we want to be oriented towards the next point
            # and at the next point also be oriented in the same direction
            direction_vec = next_wp - current_wp
        else:
            # should be going straight towards this point from the previous
            prev_wp = agent.waypoints[agent.current_wp_idx-1]
            direction_vec = current_wp - prev_wp
            continue #skipping odds

        direction = np.arctan2(direction_vec[1], direction_vec[0])


        # generate some in-between points between current wp and next wp
        # in order to see if we need to "cut" the current wp.
        potential_skipping_wps = [geom.trace_line_segment(current_wp, next_wp, ratio) for ratio in np.linspace(0,1,10)]
        # time it will take to go from current pose, follow the path, then straight distance to next wp
        potential_times = []
        potential_paths = []

        current_pose = agent.internal_auv.pose
        for wp in potential_skipping_wps:
            next_pose = (wp[0], wp[1], direction)
            path = dubins_path(current_pose, next_pose, turning_rad=turning_rad)

            path_dist = path.path_length() + geom.euclid_distance(next_pose[:2], next_wp)
            path_time = path_dist/speed
            potential_times.append(path_time)


            path_poses, _ = path.sample_many(0.5)

            # skip if the path is nothing
            if len(path_poses) < 1:
                continue

            path_poses.append((next_wp[0], next_wp[1], direction))
            path_poses = np.array(path_poses)
            potential_paths.append(path_poses)

            plt.plot(path_poses[:,0], path_poses[:,1], c='b', alpha=0.4, linestyle='-')

        # pick a path that is closest to the _wanted_ time
        # not early, not late
        potential_times = np.array(potential_times)
        potential_times -= time_to_next_wp
        potential_times = np.abs(potential_times)
        best_idx = np.argmin(potential_times)

        picked_path = potential_paths[best_idx]
        plt.plot(picked_path[:,0], picked_path[:,1], c='r', alpha=0.7, linestyle='solid')

        # move the agent to the last point on the path
        new_pose = (picked_path[-1][0], picked_path[-1][1], picked_path[-1][2])
        agent.internal_auv.set_pose(new_pose)
        agent.current_wp_idx += 1
        agent.current_wp_idx += 1 # skipping odds



