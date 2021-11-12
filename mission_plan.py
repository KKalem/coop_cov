#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import os
import pickle

import numpy as np
import itertools

from toolbox import geometry as geom

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
from matplotlib.patches import Polygon, Circle


def construct_lawnmower_paths(num_agents,
                              num_hooks,
                              hook_len,
                              swath,
                              gap_between_rows=0,
                              overlap_between_lanes=0,
                              double_sided = True,
                              center_x = True,
                              center_y = True,
                              exiting_line = True):
    assert num_agents%2==0 or not double_sided, "There must be even number of agents for a double-sided lawnmower plan!"

    def make_hook(flip_y = False, flip_x = False):
        side = 1
        if flip_y:
            side = -1

        direction = 1
        if flip_x:
            direction = -1

        o = overlap_between_lanes /2.
        p1 = [0, side*hook_len]
        p2 = [direction*swath - o, side*hook_len]
        p3 = [direction*swath - o, 0]
        p4 = [direction*2*swath - 2*o, 0]
        return np.array([p1, p2, p3, p4])

    def make_lawnmower_path(starting_pos, flip_y=False, flip_x=False):
        path = [starting_pos]
        for hook_i in range(num_hooks):
            if hook_i>=1:
                starting_pos = path[-1]

            hook = make_hook(flip_y, flip_x)
            hook += starting_pos
            path.extend(hook)

        return np.array(path)

    paths = []
    flip_x = False
    if double_sided:
        num_agents_side = int(num_agents/2)
        num_sides = 2
    else:
        num_agents_side = int(num_agents)
        num_sides = 1

    for side in range(num_sides):
        flip_x = side%2==1
        for agent_i in range(num_agents_side):
            flip_y = True
            if agent_i%2 == 0:
                flip_y = False

            if agent_i >= 1:
                pos  = paths[-1][0].copy()
                if flip_y:
                    pos[1] += 2*hook_len+gap_between_rows
                else:
                    pos[1] += gap_between_rows
            else:
                pos  = np.array([swath, swath])

            # the other agents copy the position of the
            # first one, so we just need to adjust the first one
            if double_sided and agent_i == 0:
                if flip_x:
                    pos[0] -= swath/2
                else:
                    pos[0] += swath/2

            path = make_lawnmower_path(pos , flip_y, flip_x)
            paths.append(path)

    paths = np.array(paths)
    # and then center the paths on x and y
    # makes map painting more efficient
    if center_x or center_y:
        mid_x, mid_y = np.mean(np.vstack(paths), axis=0)
        for path in paths:
            if center_x:
                path[:,0] -= mid_x
            if center_y:
                path[:,1] -= mid_y


    # and then remove the last "exiting" bit of the path
    if not exiting_line:
        paths = np.array( [path[:-1] for path in paths] )
    return paths


class MissionPlan(object):
    def __init__(self, agent_id, waypoints):
        self.wps = np.array(waypoints, dtype=float)
        assert self.wps.shape[0] > 0 and self.wps.shape[1] == 2, f"Waypoints in some weird shape={self.wps.shape}"
        self.agent_id = agent_id
        self.current_wp_index = -1

    def __repr__(self):
        return f"Plan of {self.agent_id} {self.current_wp_index+1}/{len(self.wps)}"

    @property
    def next_wp(self):
        if self.is_complete:
            return None

        if self.current_wp_index == -1:
            self.current_wp_index = 0
        return self.wps[self.current_wp_index]

    @property
    def is_complete(self):
        return self.current_wp_index >= len(self.wps)

    @property
    def started(self):
        return self.current_wp_index > -1

    def visit_current(self):
        self.current_wp_index += 1

    def unvisit_current(self):
        self.current_wp_index -= 1


if __name__=='__main__':
    plt.ion()
    plt.axis('equal')

    try:
        plt.cla()
    except:
        pass

    paths = construct_lawnmower_paths(num_agents = 6,
                                      num_hooks = 10,
                                      hook_len = 200,
                                      swath = 50,
                                      gap_between_rows=5,
                                      double_sided=False)

    for path in paths:
        plt.plot(path[:,0], path[:,1])










