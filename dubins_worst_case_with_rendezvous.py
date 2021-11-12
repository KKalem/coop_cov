#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
import dubins

from toolbox import geometry as geom
from mission_plan import MissionPlan
from pose_graph import PoseGraph, PGO_VertexIdStore
from auv import AUV
from mission_plan import construct_lawnmower_paths
from auv_agent import Agent
from main import construct_sim_objects, make_config

class TimedWaypoint(object):
    FIRST = 0
    MIDDLE = 1
    LAST = 2
    pos_to_str = {FIRST:'FIRST',
                  MIDDLE:'MIDDLE',
                  LAST:'LAST'}

    def __init__(self,
                 pose,
                 time,
                 line_idx,
                 position_in_line):
        self.pose = np.array(pose)
        self.time = time
        self.line_idx = line_idx
        assert position_in_line in [TimedWaypoint.FIRST, TimedWaypoint.MIDDLE, TimedWaypoint.LAST]
        self.position_in_line = position_in_line


    def __repr__(self):
        return f"<pose {self.pose}, t={self.time}, line={self.line_idx}, pos_in_line={TimedWaypoint.pos_to_str[self.position_in_line]}>"


    def arrow(self):
        x = self.pose[0]
        y = self.pose[1]
        h = self.pose[2]
        ux = np.cos(h)
        uy = np.sin(h)
        return (x,y,ux,uy)



class TimedPath(object):
    def __init__(self, timed_waypoints):
        """
        Take in a list of TimedWaypoints to construct a path
        """
        self.wps = timed_waypoints

    def extend(self, other_timed_path):
        """
        Extend this path with another TimedPath
        """
        self.wps.extend(other_timed_path.wps)

    def append(self, wp):
        """
        Add a new TimedWaypoint to this path
        """
        self.wps.append(wp)

    def __repr__(self):
        return f"<path begins={self.wps[0]}, ends={self.wps[-1]}>"

    def __len__(self):
        return len(self.wps)

    @property
    def xs(self):
        return [wp.pose[0] for wp in self.wps]

    @property
    def ys(self):
        return [wp.pose[1] for wp in self.wps]

    @property
    def headings(self):
        return [wp.pose[2] for wp in self.wps]

    @property
    def times(self):
        return [wp.time for wp in self.wps]

    @property
    def line_idxs(self):
        return [wp.line_idx for wp in self.wps]

    @property
    def line_wps(self):
        return [wp for wp in self.wps if wp.position_in_line != TimedWaypoint.MIDDLE]

    def plot_quiver(self, ax):
        for wp in self.wps:
            if wp.position_in_line != TimedWaypoint.MIDDLE:
                x,y,ux,uy = wp.arrow()
                ax.quiver(x,y,ux,uy, angles='xy')
                ax.text(x, y+2, f'{str(wp.time)[:5]}')




def poses_to_heading(p0, p1):
    p0 = np.array(p0)
    p1 = np.array(p1)
    heading_vec = p1 - p0
    heading = np.arctan2(heading_vec[1], heading_vec[0])
    return heading

def dubins_path(pose1, pose2, turning_rad):
    """
    pose = (x,y,heading)
    """
    path = dubins.shortest_path((pose1[0], pose1[1], pose1[2]),
                                (pose2[0], pose2[1], pose2[2]),
                                turning_rad)
    return path



def plan_hook(init_wp,
              swath,
              k,
              turning_rad,
              speed,
              ideal_line_len,
              straight_slack,
              path=None):

    if path is None:
        path = TimedPath([])

    s = ideal_line_len / (1-k)
    c0 = swath/2. - s*k
    c = swath - s*k

    assert c>=0, f"c=={c}, Negative coverage!"

    # first wp at bottom left, looking +x
    # plan a dubins path from init_pose to the first pose of the lawnmower
    if init_wp.line_idx == -1:
        wp0_pose = [0., swath/2., 0.]
        init_to_wp0_dubins = dubins_path(init_wp.pose, wp0_pose, turning_rad)
    else:
        wp0_pose = [0., init_wp.pose[1] + swath, 0.]
        # in the case of this being a continuation, add in the worst case as well
        # worst case is that the auv ends up sk away, in the direction of wp0 towards init
        _, direction_vec = geom.vec_normalize(wp0_pose[:2] - init_wp.pose[:2])
        worst_case_drift = -np.array(direction_vec) * (s*k)
        worst_case_pose_at_init = [init_wp.pose[0] + worst_case_drift[0],
                                   init_wp.pose[1] + worst_case_drift[1],
                                   init_wp.pose[2]]
        init_to_wp0_dubins = dubins_path(worst_case_pose_at_init, wp0_pose, turning_rad)

    path_duration = init_to_wp0_dubins.path_length() / speed
    poses, times = init_to_wp0_dubins.sample_many(1)
    for p,t in zip(poses, times):
        wp_dubins = TimedWaypoint(pose = p,
                                  time = init_wp.time + t,
                                  line_idx = init_wp.line_idx+1,
                                  position_in_line = TimedWaypoint.MIDDLE)
        path.append(wp_dubins)


    wp0 = TimedWaypoint(pose = wp0_pose,
                        time = path.wps[-1].time + straight_slack,
                        line_idx = init_wp.line_idx+1,
                        position_in_line=TimedWaypoint.FIRST)
    path.append(wp0)


    wp1_pos = wp0.pose[:2] + [s, -(swath/2. - c0)]
    wp1_pose = [wp1_pos[0], wp1_pos[1], poses_to_heading(wp0.pose[:2], wp1_pos)]
    wp1 = TimedWaypoint(pose = wp1_pose,
                        time = path.wps[-1].time + s/speed + straight_slack,
                        line_idx = init_wp.line_idx+1,
                        position_in_line=TimedWaypoint.LAST)
    path.append(wp1)


    wp2_pos = [ideal_line_len, wp1_pos[1]+swath]
    wp3_pos = [wp2_pos[0]-s,  wp2_pos[1]-(swath - c)]

    # needed the positions first to find the desired heading
    wp2_pose = wp2_pos + [poses_to_heading(wp2_pos, wp3_pos)]
    wp3_pose = wp3_pos + [wp2_pose[2]]

    # now need to do dubins from wp1 to wp2
    # to determine the reach time of wp2
    _, direction_vec = geom.vec_normalize(np.array(wp2_pose[:2]) - wp1_pose[:2])
    worst_case_drift = -np.array(direction_vec) * (s*k)
    worst_case_pose_at_wp1 = [wp1_pose[0] + worst_case_drift[0],
                              wp1_pose[1] + worst_case_drift[1],
                              wp1_pose[2]]
    # worst_case_pose_at_wp1 = [wp1_pose[0] - s*k, wp1_pose[1] - s*k, wp1_pose[2]]
    wp1_to_wp2_dubins = dubins_path(worst_case_pose_at_wp1, wp2_pose, turning_rad)
    path_duration = wp1_to_wp2_dubins.path_length() / speed

    # dubins path between 1 and 2
    poses, times = wp1_to_wp2_dubins.sample_many(1)
    for p, t in zip(poses, times):
        wp_dubins = TimedWaypoint(pose = p,
                                  time = wp1.time + t,
                                  line_idx = wp1.line_idx,
                                  position_in_line = TimedWaypoint.MIDDLE)
        path.append(wp_dubins)

    wp2 = TimedWaypoint(pose = wp2_pose,
                        time = wp1.time + path_duration,
                        line_idx = path.wps[-1].line_idx+1,
                        position_in_line=TimedWaypoint.FIRST)
    path.append(wp2)

    wp3 = TimedWaypoint(pose = wp3_pose,
                        time = wp2.time + s/speed + straight_slack,
                        line_idx = wp2.line_idx,
                        position_in_line=TimedWaypoint.LAST)
    path.append(wp3)

    return path




def plan_dubins_lawnmower(init_pose,
                          init_time,
                          num_lines,
                          ideal_line_len,
                          swath,
                          k,
                          turning_rad,
                          speed,
                          straight_slack = 0.1):

    init_wp = TimedWaypoint(pose = init_pose,
                            time = init_time,
                            line_idx = -1,
                            position_in_line = TimedWaypoint.MIDDLE)

    path = plan_hook(init_wp,
                     swath,
                     k,
                     turning_rad,
                     speed,
                     ideal_line_len,
                     straight_slack,
                     path=None)

    if num_lines <= 2:
        return path

    for i in range(int(np.ceil((num_lines-2)/2))):
        path = plan_hook(path.wps[-1],
                         swath,
                         k,
                         turning_rad,
                         speed,
                         ideal_line_len,
                         straight_slack,
                         path=path)

    return path


if __name__ == "__main__":
    ideal_line_len = 200
    num_lines = 4
    swath = 50
    turning_rad = 5
    speed = 1.5
    k = 0.05

    path = plan_dubins_lawnmower(init_pose = [-10,-10,0],
                                 init_time = 0,
                                 num_lines = num_lines,
                                 ideal_line_len = ideal_line_len,
                                 swath = swath,
                                 k = k,
                                 turning_rad = turning_rad,
                                 speed = speed)

    # just get the WPs that are about the lines, let the vehicle handle the curves itself
    # without explicit wps for them.
    remaining_wps = TimedPath(path.line_wps)

    plt.ion()
    fig, ax = plt.subplots(1,1)
    ax.axis('equal')
    ax.plot(path.xs, path.ys)
    path.plot_quiver(ax)
    ax.plot([0, ideal_line_len, ideal_line_len,     0,                  0],
            [0, 0,              swath*num_lines,    swath*num_lines,    0], c='k')



    # simulate an AUV moving on this path
    config = make_config(seed=42,
                         comm=False,
                         summarize_pg=False,
                         num_auvs=1,
                         num_hooks=3,
                         hook_len=100,
                         overlap_between_lanes=0,
                         gap_between_rows=0,
                         max_ticks = 1000)

    np.random.seed(config.seed)

    auvs, pgs, agents, paths = construct_sim_objects(config)

    auv = auvs[0]
    pg = pgs[0]
    agent = agents[0]

    auv.set_pose([0,0,1.57])
    current_time = 0
    selected_wps = []

    seconds_per_segment = 0.5


    while len(remaining_wps.wps)>0:
        # print(f"Remaining poses:{len(remaining_wps)}, time:{current_time}")
        prev_earlyness = -1
        best_wp = None
        best_path = None
        # first, we need to search for a wp in the segmented_poses that we can reach
        # plan a path to it
        for possible_wp_idx, possible_wp in enumerate(remaining_wps.wps):
            desired_pose = possible_wp.pose
            desired_time = possible_wp.time

            # skip the ones that "middle" points, these are on a straight path
            # that requires no dubins stuff
            if possible_wp.position_in_line == TimedWaypoint.MIDDLE:
                continue

            path = dubins_path(auv.pose, desired_pose, turning_rad)
            path_duration = path.path_length() / speed
            reach_time = current_time + path_duration

            # positive difference means we reach the point earlier than planned for
            # can we reach a point more on time that is earlier in space?
            # negative means we get there too late, dont even consider it
            earlyness = desired_time - reach_time


            # a bit of tolerance for the straight paths
            print(f"Earlyness={earlyness}")
            if earlyness < -2*seconds_per_segment:
                print(f"Skipped!")
                continue

            if earlyness > prev_earlyness:
                if best_wp is None:
                    path_poses, _ = path.sample_many(0.5)
                    path_poses = np.array(path_poses)
                    best_wp = possible_wp
                    best_path = path_poses

                prev_earlyness = earlyness
                break


        if best_path is None or best_wp is None:
            break
        plt.plot(best_path[:,0], best_path[:,1], c='r', alpha=0.8, linestyle='--')

        selected_wps.append(best_wp)


        drift = np.array([0., 0., 0.])
        # only add drift where a rendezvous is going to happen,
        # aka the "last bit" of each lawnmower line
        if best_wp.position_in_line == TimedWaypoint.LAST:
            # and ignore the final line as well
            if possible_wp_idx < len(remaining_wps.wps)-1:
                prev_wp = selected_wps[-2]
                #make sure the prev wp is a 'first'
                if prev_wp.position_in_line == TimedWaypoint.FIRST:
                    dist = geom.euclid_distance(prev_wp.pose[:2], best_wp.pose[:2])
                    drift_mag = k*dist
                    print(f"Drift mag set to {drift_mag}")

                next_pose = remaining_wps.wps[possible_wp_idx+1].pose
                cur_pose = best_wp.pose
                _, direction_vec = geom.vec_normalize(next_pose[:2] - cur_pose[:2])
                drift = np.array(direction_vec) * drift_mag
                drift = -np.array([drift[0], drift[1], 0])

        # move the AUV in the worst-case direction/distance
        worst_case_pose = drift + best_wp.pose

        # hack to "move" the auv.
        auv.set_pose(worst_case_pose)
        current_time = best_wp.time
        remaining_wps.wps = remaining_wps.wps[possible_wp_idx+1:]















