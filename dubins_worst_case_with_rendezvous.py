#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
import dubins

import matplotlib.patches as pltpatches

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
                 position_in_line,
                 uncertainty_radius=0,
                 idx_in_pattern=None,
                 uncertainty_radius_before_loop_closure=0):
        self.pose = np.array(pose)
        self.time = time
        self.line_idx = line_idx
        assert position_in_line in [TimedWaypoint.FIRST, TimedWaypoint.MIDDLE, TimedWaypoint.LAST]
        self.position_in_line = position_in_line
        self.uncertainty_radius = uncertainty_radius
        self.uncertainty_radius_before_loop_closure = uncertainty_radius_before_loop_closure
        self.idx_in_pattern = idx_in_pattern


    def __repr__(self):
        return f"<pose {self.pose}, r={self.uncertainty_radius}, t={self.time}, line={self.line_idx}, pos_in_line={TimedWaypoint.pos_to_str[self.position_in_line]}>"


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

    @property
    def total_distance(self):
        xs = self.xs
        ys = self.ys
        s = 0.0;
        for i in range(len(xs) - 1):
            s += geom.euclid_distance([xs[i], ys[i]], [xs[i+1], ys[i+1]])
        return s



    def plot_quiver(self, ax, circles=False):
        for wp in self.wps:
            if wp.position_in_line != TimedWaypoint.MIDDLE:
                x,y,ux,uy = wp.arrow()
                ax.text(x, y-2, f'[{wp.idx_in_pattern}]t={str(wp.time)[:4]}')

                if circles:
                    ax.add_patch(pltpatches.Circle((x,y),
                                                   radius=wp.uncertainty_radius,
                                                   ec='blue',
                                                   fc='blue',
                                                   alpha=0.5))
                    # ax.text(x+5, y-5, f'r={str(wp.uncertainty_radius)[:5]}')
                    if wp.uncertainty_radius_before_loop_closure is not None:
                        ax.add_patch(pltpatches.Circle((x,y),
                                                       radius=wp.uncertainty_radius_before_loop_closure,
                                                       ec='blue',
                                                       fc='blue',
                                                       alpha=0.2))




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


def worst_case_distance_pose(source_wp, away_from_this_pose):
    """
    return the pose that would be furthest away from _away_from_this_pose_,
    that is uncertainty_radius away from source_wp
    source_wp is a TimedWaypoint, away_from_this_pose is a seq of at least 2 length
    """
    prev_wp = source_wp
    wp0_pose = away_from_this_pose
    _, direction_vec = geom.vec_normalize(np.array(prev_wp.pose[:2]) - wp0_pose[:2])
    drift_vec = np.array(direction_vec) * prev_wp.uncertainty_radius
    worst_case_pose_at_init = [prev_wp.pose[0] + drift_vec[0],
                               prev_wp.pose[1] + drift_vec[1],
                               prev_wp.pose[2]]
    return worst_case_pose_at_init


def plan_dubins_lawnmower(swath,
                          k,
                          turning_rad,
                          speed,
                          rect_width,
                          rect_height,
                          straight_slack = 1,
                          kept_uncertainty_ratio_after_loop = 0.0):

    # shorten that name yo
    kural = kept_uncertainty_ratio_after_loop

    def s_next(s):
        s = (1.0 + k) * s / (1.0 - k)
        return  s

    def add_in_dubins_points(prev_wp, to_pose):
        worst_case_pose_at_prev = worst_case_distance_pose(prev_wp, to_pose)
        prev_to_wp2_dubins = dubins_path(worst_case_pose_at_prev, to_pose, turning_rad)

        poses, times = prev_to_wp2_dubins.sample_many(1)
        for p,t in zip(poses, times):
            wp_dubins = TimedWaypoint(pose = p,
                                      time = prev_wp.time + t,
                                      line_idx = prev_wp.line_idx+1,
                                      position_in_line = TimedWaypoint.MIDDLE,
                                      uncertainty_radius= prev_wp.uncertainty_radius)
            path.append(wp_dubins)



    # start from bottom left, looking at wp1
    wp0_posi = [0., swath/2.]
    # in petters code there is a +bk to rect_width here
    s1 = rect_width / (1.0 - k)
    s_old = s1
    wp1_posi = [wp0_posi[0] + s1, wp0_posi[1] - s1*k]

    wp01_heading = poses_to_heading(wp0_posi, wp1_posi)
    wp01_dist = geom.euclid_distance(wp0_posi, wp1_posi)
    wp01_time = wp01_dist / speed + straight_slack

    wp0_pose = wp0_posi + [wp01_heading]
    wp1_pose = wp1_posi + [wp01_heading]

    wp0 = TimedWaypoint(pose = wp0_pose,
                        time = 0,
                        line_idx = 0,
                        position_in_line = TimedWaypoint.FIRST,
                        uncertainty_radius = 0,
                        idx_in_pattern=0)

    wp1 = TimedWaypoint(pose = wp1_pose,
                        time = wp01_time,
                        line_idx = 0,
                        position_in_line = TimedWaypoint.LAST,
                        uncertainty_radius = wp01_dist * k * kural,
                        uncertainty_radius_before_loop_closure = wp01_dist*k,
                        idx_in_pattern=1)

    path = TimedPath([wp0, wp1])

    # because we dont accumulate uncertainty on b
    # for this application
    b = swath

    early_quit = False
    for i in range(23):
        prev_wp = path.wps[-1]

        if len(path.wps) > 5:
            ppprev_wp = path.wps[-2]
        else:
            ppprev_wp = wp0


        # just above "wp1"
        wp2_posi = [prev_wp.pose[0], prev_wp.pose[1] + b]
        # but make it touch the rect as much as you can
        # instead of going straight up
        wp2_posi[0] = rect_width + prev_wp.uncertainty_radius

        s_new = s_next(s_old)
        # again, ignore +b in the paranthesis because no accumulation there
        c_i = swath - k*(s_new + s_old)
        if c_i <= 0:
            early_quit = True
            break

        # above "wp0" and left of wp2
        wp3_posi = [wp2_posi[0] - s_new, ppprev_wp.pose[1] + c_i]

        wp23_heading = poses_to_heading(wp2_posi, wp3_posi)
        wp23_dist = geom.euclid_distance(wp2_posi, wp3_posi)
        wp23_time = wp23_dist / speed + straight_slack

        wp2_pose = wp2_posi + [wp23_heading]
        wp3_pose = wp3_posi + [wp23_heading]

        # handle the dubins curve
        add_in_dubins_points(prev_wp, wp2_pose)

        wp2 = TimedWaypoint(pose = wp2_pose,
                            time = path.wps[-1].time + straight_slack,
                            line_idx = prev_wp.line_idx +1,
                            position_in_line = TimedWaypoint.FIRST,
                            uncertainty_radius = prev_wp.uncertainty_radius,
                            idx_in_pattern=2)

        wp3 = TimedWaypoint(pose = wp3_pose,
                            time = wp2.time + wp23_time,
                            line_idx = wp2.line_idx,
                            position_in_line = TimedWaypoint.LAST,
                            uncertainty_radius = wp2.uncertainty_radius * kural + wp23_dist * k,
                            uncertainty_radius_before_loop_closure = wp2.uncertainty_radius + wp23_dist*k,
                            idx_in_pattern=3)
        wp3.pose[0] = -wp3.uncertainty_radius_before_loop_closure

        path.append(wp2)
        path.append(wp3)

        if rect_height < wp3.pose[1] - swath/2. - wp3.uncertainty_radius_before_loop_closure:
            print('wp3 break')
            break


        # just above wp3
        wp4_posi = [wp3_posi[0], wp3_posi[1] + b]
        wp4_posi[0] = -wp3.uncertainty_radius

        s_old = s_new
        s_new = s_next(s_old)
        c_i = swath - k*(s_new + s_old)

        # above wp2, right of wp4
        wp5_posi = [wp4_posi[0] + s_new, wp2_posi[1] + c_i]

        wp45_heading = poses_to_heading(wp4_posi, wp5_posi)
        wp45_dist = geom.euclid_distance(wp4_posi, wp5_posi)
        wp45_time = wp45_dist / speed + straight_slack

        wp4_pose = wp4_posi + [wp45_heading]
        wp5_pose = wp5_posi + [wp45_heading]

        # handle the dubins curve
        add_in_dubins_points(wp3, wp4_pose)

        wp4 = TimedWaypoint(pose = wp4_pose,
                            time = path.wps[-1].time + straight_slack,
                            line_idx = wp3.line_idx + 1,
                            position_in_line = TimedWaypoint.FIRST,
                            uncertainty_radius = wp3.uncertainty_radius,
                            idx_in_pattern=4)

        wp5 = TimedWaypoint(pose = wp5_pose,
                            time = wp4.time + wp45_time,
                            line_idx = wp4.line_idx,
                            position_in_line = TimedWaypoint.LAST,
                            uncertainty_radius = wp4.uncertainty_radius * kural + wp45_dist * k,
                            uncertainty_radius_before_loop_closure = wp4.uncertainty_radius + wp45_dist * k,
                            idx_in_pattern=5)
        wp5.pose[0] = rect_width + wp5.uncertainty_radius_before_loop_closure

        path.append(wp4)
        path.append(wp5)

        s_old = s_new
        if rect_height < wp5.pose[1] - swath/2. - wp5.uncertainty_radius_before_loop_closure:
            break

    return path




if __name__ == "__main__":
    ideal_line_len = 1000
    rect_height = 1000
    swath = 50
    turning_rad = 5
    speed = 1.5
    k = 0.01
    kural = 0.7

    planned_path = plan_dubins_lawnmower(swath = swath,
                                         k = k,
                                         turning_rad = turning_rad,
                                         speed = speed,
                                         rect_height = rect_height,
                                         rect_width = ideal_line_len,
                                         kept_uncertainty_ratio_after_loop = kural)


    # just get the WPs that are about the lines, let the vehicle handle the curves itself
    # without explicit wps for them.
    remaining_wps = TimedPath(planned_path.line_wps)

    plt.ion()
    fig, ax = plt.subplots(1,1)
    ax.axis('equal')
    ax.plot(planned_path.xs, planned_path.ys)
    planned_path.plot_quiver(ax, circles=True)
    ax.plot([0, ideal_line_len, ideal_line_len,     0,                  0],
            [0, 0,              rect_height,        rect_height,        0], c='k')



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

    auv.set_pose(planned_path.wps[0].pose)
    current_time = -1
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


        if best_path is None or best_wp is None or best_path.shape == (0,):
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
                    drift_mag = prev_wp.uncertainty_radius + k*dist
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















