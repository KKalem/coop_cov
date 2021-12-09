#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import numpy as np
import dubins

from toolbox import geometry as geom
import matplotlib.patches as pltpatches


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


    def mirror_around_x(self):
        middle = np.average(self.ys)
        for wp in self.wps:
            x,y,h = wp.pose
            new_y = (-(y - middle))+middle
            h_vec_x = np.cos(h)
            h_vec_y = np.sin(h)
            new_h = np.arctan2(-h_vec_y, h_vec_x)
            wp.pose = np.array([x, new_y, new_h])

    def mirror_around_y(self):
        middle = np.average(self.xs)
        for wp in self.wps:
            x,y,h = wp.pose
            new_x = (-(x - middle))+middle
            h_vec_x = np.cos(h)
            h_vec_y = np.sin(h)
            new_h = np.arctan2(h_vec_y, -h_vec_x)
            wp.pose = np.array([new_x, y, new_h])


    def transpose(self, dx, dy):
        for wp in self.wps:
            wp.pose[0] += dx
            wp.pose[1] += dy



    def plot_quiver(self, ax, circles=False):
        for wp in self.wps:
            if wp.position_in_line != TimedWaypoint.MIDDLE:
                x,y,ux,uy = wp.arrow()
                if wp.idx_in_pattern is not None:
                    ax.text(x, y-2, f'[{wp.idx_in_pattern}]t={int(wp.time)}')
                else:
                    ax.text(x, y-2, f'[t={int(wp.time)}')


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


def plan_simple_lawnmower(num_agents,
                          swath,
                          rect_width,
                          rect_height,
                          speed,
                          straight_slack = 1,
                          gap_between_rows=0,
                          overlap_between_lanes=0,
                          double_sided=False,
                          center_x=False,
                          center_y=False,
                          exiting_line=False):
    # dubins and simple are 90 degree off from each other, im gonna take the direction of
    # the dubins planner to be the 'true' one and rotate the simple plan after its constructed
    # width determines hook_len, height determines num_hooks.
    hook_len = (rect_width - (num_agents-1 * gap_between_rows)) / num_agents
    num_hooks = int(rect_height / ((swath-overlap_between_lanes)*2))

    paths = construct_lawnmower_paths(num_agents=num_agents,
                                      num_hooks=num_hooks,
                                      hook_len=hook_len,
                                      swath=swath,
                                      gap_between_rows=gap_between_rows,
                                      overlap_between_lanes=overlap_between_lanes,
                                      double_sided=double_sided,
                                      center_x=center_x,
                                      center_y=center_y,
                                      exiting_line=exiting_line)

    # this goes from left to right, i want it to go bottom up.
    for i in range(len(paths)):
        pr = geom.vec2_rotate(paths[i], np.pi/2)
        paths[i] = pr

    # and then move the bottom left to be 0,0
    mins = np.min(paths, axis=1)
    minx, miny = np.min(mins, axis=0)
    # but also move the miny half a swath up for coverage
    miny -= swath/2

    for i in range(len(paths)):
        paths[i][:,0] -= minx
        paths[i][:,1] -= miny

    # and now create a timed path out of these points
    timed_paths = []
    for path in paths:
        wp = TimedWaypoint(pose = list(path[0]) + [0.],
                           time = 0,
                           line_idx = 0,
                           position_in_line = TimedWaypoint.FIRST)

        timed_path = TimedPath([wp])
        current_line_idx = 0
        for i in range(1, len(path)):
            prev = timed_path.wps[i-1]

            heading_vec = path[i] - prev.pose[:2]
            heading_angle = np.arctan2(heading_vec[1], heading_vec[0])

            dist = geom.euclid_distance(prev.pose[:2], path[i])

            # at 1,3,5... start a new line
            if i-1%2 == 0:
                current_line_idx += 1

            # even poses are firsts, odds are lasts
            # in this lawnmower, there are no mids
            if i%2 == 0:
                posi = TimedWaypoint.FIRST
            else:
                posi = TimedWaypoint.LAST

            wp = TimedWaypoint(pose = list(path[i]) + [heading_angle],
                               time = prev.time + dist/speed + straight_slack,
                               line_idx = current_line_idx,
                               position_in_line = posi)

            timed_path.append(wp)
        timed_paths.append(timed_path)

    return timed_paths



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



def dubins_path(pose1, pose2, turning_rad):
    """
    pose = (x,y,heading)
    """
    path = dubins.shortest_path((pose1[0], pose1[1], pose1[2]),
                                (pose2[0], pose2[1], pose2[2]),
                                turning_rad)
    return path


def plan_dubins_lawnmower(num_agents,
                          swath,
                          rect_width,
                          rect_height,
                          speed,
                          k,
                          turning_rad,
                          straight_slack = 1,
                          kept_uncertainty_ratio_after_loop = 0.0):

    # split the width into num_agents chunks and plan a path for each
    # every other path should be flipped around the y axis
    single_width = rect_width / num_agents
    # these are all the same, at this point
    timed_paths = [construct_dubins_path(swath,
                                         single_width,
                                         rect_height,
                                         speed,
                                         k,
                                         turning_rad,
                                         straight_slack,
                                         kept_uncertainty_ratio_after_loop) for i in range(num_agents)]

    for i, path in enumerate(timed_paths):
        # flip'em around y
        if i%2 != 0:
            timed_paths[i].mirror_around_y()

        # and transpose
        dx = i*single_width
        path.transpose(dx, 0)


    return timed_paths


def construct_dubins_path(swath,
                          rect_width,
                          rect_height,
                          speed,
                          k,
                          turning_rad,
                          straight_slack = 1,
                          kept_uncertainty_ratio_after_loop = 0.0):

    def poses_to_heading(p0, p1):
        p0 = np.array(p0)
        p1 = np.array(p1)
        heading_vec = p1 - p0
        heading = np.arctan2(heading_vec[1], heading_vec[0])
        return heading

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


def test_dubins_lawnmower_path(num_agents,
                               swath,
                               rect_width,
                               rect_height):

    speed = 1.5
    turning_rad = 5
    k = 0.01
    kural = 0.7

    planned_paths = plan_dubins_lawnmower(num_agents = num_agents,
                                          swath = swath,
                                          k = k,
                                          turning_rad = turning_rad,
                                          speed = speed,
                                          rect_height = rect_height,
                                          rect_width = rect_width,
                                          kept_uncertainty_ratio_after_loop = kural)

    plt.ion()
    fig, ax = plt.subplots(1,1)
    ax.axis('equal')
    ax.plot([0, rect_width, rect_width,     0,                  0],
            [0, 0,          rect_height,    rect_height,        0], c='k')
    for planned_path in planned_paths:
        ax.plot(planned_path.xs, planned_path.ys)
        planned_path.plot_quiver(ax, circles=True)



def test_simple_lawnmower(num_agents,
                          swath,
                          rect_width,
                          rect_height):

    speed = 1.5

    timed_paths = plan_simple_lawnmower(num_agents,
                                        swath,
                                        rect_width,
                                        rect_height,
                                        speed)

    fig, ax = plt.subplots(1,1)
    ax.axis('equal')
    ax.plot([0, rect_width, rect_width,     0,                  0],
            [0, 0,          rect_height,    rect_height,        0], c='k')

    for path in timed_paths:
        ax.plot(path.xs, path.ys)
        path.plot_quiver(ax)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype'] = 42
    from matplotlib.patches import Polygon, Circle
    from main import construct_sim_objects, make_config
    plt.ion()

    rect_width = 1000
    rect_height = 1000
    swath = 50
    num_agents = 4

    test_dubins_lawnmower_path(num_agents, swath, rect_width, rect_height)
    test_simple_lawnmower(num_agents, swath, rect_width, rect_height)

