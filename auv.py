#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
from toolbox import geometry as geom
from shapely.geometry import Polygon, LineString, MultiLineString


class AUV(object):
    def __init__(self,
                 auv_id,
                 init_pos = None,
                 init_heading = None,
                 target_threshold = 5.,
                 forward_speed = 1.5,
                 auv_length=1.5,
                 max_turn_angle=35.):
        """
        init_pos = [x,y]
        init_heading = degrees from +x towards +y, ccw
        forward_speed = m/s
        auv_length = m
        """

        self.auv_id = auv_id

        if init_pos is None:
            init_pos = np.array([0.,0.])
        self.pos = np.array(init_pos, dtype=float)

        if init_heading is None:
            init_heading = 0.
        self.heading = np.deg2rad(init_heading)

        self.forward_speed = forward_speed
        self.auv_length = auv_length
        self.max_turn_angle = max_turn_angle

        self._pose_trace = []
        # for each pose, are we doing coverage?
        self._coverage_trace = []
        self.covering = False
        self.pose = [self.pos[0], self.pos[1], self.heading]

        self.target_pos = None
        self.target_threshold = target_threshold


    def __str__(self):
        return f'[auv:{self.auv_id}@{self.pose}]'

    def log(self, *args):
        if len(args) == 1:
            args = args[0]
        print(f'[AUV:{self.auv_id}]\t{args}')

    @property
    def distance_traveled(self):
        trace = self.pose_trace
        diffs = trace[:-1, :2] - trace[1:, :2]
        lens = geom.vec_len(diffs)
        dist = sum(lens)
        return dist


    @property
    def reached_target(self):
        if self.target_pos is None or self.pos is None:
            return False

        dist = geom.euclid_distance(self.target_pos, self.pos)
        if dist <= self.target_threshold:
            return True

        return False


    @property
    def pose_trace(self):
        return np.array(self._pose_trace)

    @property
    def apose(self):
        return np.array(self.pose)

    @property
    def heading_vec(self):
        return np.array((np.cos(self.heading),np.sin(self.heading)))

    def _get_turn_direction(self):
        if self.target_pos is None:
            return None, 0

        if self.reached_target:
            return None, 0

        diff = self.target_pos - self.pos
        angle_to_target = geom.vec2_directed_angle(self.heading_vec, diff)
        turn_direction = np.sign(angle_to_target)

        return turn_direction, angle_to_target


    def _move(self,
              dt,
              turn_direction=None,
              turn_amount=None,
              drift_x=0.,
              drift_y=0.,
              drift_heading=0.):
        """
        moves the auv which is modeled as a bicycle
        drifts are meters per second and radians per second
        """
        # if these arent given, run autonomously
        if turn_direction is None or turn_amount is None:
            turn_direction, turn_amount = self._get_turn_direction()

            if turn_direction is None or dt==0.:
                # we dont want to move apparently.
                return None, 0

        # if they ARE given, just execute them

        max_turn = np.deg2rad(self.max_turn_angle)
        steering_angle = turn_direction * min(max_turn, abs(turn_amount))
        dist = self.forward_speed * dt
        hdg = self.heading


        if abs(steering_angle) > 0.001: # is robot turning?
            beta = (dist / self.auv_length) * np.tan(steering_angle)
            r = self.auv_length / np.tan(steering_angle) # radius

            dx = -r*np.sin(hdg) + r*np.sin(hdg + beta)
            dy =  r*np.cos(hdg) - r*np.cos(hdg + beta)
            dh = beta

        else: # moving in straight line
            dx = dist*np.cos(hdg)
            dy = dist*np.sin(hdg)
            dh = 0.

        self.pos += [dx,dy]
        self.pos += [drift_x, drift_y]
        self.set_heading(self.heading + dh + drift_heading)
        return turn_direction, turn_amount


    def set_heading(self, heading):
        heading = heading%(np.pi*2)
        self.heading = heading
        self.pose[2] = self.heading

    def set_heading_angles(self, heading):
        heading = heading%360
        rad = np.deg2rad(self.heading)
        self.set_heading(rad)

    def set_position(self, pos):
        self.pos[0] = pos[0]
        self.pos[1] = pos[1]
        self.pose[0] = pos[0]
        self.pose[1] = pos[1]

    def set_pose(self, pose):
        old_pose = np.array(self.pose)
        self.set_position(pose[:2])
        self.set_heading(pose[2])
        # self.log(f"Set pose {old_pose} -> {np.array(self.pose)}")


    def set_target(self, target_pos, cover=True):
        if target_pos is None:
            self.target_pos = None
            self.covering = False
        else:
            self.target_pos = np.array(target_pos)
            self.covering = cover


    def coverage_polygon(self, swath, pg=None, shapely=False, beam_radius=1):
        # create a vector for each side of the swath
        # stack it up for each pose in the trace
        # then rotate this vector with heading of pose trace
        # and then displace it with pose trace position
        if pg is None:
            t = self.pose_trace
        else:
            t = pg.odom_pose_trace

        disjoints = [[]]
        for i, covering in enumerate(self._coverage_trace):
            if covering:
                disjoints[-1].append(t[i])
            elif len(disjoints[-1]) > 0:
                disjoints.append([])

        polies = []
        for t in disjoints:
            if len(t) < 2:
                continue

            len_trace = len(t)
            t = np.array(t)
            right_swath = np.array([[0,-swath/2]]*len_trace)
            left_swath = np.array([[0,swath/2]]*len_trace)
            left_swath = geom.vec2_rotate(left_swath, t[:,2])
            right_swath = geom.vec2_rotate(right_swath, t[:,2])
            left_swath[:,0] += t[:,0]
            left_swath[:,1] += t[:,1]
            right_swath[:,0] += t[:,0]
            right_swath[:,1] += t[:,1]
            if not shapely:
                poly = np.vstack((right_swath, np.flip(left_swath, axis=0)))
                polies.append(poly)
            else:
                lines = list(zip(right_swath, left_swath))
                mls = MultiLineString(lines)
                poly = mls.buffer(distance = beam_radius,
                                  cap_style = 2)
                polies.append(poly)



        return polies




    def update(self,
               dt,
               turn_direction = None,
               turn_amount = None,
               drift_x = 0.,
               drift_y = 0.,
               drift_heading = 0.,
               cover=None
               ):
        """
        the auv always moves forward at max speed and turns at max speed, so the only
        controlled parameter is the direction of the turn.

        drifts are m/s and radians
        """

        td, tu = self._move(dt,
                            turn_direction,
                            turn_amount,
                            drift_x,
                            drift_y,
                            drift_heading)

        self.pose = [self.pos[0], self.pos[1], self.heading]
        self._pose_trace.append(self.pose)
        if cover is None:
            self._coverage_trace.append(self.covering)
        else:
            self._coverage_trace.append(cover)

        return td, tu



if __name__ == '__main__':
    auv = AUV(auv_id=0,
              init_pos = [0,0],
              init_heading = 30,
              target_threshold = 0.5)

    auv.set_target([30,40])

    import time
    t0 = time.time()
    for i in range(1000):
        auv.update(dt=0.5)
        if auv.reached_target:
            print(auv)
            break
    t1 = time.time()
    print(f"Update rate:{i/(t1-t0)} updates per sec.")

    import matplotlib.pyplot as plt
    from descartes import PolygonPatch
    try:
        __IPYTHON__
        plt.ion()
    except:
        pass

    pt = auv.pose_trace
    fig, ax = plt.subplots(1,1)
    plt.axis('equal')
    ax.plot(pt[:,0], pt[:,1])

    polies = auv.coverage_polygon(swath=5)
    for poly in polies:
        ax.scatter(poly[:,0], poly[:,1], alpha=0.1)
        ax.fill(poly[:,0], poly[:,1], alpha=0.1)

    spolies = auv.coverage_polygon(swath=5, shapely=True)
    for spoly in spolies:
        ax.add_patch(PolygonPatch(spoly, fc='r', ec='r', alpha=0.1))





