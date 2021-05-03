#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import numpy as np
from toolbox import geometry as geom


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
        self.pose = [self.pos[0], self.pos[1], self.heading]

        self.reached_target = False
        self.target_pos = None
        self.target_threshold = target_threshold


    def __str__(self):
        return f'[auv:{self.auv_id}@{self.pose}]'

    def log(self, *args):
        print(f'[AUV:{self.auv_id}]\t{args}')

    @property
    def pose_trace(self):
        return np.array(self._pose_trace)

    @property
    def heading_vec(self):
        return np.array((np.cos(self.heading),np.sin(self.heading)))

    def _get_turn_direction(self):
        if self.target_pos is None:
            return None, 0

        diff = self.target_pos - self.pos
        if abs(diff[0]) <= self.target_threshold and abs(diff[1]) <= self.target_threshold:
            self.reached_target = True

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
            if self.reached_target:
                turn_direction = None
            else:
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
        self.set_position(pose[:2])
        self.set_heading(pose[2])


    def set_target(self, target_pos):
        if target_pos is None:
            self.reached_target = True
        else:
            self.reached_target = False
            self.target_pos = np.array(target_pos)


    def coverage_polygon(self, swath):
        # create a vector for each side of the swath
        # stack it up for each pose in the trace
        # then rotate this vector with heading of pose trace
        # and then displace it with pose trace position
        t = self.pose_trace
        len_trace = len(t)
        right_swath = np.array([[0,-swath/2]]*len_trace)
        left_swath = np.array([[0,swath/2]]*len_trace)
        left_swath = geom.vec2_rotate(left_swath, t[:,2])
        right_swath = geom.vec2_rotate(right_swath, t[:,2])
        left_swath[:,0] += t[:,0]
        left_swath[:,1] += t[:,1]
        right_swath[:,0] += t[:,0]
        right_swath[:,1] += t[:,1]
        poly = np.vstack((right_swath, np.flip(left_swath, axis=0)))
        return poly


    def update(self,
               dt,
               turn_direction = None,
               turn_amount = None,
               drift_x = 0.,
               drift_y = 0.,
               drift_heading = 0.
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
    try:
        __IPYTHON__
        plt.ion()
    except:
        pass

    pt = auv.pose_trace
    fig, ax = plt.subplots(1,1)
    plt.axis('equal')
    ax.plot(pt[:,0], pt[:,1])

    from matplotlib.patches import Polygon

    poly = auv.coverage_polygon(swath=5)
    p = Polygon(xy=poly, closed=True, alpha=0.1)
    ax.scatter(poly[:,0], poly[:,1], alpha=0.1)
    ax.add_artist(p)





