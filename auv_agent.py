#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)

import numpy as np
from toolbox import geometry as geom
from auv import AUV


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



