#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)

import numpy as np
from toolbox import geometry as geom
from auv import AUV
from mission_plan import TimedWaypoint, MissionPlan


class Agent(object):
    def __init__(self,
                 real_auv,
                 pose_graph,
                 mission_plan):

        # a reference to the actual physical auv
        # for ceonvenience
        self._real_auv = real_auv

        self.pg = pose_graph
        self.mission_plan = mission_plan

        # this auv model will be used to create the pose graph from
        # noisy measurements of the real auv
        self.internal_auv = AUV(auv_id = real_auv.auv_id,
                                init_pos = real_auv.pose[:2],
                                init_heading = real_auv.pose[2],
                                target_threshold = real_auv.target_threshold,
                                forward_speed = real_auv.forward_speed,
                                auv_length = real_auv.auv_length,
                                max_turn_angle = real_auv.max_turn_angle)

        self.id = real_auv.auv_id

        # keep track of time passed for waiting purposes
        self.time = 0

        # to keep track of when we were connected to another auv
        # so that we can optimize the PG when we disconnect
        self.connection_trace = []

        # keep a record of how many vertices and edges we received through "fill_in_since_last_interaction"
        # clock, num list
        self.received_data = {'verts':[(0.,0.)],
                              'edges':[(0.,0.)]}


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

        self.time += dt

        current_timed_wp = self.mission_plan.get_current_wp(self.id)

        if current_timed_wp is None or (self.internal_auv.reached_target and self.time >= current_timed_wp.time):
            # we have reached the point, and we dont need to wait here
            # get the next wp
            self.mission_plan.visit_current_wp(self.id)
            current_timed_wp = self.mission_plan.get_current_wp(self.id)

        if current_timed_wp is None:
            # this agent is 'done'
            # do nothing
            return


        # if this is a 'last' waypoint, stop covering
        # if its a 'first', start covering
        if current_timed_wp.position_in_line == TimedWaypoint.FIRST:
            cover = True
        else:
            cover = False

        # TODO do a dubins plan here to determine the position
        # the auv controller should stay as a simple heading controller only
        target_posi = current_timed_wp.pose[:2]

        self.internal_auv.set_target(target_posi, cover=cover)

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
                    comm_dist,
                    summarize_pg=False):

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

                num_vs, num_es = self.pg.fill_in_since_last_interaction(agent.pg, use_summary=summarize_pg)
                self.received_data['verts'].append((self.time, num_vs))
                self.received_data['edges'].append((self.time, num_es))

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
                # success = self.pg.optimize(save_before=False)
                success = self.pg.optimize(use_summary=summarize_pg, save_before=False)
                if success:
                    self.internal_auv.set_pose(self.pg.odom_tip_vertex.pose)



    def distance_traveled_error(self):
        # from the GT auv, find distance traveled
        travel = self._real_auv.distance_traveled
        final_error = geom.euclid_distance(self._real_auv.apose, self.internal_auv.apose)
        error = final_error / travel
        self.log(f"Travel: {travel}, err:{final_error}, percent:{error*100}")
        return error





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype'] = 42
    plt.ion()
    from pose_graph import PoseGraph, PGO_VertexIdStore
    from drift_model import DriftModel

    mplan = MissionPlan(plan_type = MissionPlan.PLAN_TYPE_SIMPLE,
                        num_agents = 5,
                        swath = 50,
                        rect_width = 200,
                        rect_height = 500,
                        speed = 1.5,
                        uncertainty_accumulation_rate_k = 0.01,
                        turning_rad = 5)

    pg_id_store = PGO_VertexIdStore()

    agents = []
    for i, timed_path in enumerate(mplan.timed_paths):
        auv = AUV(auv_id = i,
                  init_pos = timed_path.wps[0].pose[:2],
                  init_heading = timed_path.wps[0].pose[2],
                  target_threshold = 5,
                  forward_speed = mplan.config['speed'])

        pg = PoseGraph(pg_id = i,
                       id_store = pg_id_store)

        agent = Agent(real_auv = auv,
                      pose_graph = pg,
                      mission_plan = mplan)

        agents.append(agent)



    drift_model = DriftModel(num_spirals = 10,
                             num_ripples = 0,
                             area_xsize = mplan.config['rect_width'],
                             area_ysize = mplan.config['rect_height'],
                             scale_size = 1)


    seed = 142
    np.random.seed(seed)

    dt = 0.05
    step = 0
    while True:
        # print(f'Step {step}, time {step*dt}')
        step += 1
        for agent in agents:
            # drift_x, drift_y = drift_model.sample(agent._real_auv.pos[0], agent._real_auv.pos[1])
            drift_x, drift_y = 0,0

            agent.update(dt = dt,
                         drift_x = drift_x*dt*0.01,
                         drift_y = drift_y*dt*0.01)

        if mplan.is_complete:
            break


    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    drift_model.visualize(ax, 10, alpha=0.3)

    for agent in agents:
        real_trace = agent._real_auv.pose_trace
        if len(real_trace) > 0:
            ax.plot(real_trace[:,0], real_trace[:,1], alpha=0.5)
            internal_trace = agent.internal_auv.pose_trace
            ax.plot(internal_trace[:,0], internal_trace[:,1], alpha=0.5, linestyle='--')

    for path in mplan.timed_paths:
        ax.plot(path.xs, path.ys, alpha=0.1, linestyle=':')
        path.plot_quiver(ax)





    ag = agents[0]
    auv = ag._real_auv





