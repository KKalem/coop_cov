#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)

import numpy as np
from toolbox import geometry as geom

from mission_plan import MissionPlan
from pose_graph import PoseGraph



class AUVAgent(object):
    def __init__(self,
                 auv,
                 swath,
                 comm_range,
                 pgo_id_store):
        self.disable_logs = False

        self.t = 0
        self.agent_id = auv.auv_id

        self.auv = auv
        self.swath = swath
        self.comm_range = comm_range
        self.pose_graph = PoseGraph(pgo_id = auv.auv_id,
                                    pgo_id_store = pgo_id_store)
        self.mission_plan = None

        # initialize the graph immediately
        self.pose_graph.append_pose(self.auv.pose)

        # remember when we last interacted with some other agent
        self.last_interaction_t = -1


    @property
    def pose_trace(self):
        return self.auv.pose_trace

    @property
    def coverage_polygon(self):
        return self.auv.coverage_polygon(swath=self.swath)

    def log(self, *args):
        if not self.disable_logs:
            if len(args) == 1:
                args = args[0]

            print(f"[AGNT:{self.agent_id}, {self.t}]\t{args}")


    def set_plan(self, path):
        self.mission_plan = MissionPlan(agent_id = self.agent_id,
                                        waypoints = path)
        self.log(f"Set={self.mission_plan}")


    def communicate_and_measure(self,
                                other_agent,
                                real_auv,
                                other_real_auv):
        # is this guy us?
        if other_agent.agent_id == self.agent_id:
            return

        # is this guy close enough?
        dist = geom.euclid_distance(real_auv.pos, other_real_auv.pos)
        if dist > self.comm_range:
            return

        # synch pose graphs
        self.pose_graph.communicate_with_other(other_agent.pose_graph)
        self.pose_graph.measure_other_agent(real_auv.pose,
                                            other_real_auv.pose,
                                            other_agent.pose_graph)

        self.last_interaction_t = self.t




    def update(self, dt):
        # got a plan
        if self.mission_plan is not None:
            # first update call ever
            if not self.mission_plan.started:
                self.auv.set_target(self.mission_plan.next_wp)
            # not the first call, check if the auv reached the wp
            if self.auv.reached_target:
                # reached target, change to next target
                self.mission_plan.visit_current()
                # check if mission is complete
                if not self.mission_plan.is_complete:
                    self.auv.set_target(self.mission_plan.next_wp)

        #TODO
        # we just lost connection with someone
        if self.t - 1 == self.last_interaction_t:
            self.log(f"Lost connection")
            self.pose_graph.optimize()


        # whatever we do, auv "runs"
        self.t += 1
        turn_direction, turn_amount = self.auv.update(dt)
        self.pose_graph.append_pose(self.auv.pose)

        return turn_direction, turn_amount



if __name__ == "__main__":
    from pose_graph import PGO_VertexIdStore
    from auv import AUV
    from mission_plan import construct_lawnmower_paths
    from tqdm import tqdm

    num_auvs = 2
    num_hooks = 3
    hook_len = 100
    gap_between_rows = 5
    swath = 50
    comm_range = 10
    dt = 0.5
    seed = 42
    std_shift = 0.3
    consistent_x_drift = 0
    consistent_y_drift = 0

    interact_period_ticks = 20

    np.random.seed(seed)

    paths = construct_lawnmower_paths(num_agents = num_auvs,
                                      num_hooks=num_hooks,
                                      hook_len=hook_len,
                                      swath=swath,
                                      gap_between_rows=gap_between_rows,
                                      double_sided=False)

    pgo_id_store = PGO_VertexIdStore()

    auvs = [AUV(auv_id = i,
                init_pos = paths[i][0],
                init_heading = 0) for i in range(len(paths))]

    agents = [AUVAgent(auv = auv,
                       swath = swath,
                       comm_range = comm_range,
                       pgo_id_store = pgo_id_store) for auv in auvs]

    [agent.set_plan(path) for agent,path in zip(agents,paths)]

    real_auvs = [AUV(auv_id = i,
                     init_pos = paths[i][0],
                     init_heading = 0) for i in range(len(paths))]


    done = False
    t = 0
    with tqdm(total=paths.shape[0]*paths.shape[1]) as pbar:
        while not done:
            t += 1
            pbar.desc = f'Tick:{t}'
            for agent, real_auv in zip(agents, real_auvs):
                # allow agents to interact once in a while
                if t%interact_period_ticks:
                    for other_agent, other_real_auv in zip(agents, real_auvs):
                        agent.communicate_and_measure(other_agent,
                                                      real_auv,
                                                      other_real_auv)

                # update the agent and get the control
                turn_direction, turn_amount = agent.update(dt)


                # apply the control and disturbance to the the real auv
                drift_x = np.random.normal(0, std_shift) + consistent_x_drift
                drift_y = np.random.normal(0, std_shift) + consistent_y_drift
                real_auv.update(dt,
                                turn_direction,
                                turn_amount,
                                drift_x = drift_x,
                                drift_y = drift_y,
                                drift_heading = 0)

                if agent.auv.reached_target:
                    pbar.update(1)

                if agent.mission_plan.is_complete:
                    done = True

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import LineCollection

    try:
        __IPYTHON__
        plt.ion()
    except:
        pass

    colors = ['red', 'green', 'purple', 'orange', 'blue', 'cyan']
    fig, ax = plt.subplots(1,1)
    plt.axis('equal')
    for color, agent, real_auv in zip(colors, agents, real_auvs):
        ax.plot(real_auv.pose_trace[:,0], real_auv.pose_trace[:,1], c=color, alpha=0.7)
        ax.plot(agent.pose_trace[:,0], agent.pose_trace[:,1], c=color, linestyle='--', alpha=0.7)
        ax.add_artist(Polygon(xy=real_auv.coverage_polygon(swath), closed=True, alpha=0.1, color=color))

        lc = LineCollection(segments = agent.pose_graph.all_edges, colors=color, alpha=0.1)
        ax.add_collection(lc)

















