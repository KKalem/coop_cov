#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)

import time

import numpy as np
from toolbox import geometry as geom

from pose_graph import PoseGraphOptimization


class AUVAgent(object):
    def __init__(self):
        self.disable_logs = False

        self.t = 0
        self.agent_id = agent_id



    def log(self, *args):
        if not self.disable_logs:
            print(f'[t:{int(self.t)} a:{self.agent_id}]\t{args}')



    def read_compass(self, heading):
        # update the heading information of the internal auv
        # model from the real auv + some compass noise
        self.auv.set_heading(heading)



    def _add_pgo_vert(self, pose_id, pose, fixed):
        self.pgo.add_vertex(pose_id, pose, fixed)


    def _add_pgo_edge(self, pose_ids, poses, fixeds, information=None):
        if information is None:
            information = [1., 1., 10.]

        for pose_id, pose, fixed in zip(pose_ids, poses, fixeds):
            if pose_id is None:
                self.log(f'A pose id was None when adding pgo edge')
                return
            self._add_pgo_vert(pose_id, pose, fixed)

        prev_pose_id, pose_id = pose_ids
        prev_pose, pose = poses
        edge_id = self.pgo_id_store.get_new_id()
        edge = {'vertices':(prev_pose_id, pose_id),
                'poses':(prev_pose, pose),
                'eid':edge_id,
                'information':information}

        self.pgo.add_edge(**edge)

        return edge


    def add_odom_edge_to_pgo(self):
        if self.landed:
            # you are landed, dont add the same damn points to the graph
            # but also make sure that we have at least one pose in the graph, 
            # in case we start landed
            return

        # the pose graph, we construct at the agent side, since
        # thats what the agent knows about, not the physical reality
        new_pose = make_auv_pose(self.auv)
        new_pose_id = self.pgo_id_store.get_new_id()
        self.prev_own_vert_id = self.last_own_vert_id
        self.last_own_vert_id = new_pose_id

        if len(self.auv.pos_trace) <= 2:
            self.log(f'First odom! Putting edge from 0,0 to here')
            # this is the first odometry we ever got
            # so add an edge from the origin to this point
            # and fix the origin vertex
            prev_pos = np.array([0.,0.])
            prev_heading = 0.
            prev_pose = [prev_pos[0], prev_pos[1], np.deg2rad(prev_heading)]
            prev_pose_id = self.pgo_id_store.origin_id
            prev_fixed = True
        else:
            # -1 is current pose
            prev_pos = self.auv.pos_trace[-2]
            prev_heading = self.auv.heading_trace[-2]
            prev_pose = [prev_pos[0], prev_pos[1], np.deg2rad(prev_heading)]
            prev_pose_id = self.prev_own_vert_id
            prev_fixed = False


        edge = self._add_pgo_edge(pose_ids = (prev_pose_id, new_pose_id),
                                  poses = (prev_pose, new_pose),
                                  fixeds = (prev_fixed, False))

        self.odometry_pgo_vertex_ids.append(new_pose_id)

        if self.collect_odometry_into_gossip:
            gossip = (self.pgo.vertex(self.prev_own_vert_id),
                      self.pgo.vertex(self.last_own_vert_id),
                      edge)
            for i in range(6):
                self.gossip_buffers[i].append(gossip)


    def set_auv_state(self, state):
        # state is a [x,y,h] triple
        # set_state requires a shape like this because of ekf
        x = np.array([[0],[0],[0]], dtype=float)
        x[:,0] = state
        self.auv.set_state(x)




    def optimize_pgo(self, reason='', summarize_manually=False):
        if not self.optimize_pgo_live:
            self.log('Live optim disabled')
            return

        if self.landed and self.optimized_when_landed:
            # just dont, you aint going anywhere
            self.log('Not optimizing, landed')
            return


        self.log(f'Optimizing with {len(self.pgo.vertices())} vertices', reason)
        t0 = time.time()
        success = self.pgo.optimize(max_iterations=200)

        if self.summarize_pgo or summarize_manually:
            self.pgo = new_summary_graph(self.pgo, self.pgo_id_store, keep_vertex = self.pgo.vertex_ids[-1])
        t1 = time.time()
        self._optim_timings.append(t1-t0)

        if not success:
            self.log(f'...skipped due to OPTIM FAILURE')
            return

        self.log(f'...done')
        if self.landed:
            self.optimized_when_landed = True
            self.log(f'Optimized when landed~')

        if self.last_own_vert_id is None:
            self.log(f'Last own vert id was None! Could not set new estimate after optim.')
            return

        # always use our own vertices to set the auv state
        # otherwise we might set our own state to someone else's state
        best_estimate_vert = self.pgo.vertex(self.last_own_vert_id)

        # this happens when we are landed and summarized our own 
        # vertex away because we optimized with poses from someone else
        if best_estimate_vert is None:
            self.log(f'Own last pose was summarized')
            return

        best_estimate = best_estimate_vert.estimate().to_vector()
        self.set_auv_state(best_estimate)




    def add_landmark_edge_to_pgo(self, real_auv):
        if self.landed and len(self.pgo.edge_ids) > 0:
            # you are landed, dont add the same damn points to the graph
            return

        real_auv_pose = make_auv_pose(real_auv)
        can_see_landmark = False
        for landmark_id, landmark in self.landmarks:
            dist = geom.euclid_distance(landmark, real_auv_pose[:2])
            if dist < self.landmark_detection_range:
                can_see_landmark = True
                lm_pose = [landmark[0], landmark[1], 0.]
                self._add_pgo_edge(pose_ids = (self.last_own_vert_id, landmark_id),
                                   poses = (real_auv_pose, lm_pose),
                                   fixeds = (False, False),
                                   information = 1.)


        if len(self.pgo.vertices()) > 3:
            if not can_see_landmark:
                # we cant see any landmarks right now, did we _just leave_
                # the range of one?
                if self.was_in_landmark_range:
                    # we _were_, but not anymore
                    self.was_in_landmark_range = False
                    # this is the time to update our pose estimate
                    # by optimizing the pose graph
                    self.optimize_pgo('left landmark')
            else:
                # we can see some landmarks, did we _just enter_
                # the range of one?
                if not self.was_in_landmark_range:
                    # we _just entered_ range, optimize again!
                    self.optimize_pgo('entered landmark')
                    self.was_in_landmark_range = True



    def measure_other_agent(self, other_agent, other_auv, real_auv):
        # the current comms gives us a measurement
        # of where the other guy is _right now_
        # so we can add a brand new edge for that measurement
        other_auv_pose = make_auv_pose(other_auv)
        real_auv_pose = make_auv_pose(real_auv)
        edge = self._add_pgo_edge(poses = (real_auv_pose, other_auv_pose),
                                  pose_ids = (self.last_own_vert_id, other_agent.last_own_vert_id),
                                  fixeds = (False, False))
        self.measurement_pgo_edges.append(edge)



        # merge the pose graphs as well
        if self.enable_merging_pgos:
            if self.optimize_pgo_live:
                t0 = time.time()
                self.pgo.merge_with(other_agent.pgo)
                t1 = time.time()
                # there used to be a full_pgo here, keeping the tuple to not break stuff
                t2 = time.time()
                self._merge_timings.append( (t1-t0, t2-t1) )

        return edge




    def interact_with_other_agents(self, all_agents, all_auvs, real_auv):
        self.can_see_other = False
        self.can_see_other_stationary_agent = False
        for other_agent, other_auv in zip(all_agents, all_auvs):
            # skip self
            if other_auv.auv_id == self.auv.auv_id or self.agent_id == other_agent.agent_id:
                continue

            # LONG RANGE STUFF

            # long-range when landed?
            if self.enable_longrange_when_landed:
                # we are both landed, and i havent seen the other dude's current vert
                if other_agent.state == 'landed' and self.state == 'landed':
                    if  other_agent.last_own_vert_id is not None:
                        if self.pgo.vertex(other_agent.last_own_vert_id) is None:
                            # we can transfer if we're both landed
                            # but thats it, this should not affect our coven
                            # just some extra data for loop closures
                            edge = self.measure_other_agent(other_agent,
                                                            other_auv,
                                                            real_auv)

                            if self.enable_gossip or self.collect_odometry_into_gossip:
                                # also record this into gossip buffer
                                gossip = (self.pgo.vertex(self.last_own_vert_id),
                                          other_agent.pgo.vertex(other_agent.last_own_vert_id),
                                          edge)
                                for i in range(6):
                                    self.gossip_buffers[i].append(gossip)


            # SHORT RANGE STUFF

            # close enough?
            dist = geom.euclid_distance(other_auv.pos, real_auv.pos)
            if dist > self.comms_range:
                continue

            # only measure our paired agent if we have one
            if self.paired_agent is None or other_agent.agent_id == self.paired_agent.agent_id:
                # other_agent can transmit to agent
                self.measure_other_agent(other_agent,
                                         other_auv,
                                         real_auv)
                self.can_see_other = True

                # maybe WE are the stationary agent and the other dude immediately started
                # petalling after seeing us before we could see it be in 'waiting other' state
                # if thats the case, we _did_ see it too.
                # there is a _very_ small chance this can cause 'chains' but meh
                if other_agent.state in ['waiting_other', 'landed'] or\
                   other_agent.can_see_other_stationary_agent:
                    self.can_see_other_stationary_agent = True



            # if enabled, measure everyone else too
            if self.enable_measure_all_agents and self.paired_agent is not None:
                # dont double-measure the paired dood
                if other_agent.agent_id != self.paired_agent.agent_id:
                    self.measure_other_agent(other_agent,
                                             other_auv,
                                             real_auv)



            if self.enable_gossip or self.collect_odometry_into_gossip:
                # listen to the gossip too
                gossip_buffer = other_agent.gossip_buffers[self.agent_id]
                other_agent.gossip_buffers[self.agent_id] = []
                for v1,v2,pgo_edge in gossip_buffer:
                    for v in [v1,v2]:
                        if self.pgo.vertex(v.id()) is None:
                            self.pgo.add_vertexse2(v)
                    self.pgo.add_edge(**pgo_edge)


            # we must be moving to a new flower, in which case
            # we want to collect the measurement to landed agents
            # we see on the way
            if self.collect_odometry_into_gossip:
                if other_agent.state in ['waiting_other', 'landed']:
                    edge = self.measure_other_agent(other_agent,
                                                    other_auv,
                                                    real_auv)
                    gossip = (self.pgo.vertex(self.last_own_vert_id),
                              other_agent.pgo.vertex(other_agent.last_own_vert_id),
                              edge)
                    for i in range(6):
                        self.gossip_buffers[i].append(gossip)






        if not self.can_see_other:
            # we cant see any other agents,
            # but did we just lose connection?
            if self.was_in_agent_range:
                self.optimize_pgo('Left agent range')
                self.was_in_agent_range = False

        else:
            # we can see, did we just make the connection?
            if not self.was_in_agent_range:
                self.optimize_pgo('Entered agent range')
                self.was_in_agent_range = True



    def unland(self):
        self.log('Unlanded')
        self.state = 'init'
        self.landed = False
        self.optimized_when_landed = False
        self.got_gps_when_landed = False


    def land(self):
        self.log(f'Landed, was in agent range:{self.can_see_other_stationary_agent}')
        self.state = 'landed'
        self.landed = True
        self.landing_points.add(tuple(self.auv.pos.copy()))


    def visit_current_wp(self):
        new_state = self.current_action.visited_current_wp()
        if new_state == 'landed':
            self.land()
        else:
            self.state = new_state



    def take_gps(self, real_auv):
          # check if it is the other dude that is lost before we take a gps.
          # we do this by checking if the previous gps point
          # is already close to the current target point
          target_pos = self.current_action.get_wp()
          if target_pos is not None and len(self.gps_points) > 0:
              diff = target_pos - self.gps_points[-1]
              is_close, dist = point_is_close(diff, self.target_threshold)
              if is_close:
                  self.log(f'Its the other dude that is lost, aint taking GPS')
                  self.waiting_clock = 0
                  self.got_gps_when_landed = True
                  return False


          # we waited, and still couldnt connect with anyone else
          # we can now assume we are lost and take a gps
          # then start moving towards the target with this new information
          self.log(f'Waited {self.waiting_clock}s before GPSing')
          self.set_auv_state(make_auv_pose(real_auv))
          # also, we dont take gps all the time, only once per 'landing'
          self.got_gps_when_landed = True
          self.waiting_clock = 0

          # finally, since this is a gps position, we can modify the last
          # vertex we had in the graph to be this correct gps pose
          self.pgo.update_vertex(self.last_own_vert_id, make_auv_pose(self.auv), fixed=True)
          # and optimize the whole thing with this gps
          self.optimize_pgo('Got GPS')

          # we want to start moving after the gps is acquired
          # because we probably need to correct our landing position
          # with this new info
          self.state = 'moving'
          # re-do the last wp
          self.current_action.unvisit_current_wp()

          # keep track for plotting later
          self.gps_points.append(self.auv.pos.copy())
          return True




    def update(self, dt, env_map, real_auv, all_agents, all_auvs):
        """
        update the inner model of the auv and return the control signal
        that was applied to it. This signal should be passed on to a 'real'
        outside auv that might have noises in places that the inner does not
        """
        self.t += dt

        assert self.state in AUVAgent.states, f"UNKNOWN STATE: {self.state}"


        if self.state == 'init':
            # first ever time we are running the update func
            self.log(f'Initting action')
            try:
                self.current_action = self.planned_actions[self.next_action_index]
                self.next_action_index += 1
                self.state = 'moving'
                self.log(f'Got new action {self.current_action}')
            except IndexError:
                self.log(f'No actions found during init, landing')
                self.land()


        if self.state == 'landed':
            # do nothing, we are landed.
            # we just wait for a signal to unland
            return self._move(dt, None, env_map)


        if self.state == 'waiting_other':
            # if we can see another agent, we land
            if self.can_see_other_stationary_agent:
                self.log(f'Waited {self.waiting_clock}s before continuing')
                self.tried_full_pgo = False
                self.waiting_clock = 0
                # this changes state
                self.visit_current_wp()
                return self._move(dt, None, env_map)
            # if we cant see another agent within a time limit, we take a gps
            # and start moving again
            if self.waiting_clock > self.max_wait_time:
                if self.enable_gps_when_lost and not self.got_gps_when_landed:
                    self.log("Taking GPS")
                    self.take_gps(real_auv)
                    return self._move(dt, None, env_map)
                else:
                   self.log(f'Waited {self.waiting_clock}s and tried everything before getting lost')
                   self.state = 'lost'


            # we wait...
            self.waiting_clock += dt
            # either way, we dont move in this state
            return self._move(dt, None, env_map)


        if self.state == 'moving':
            # move towards the waypoint of our action
            target_pos = self.current_action.get_wp()
            if target_pos is None:
                self.log('Action has no wp, initing')
                self.state = 'init'
                return self._move(dt, None, env_map)

            # have we reached the target?
            diff = target_pos - self.auv.pos
            is_close, dist = point_is_close(diff, self.target_threshold)
            if is_close:
                real_diff = target_pos - real_auv.pos
                _, real_dist = point_is_close(real_diff, self.target_threshold)
                self.log(f'Reached wp:{self.current_action.current_wp_index}/{len(self.current_action.waypoints)} real_dist:{real_dist}')
                # reached target, tell action
                # we will pick our next target in the next update
                self.visit_current_wp()
                return self._move(dt, None, env_map)


            # reset, since we moved around
            self.got_gps_when_landed = False
            turn = self.get_turn_direction(target_pos)
            return self._move(dt, turn, env_map)





def euclid_from_diff(diff):
    return np.sqrt(diff[0]**2+diff[1]**2)

def point_is_close(diff, target_threshold):
    # can we avoid a sqrt?
    if abs(diff[0]) < target_threshold and abs(diff[1]) < target_threshold:
        return True, -1

    # we couldnt :/
    dist = euclid_from_diff(diff)
    if dist < target_threshold:
        return True, dist

    return False, dist

def make_auv_pose(auv):
    return [auv.pos[0],
            auv.pos[1],
            np.deg2rad(auv.heading)]


class AUVAction(object):
    def __init__(self):
        self.waypoints = []
        self.post_wp_states = []
        self.current_wp_index = 0

    def add_wp(self, wp, post_wp_state):
        assert post_wp_state in AUVAgent.states, "Unknown state!"
        self.waypoints.append(wp)
        self.post_wp_states.append(post_wp_state)


    def visited_current_wp(self):
        post_wp_state = self.post_wp_states[self.current_wp_index]
        self.current_wp_index += 1
        return post_wp_state


    def unvisit_current_wp(self):
        self.current_wp_index -= 1


    def get_wp(self):
        if self.is_done():
            return None

        return self.waypoints[self.current_wp_index]

    def is_done(self):
        if self.current_wp_index >= len(self.waypoints):
            return True

        return False

    def __repr__(self):
        return f'Act:{self.current_wp_index}/{len(self.waypoints)} wps. Last state:{self.post_wp_states[-1]}'






