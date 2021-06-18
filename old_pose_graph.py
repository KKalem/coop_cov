#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 ozer <ozer@ozer-kthdesktop>
#
# Distributed under terms of the MIT license.

class OldPoseGraph(object):
    def __init__(self,
                 pgo_id,
                 pgo_id_store):

        self.pgo_id = pgo_id
        self.pgo_id_store = pgo_id_store


        # keep track of which poses are _ours_
        self.own_pose_ids = []
        # keep track of which poses were ever fixed
        self.fixed_pose_ids = []
        # independent of the PGOs, keep a trace of poses after corrections
        self.corrected_pose_trace = {}
        # and before corrections too
        self.raw_pose_trace = {}
        # both indexed by pose vertex ids

        self.last_pose = None
        self.last_pose_id = None

        self.min_poses_between_optims = 20
        self.poses_since_last_optimization = 0

        # which pose did we last hear about from which other pg?
        self.other_last_pose_ids = {}

        # keep a pose_id -> [edges] mapping so that we can add those edges
        # when we hear about this pose from another PG
        self.pose_edges = {}

        self.num_pgos_optimized = 0

        # a list of poses (not pids) that we received from other graphs
        #XXX mostly for debugging?
        self.edges_from_others = []



    def log(self, *args):
        if len(args) == 1:
            args = args[0]
        print(f'[PG:{self.pgo_id}]\t{args}')


    def append_pose(self,
                    pose):
        """
        pose is [x,y,heading(rad)]
        """
        pose_id = self.pgo_id_store.get_new_id()

        if type(pose) != np.ndarray:
            pose = np.array(pose)

        if self.last_pose is None:
            # this is the first pose in the graphs
            # so it is not connected to any previous pose --> BS.
            # this might not be the _actual_ first pose but a middle-pose that 
            # an optimization happen. so it MIGHT have a parent!!!
            # fix it 
            self.pgo.add_vertex(pose_id, pose, fixed=True)
            self.own_pose_ids.append(pose_id)
            self.log(f"First pose in the graph = {pose_id}@{pose}")
            self.fixed_pose_ids.append(pose_id)
        else:
            # there is a previous pose we should make an edge with
            # check if the new pose is too close to the previous one
            # this happens if the vehicle is not moving.
            diff = np.abs(pose - self.last_pose)
            if diff[0] < 0.1 and diff[1] < 0.1:
                return

            self.pgo.add_vertex(pose_id, pose, fixed=False)
            self.own_pose_ids.append(pose_id)

            information = [1., 1., 1000.]

            self.add_new_edge_between_poses(self.last_pose, self.last_pose_id,
                                            pose, pose_id,
                                            information)


        self.last_pose = pose
        self.last_pose_id = pose_id
        self.poses_since_last_optimization += 1
        self.raw_pose_trace[pose_id] = pose


    @property
    def pose_trace(self):
        poses = []
        for d in [self.corrected_pose_trace, self.raw_pose_trace]:
            vert_ids = list(d.keys())
            vert_ids.sort()
            for vid in vert_ids:
                poses.append(d[vid])
        return np.array(poses)

    @property
    def fixed_poses(self):
        poses = []
        for pid in self.fixed_pose_ids:
            p = self.pose_from_pid(pid)
            if p is not None:
                poses.append(p)

        return poses


    @property
    def all_edges(self):
        all_edges = []
        for pid, edges in self.pose_edges.items():
            for edge in edges:
                pid1, pid2 = edge['vertices']
                p1 = self.pose_from_pid(pid1)
                p2 = self.pose_from_pid(pid2)
                if p1 is not None and p2 is not None:
                    all_edges.append((p1[:2],p2[:2]))
        return all_edges


    @property
    def last_corrected_pose(self):
        maxkey = max(list(self.corrected_pose_trace.keys()))
        return self.corrected_pose_trace.get(maxkey)



    def pose_from_pid(self, pid):
        p = self.corrected_pose_trace.get(pid)
        if p is not None:
            return p

        p = self.raw_pose_trace.get(pid)
        if p is not None:
            return p

        p = self.pgo.get_pose_array(pid)
        return p


    def edges_from_pid(self, pid):
        e = self.pose_edges.get(pid)
        return e



    def optimize(self):
        if self.poses_since_last_optimization <= self.min_poses_between_optims:
            return False

        self.pgo.save(f"{self.pgo_id}_{self.num_pgos_optimized}.g2o")

        success = self.pgo.optimize(max_iterations=200)
        if not success:
            self.log(f'Optimization failed!')
            return False

        # optimized. copy the corrected poses into our own stash
        # and re-create a fresh pgo
        self.corrected_pose_trace.update(self.pgo.get_all_poses_dict(id_list=self.own_pose_ids))
        self.raw_pose_trace = {}
        self.pgo = PoseGraphOptimization(pgo_id = self.pgo_id)
        self.poses_since_last_optimization = 0
        self.num_pgos_optimized += 1
        return True


    def provide_pose_to_other(self, pid):
        # return a pose from our own current raw poses only
        # also return if this pose is fixed
        return self.pgo.get_pose_array(pid), self.pgo.pose_is_fixed(pid)



    def communicate_with_other(self,
                               other_pg):

        # last known pose id of the other pg
        # we either know it, or we default to 0
        other_last_id = self.other_last_pose_ids.get(other_pg.pgo_id, 0)

        # start going DOWN from their current id
        # until either we reach the beginning of their raw
        # poses, or we reach the last known pose id from before
        # effectively fill in the missing info between now and past
        # yes there are 'empty' slots here, i dont care
        # with some extras too, just to _make sure_ that we get the latest of all
        if other_pg.last_pose_id is None:
            self.log(f"{other_pg.pgo_id}'s lsat_pose_id is None, can't collect poses from it!")
            return


        begin = other_last_id-20
        end = other_pg.last_pose_id+10
        # filter out the pids that we know we have in our own pgo
        # pid_list = list(filter(lambda pid: self.pgo.vertex(pid) is None, reversed(range(begin, end))))
        pid_list = list(reversed(range(begin, end)))
        added_pids = []
        for pid in pid_list:
            new_pose, fixed = other_pg.provide_pose_to_other(pid)
            if new_pose is not None:
                self.pgo.add_vertex(pid, new_pose, fixed)
                added_pids.append(pid)

        # add the edges that belong to the same poses we just added
        # for pid in added_pids:
        for pid in added_pids:
            new_edges = other_pg.edges_from_pid(pid)
            for new_edge in new_edges:
                if new_edge is not None:
                    self.add_edge_between_poses(new_edge)
                    self.edges_from_others.append(new_edge['poses'])

        # remember the last vertex we touched this pg
        self.other_last_pose_ids[other_pg.pgo_id] = other_pg.last_pose_id


    def measure_other_agent(self,
                            self_pose,
                            other_pose,
                            other_pg):

        if other_pg.last_pose_id is None:
            self.log(f"{other_pg.pgo_id}'s last_pose_id is None, cant measure it!")

        success = self.add_new_edge_between_poses(self_pose = self_pose,
                                                  self_pose_id = self.last_pose_id,
                                                  other_pose = other_pose,
                                                  other_pose_id = other_pg.last_pose_id,
                                                  information = [100., 100., 100.])
        if not success:
            self.log(f"Could not measure {other_pg.pgo_id}!")


    def add_new_edge_between_poses(self,
                                   self_pose, self_pose_id,
                                   other_pose, other_pose_id,
                                   information):
        """
        add and edge to our pgo between our pose and the pose of the
        other pose graph
        """
        assert type(information) in [list, np.ndarray] and len(information) == 3, "Information length is not 3!"

        edge_id = self.pgo_id_store.get_new_id()

        edge = {'vertices':(self_pose_id, other_pose_id),
                'poses':(self_pose, other_pose),
                'eid':edge_id,
                'information':information}

        success = self.add_edge_between_poses(edge)
        if not success:
            self.log(f"Edge={edge} couldnt be added between poses")

        return success


    def add_edge_between_poses(self, edge):
        pid1, pid2 = edge['vertices']

        for pid in (pid1, pid2):
            if pid is None or self.pgo.vertex(pid) is None:
                # self.log("{pid} was {self.pgo.vertex(pid)} for edge={edge}")
                return False

            if self.pose_edges.get(pid) is None:
                self.pose_edges[pid] = []


        self.pose_edges[pid1].append(edge)

        self.pgo.add_edge(**edge)
        return True

