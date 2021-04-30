#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import numpy as np
import g2o
from toolbox import geometry as geom

def make_transform(xytheta):
    x,y,theta = xytheta
    l1 = [np.cos(theta), -np.sin(theta), x]
    l2 = [np.sin(theta), np.cos(theta), y]
    l3 = [0., 0., 1.]
    mat = np.array([l1,l2,l3], dtype=float)
    return mat








class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self,
                 pgo_id,
                 min_num_of_verts_to_optimize=10):

        super().__init__()
        solver = g2o.BlockSolverSE2(g2o.LinearSolverCholmodSE2())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        self.pgo_id = pgo_id

        # if true, at every optimization this pose graph
        # will remove all odom vertices until the previous
        # optimized end point
        # then it will add a large edge between the two optimized
        # heads with high info matrix
        self.min_num_of_verts_to_optimize = min_num_of_verts_to_optimize

        # keep track of the verts and edges that we merged last
        # so that we dont go over thousands of them every time
        self.other_pgo_last_merged_vertex_id = {}
        self.other_pgo_last_merged_edge_id = {}

        # we need this because edges dont have a dictionary
        # like vertices do by default
        self.edge_id_to_edge = {}

        # keep these in a separate list so that we dont have to sort them
        # since .vertices() and .edges() are a dict and a set
        self.vertex_ids = []
        self.edge_ids = []


    def log(self, *args):
        if len(args) == 1:
            args = args[0]
        print(f'[PGO:{self.pgo_id}]\t{args}')


    def optimize(self,
                 max_iterations=20):
        if len(self.vertices()) < self.min_num_of_verts_to_optimize:
            return False

        super().initialize_optimization()

        super().optimize(max_iterations)

        return True




    def get_last_merged_ids(self, other_pgo_id):
        vid = self.other_pgo_last_merged_edge_id.get(other_pgo_id)
        if vid is None:
            self.other_pgo_last_merged_vertex_id[other_pgo_id] = -1
            vid = -1

        eid = self.other_pgo_last_merged_edge_id.get(other_pgo_id)
        if eid is None:
            self.other_pgo_last_merged_edge_id[other_pgo_id] = -1
            eid = -1

        return vid, eid


    def merge_with(self, other_pgo):
        # add all the vertices and edges from the given other_pgo
        # into this one by copying them in

        # what are the vert and edge that we last merged on?
        # we need not go lower than these when merging
        # assumption here is that _every_ new thing added will have ids
        # larger than anything before, no id reuse!
        other_pgo_id = other_pgo.pgo_id
        last_v_id, last_e_id = self.get_last_merged_ids(other_pgo_id)

        other_vert_ids = other_pgo.vertex_ids
        if len(other_vert_ids) < 1:
            self.log(f'Other pgo {other_pgo_id} was empty, nothing to merge')
            return


        # reversed, because we want to look at the _last_ vertex first
        # so that we can stop when we see a specific one
        for v_id in reversed(other_vert_ids):
            if v_id <= last_v_id:
                # we have all the other verts already, skip the rest
                break

            vert = self.vertex(v_id)
            if vert is not None:
                # we already know this vert, skip it
                continue

            new_vert = other_pgo.vertex(v_id)
            self.other_pgo_last_merged_vertex_id[other_pgo_id] = v_id
            self.add_vertexse2(new_vert)


        # we now got the verts added to ourself.
        # now we need the edges between them from the other pgo
        other_edge_ids = other_pgo.edge_ids
        self_edges = self.edge_ids
        i=0
        for e_id in reversed(other_edge_ids):
            e = other_pgo.edge_id_to_edge.get(e_id)
            if e is None:
                continue

            v1, v2 = e.vertices()
            if v1 is None or v2 is None:
                # this edge's verts were summarized away
                continue

            eid = e.id()
            if eid <= last_e_id:
                break

            if eid in self_edges:
                # we already have this edge
                continue

            self.add_edgese2(vse2s = [v1, v2],
                             eid = e.id(),
                             information = e.information())
            self.other_pgo_last_merged_edge_id[other_pgo_id] = eid
            i += 1




    def add_vertex(self, v_id, pose, fixed=False):
        if self.vertex(v_id) is not None:
            return False

        v_se3 = g2o.VertexSE2()
        v_se3.set_id(v_id)
        v_se3.set_estimate(g2o.SE2(pose))
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
        self.vertex_ids.append(v_id)
        return True


    def update_vertex(self, v_id, pose, fixed=False):
        v = self.vertex(v_id)
        v.set_estimate(g2o.SE2(pose))
        v.set_fixed(fixed)


    def add_vertexse2(self, vse2):
        # must make a duplicate of vse2, because
        # the same _object_ vert can not be part of
        # multiple graphs, since they can each modify
        # it in-place.
        v_se2 = g2o.VertexSE2()
        v_se2.set_id(vse2.id())
        v_se2.set_estimate(g2o.SE2(vse2.estimate().to_vector()))
        v_se2.set_fixed(vse2.fixed())
        super().add_vertex(v_se2)
        self.vertex_ids.append(vse2.id())


    def add_edgese2(self,
                    vse2s,
                    eid,
                    information=1.,
                    robust_kernel=None):
        vertices = []
        poses = []
        for v in vse2s:
            vertices.append(v.id())
            poses.append(v.estimate().to_vector())

        self.add_edge(vertices,
                      poses,
                      eid,
                      information,
                      robust_kernel)


    def add_edge(self,
                 vertices,
                 poses,
                 eid,
                 information=1.,
                 robust_kernel=None):

        assert all([x is not None for x in vertices]), f"A vertex is None! : {vertices}"
        assert all([x is not None for x in poses]), f"A pose is None! : {poses}"

        if self.edge_id_to_edge.get(eid) is not None:
            return

        # measurement is done from p1's point of ref.
        # so we find the transform from p1 to p2
        p1, p2 = np.array(poses, dtype=float)
        m = np.zeros_like(p1)

        wTa = make_transform(p1)
        wTb = make_transform(p2)
        invwta = np.linalg.inv(wTa)
        D = np.dot(invwta, wTb)

        # numerical shenanigans
        if abs(D[0,0]) > 1:
            D[0,0] = 1*np.sign(D[0,0])

        m[0], m[1], m[2] = D[0,2], D[1,2], np.arccos(D[0,0])
        # arccos doesnt preserve the sign of the angle
        m[2] *= -np.sign(np.arcsin(D[0,1]))


        if type(information) == type(1.) or type(information) == type(1):
            information = np.identity(3) * information
        else:
            information = np.array(information)
            if information.shape == (3,):
                information = np.diag(information)


        edge = g2o.EdgeSE2()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(g2o.SE2(m))  # relative pose
        edge.set_information(information)
        edge.set_id(eid)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

        # record this ourselves because super() doesnt
        self.edge_id_to_edge[eid] = edge
        self.edge_ids.append(eid)

    def get_pose(self, id):
        v = self.vertex(id)
        if v is None:
            return None

        return self.vertex(id).estimate()

    def get_pose_array(self, id):
        p = self.get_pose(id)
        if p is None:
            return None

        return self.get_pose(id).vector()

    def pose_is_fixed(self, id):
        v = self.vertex(id)
        if v is None:
            return None

        return v.fixed()

    def get_all_poses(self, id_list=None):
        # id_list can be a filter to only get verts in that list
        poses = []
        vert_ids = list(self.vertices().keys())
        vert_ids.sort()
        for vertex_id in vert_ids:
            if id_list is not None and vertex_id in id_list:
                poses.append(self.get_pose_array(vertex_id))

            if id_list is None:
                poses.append(self.get_pose_array(vertex_id))

        return np.array(poses)

    def get_all_poses_dict(self, id_list=None):
        poses = {}
        vert_ids = list(self.vertices().keys())
        for vertex_id in vert_ids:
            if id_list is not None and vertex_id in id_list:
                poses[vertex_id] = self.get_pose_array(vertex_id)

            if id_list is None:
                poses[vertex_id] = self.get_pose_array(vertex_id)

        return poses





class PGO_VertexIdStore(object):
    def __init__(self):
        self.ids = []
        self.origin_id = 0

    def get_new_id(self):
        if len(self.ids) > 0:
            new_id = self.ids[-1]+1
        else:
            new_id = 1
        self.ids.append(new_id)
        return new_id



def new_summary_graph(pgo, pgo_id_store, keep_vertex=None):
    """
    returns a new pgo instance that has either 3 or 2 vertices in it.
    the first and last vertices of the original pgo and the kept vertex if given
    """

    new_pgo = PoseGraphOptimization(pgo_id = pgo.pgo_id)

    origin_vert = pgo.vertex(pgo.vertex_ids[0])
    tip_vert = pgo.vertex(pgo.vertex_ids[-1])

    new_pgo.add_vertexse2(origin_vert)

    # origin -> mid_vert
    if keep_vertex is not None and keep_vertex not in [origin_vert.id(), tip_vert.id()]:
        mid_vert = pgo.vertex(keep_vertex)
        new_pgo.add_vertexse2(mid_vert)
        new_pgo.add_edgese2(vse2s = [origin_vert, mid_vert],
                            eid = pgo_id_store.get_new_id(),
                            information = 10.)
    else:
        mid_vert = origin_vert


    new_pgo.add_vertexse2(tip_vert)

    # mid_vert/origin -> tip_vert
    new_pgo.add_edgese2(vse2s = [mid_vert, tip_vert],
                        eid = pgo_id_store.get_new_id(),
                        information = 10.)

    return new_pgo





class PoseGraph(object):
    def __init__(self,
                 pgo_id,
                 pgo_id_store):
        """
        A collection of PGOs
        """

        self.pgo_id = pgo_id
        self.pgo_id_store = pgo_id_store

        self.pgo = PoseGraphOptimization(pgo_id = self.pgo_id)

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



    def log(self, *args):
        if len(args) == 1:
            args = args[0]
        print(f'[pgo_id:{self.pgo_id}]\t{args}')


    def append_pose(self,
                    pose):
        """
        pose is [x,y,heading(rad)]
        """
        pose_id = self.pgo_id_store.get_new_id()

        if type(pose) != np.ndarray:
            pose = np.array(pose)

        if self.last_pose is None or self.poses_since_last_optimization == 0:
            # this is the first pose in the graphs
            # so it is not connected to any previous pose
            # fix this one
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

            information = [1., 1., 10.]

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
        return poses

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
                all_edges.append((p1,p2))
        return all_edges


    def pose_from_pid(self, pid):
        p = self.corrected_pose_trace.get(pid)
        if p is not None:
            return p

        p = self.raw_pose_trace.get(pid)
        if p is not None:
            return p

        p = self.pgo.get_pose_array(pid) 
        return p


    def optimize(self):
        if self.poses_since_last_optimization <= self.min_poses_between_optims:
            self.log(f'Not enough poses yet:{self.poses_since_last_optimization}/{self.min_poses_between_optims}')
            return False

        success = self.pgo.optimize(max_iterations=200)
        if not success:
            self.log(f'Optimization failed!')
            return False

        # optimized. copy the corrected poses into our own stash
        # and re-create a fresh pgo
        self.corrected_pose_trace.update(self.pgo.get_all_poses_dict(id_list=self.own_pose_ids))
        self.raw_pose_trace = {}
        #TODO uncomment once debugged
        # self.pgo = PoseGraphOptimization(pgo_id = self.pgo_id)
        self.poses_since_last_optimization = 0


    def provide_pose_to_other(self, pid):
        # return a pose from our own current raw poses only
        # also return if this pose is fixed
        return self.pgo.get_pose_array(pid), self.pgo.pose_is_fixed(pid)


    def search_edges_from_pose_id(self, pid, target_pid=None, num_edges=None):
        """
        provide multiple edges starting from the given pose id (pid)
        either stop at the number of edges or once the target_pid is found
        BFS
        """

        assert target_pid is not None or num_edges is not None, "You must give at least one of target_pid or num_edges!"

        # if the searches doesnt know _from where_ to start,
        # start from the root of our poses
        if pid == -1:
            pid = self.own_pose_ids[0]

        if pid == target_pid:
            return [], True

        all_edges = []
        open_pids = set()
        open_pids.add(pid)
        closed_pids = set()
        while True:
            if num_edges is not None:
                if len(all_edges) >= num_edges:
                    return all_edges, True

            if target_pid is not None:
                if target_pid in closed_pids:
                    return all_edges, True

            try:
                pid = open_pids.pop()
            except KeyError:
                # open_pids empty, cant follow anymore, return
                return all_edges, False

            if pid in closed_pids:
                continue

            closed_pids.add(pid)
            edges = self.pose_edges.get(pid)

            if edges is None or len(edges) == 0:
                continue

            # edge = {'vertices':(self_pose_id, other_pose_id),
                    # 'poses':(self_pose, other_pose),
                    # 'eid':edge_id,
                    # 'information':information}
            for edge in edges:
                pid1, pid2 = edge['vertices']
                all_edges.append(edge)
                open_pids.add(pid1)
                open_pids.add(pid2)


        return all_edges


    def communicate_with_other(self,
                               other_pg):
        # get the vertices
        self._collect_poses_from_other(other_pg)
        # get the edges
        self._add_past_edges_of_other(other_pg)
        # remember the last vertex we touched this pg
        self.other_last_pose_ids[other_pg.pgo_id] = other_pg.last_pose_id


    def _collect_poses_from_other(self, other_pg):
        # last known pose id of the other pg
        # we either know it, or we default to 0
        other_last_id = self.other_last_pose_ids.get(other_pg.pgo_id, 0)


        # start going DOWN from their current id
        # until either we reach the beginning of their raw
        # poses, or we reach the last known pose id from before
        # effectively fill in the missing info between now and past
        # yes there are 'empty' slots here, i dont care
        # with some extras too, just to _make sure_ that we get the latest of all
        pid_list = list(reversed(range(other_last_id-10, other_pg.last_pose_id+10)))
        added_pids = []
        for pid in pid_list:
            new_pose, fixed = other_pg.provide_pose_to_other(pid)
            if new_pose is not None:
                self.pgo.add_vertex(pid, new_pose, fixed)
                added_pids.append(pid)



    def _add_past_edges_of_other(self, other_pg):
        """
        add edges from the other_pg.
        We want the edges that the other pg added between its odom measurements
        """
        # find a pose_id in the past of the other pg
        # could be the first time ever we see this other dude, if thats the case
        # ask it to fill us in on _all_ the edges (-1)
        # otherwise just ask since the last time we saw it
        past_pose_id = self.other_last_pose_ids.get(other_pg.pgo_id, -1)
        if past_pose_id == other_pg.last_pose_id:
            return

        edges, success = other_pg.search_edges_from_pose_id(pid = past_pose_id,
                                                            target_pid = other_pg.last_pose_id)

        for edge in edges:
            self.add_edge_between_poses(edge)



    def measure_other_agent(self,
                            self_pose,
                            other_pose,
                            other_pg):

        success = self.add_new_edge_between_poses(self_pose = self_pose,
                                                  self_pose_id = self.last_pose_id,
                                                  other_pose = other_pose,
                                                  other_pose_id = other_pg.last_pose_id,
                                                  information = [10., 10., 10.])
        if not success:
            self.log("Could not measure {other_pg.pgo_id}!")


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

        return success


    def add_edge_between_poses(self, edge):
        pid1, pid2 = edge['vertices']

        for pid in (pid1, pid2):
            if self.pose_edges.get(pid) is None:
                self.pose_edges[pid] = []

            if self.pgo.vertex(pid) is None:
                return False

        self.pose_edges[pid1].append(edge)

        self.pgo.add_edge(**edge)
        return True











if __name__=='__main__':
    import matplotlib.pyplot as plt
    try:
        __IPYTHON__
        plt.ion()
    except:
        pass


    dt = 0.5
    id_store = PGO_VertexIdStore()

    target_threshold = 1

    left = -20
    right = 20
    mid = 5
    mid_left = -45
    up = 20

    from auv import AUV
    auv = AUV(auv_id=0,
              init_pos = [left,0],
              init_heading = 0,
              target_threshold=target_threshold)


    pg = PoseGraph(pgo_id = 0,
                   pgo_id_store = id_store)


    auv1 = AUV(auv_id=1,
               init_pos = [right,0],
               init_heading = 0,
               target_threshold=target_threshold)


    pg1 = PoseGraph(pgo_id = 1,
                    pgo_id_store = id_store)

    auv2 = AUV(auv_id=2,
               init_pos = [left-5,0],
               init_heading = 0,
               target_threshold=target_threshold)


    pg2 = PoseGraph(pgo_id = 2,
                    pgo_id_store = id_store)


    def interact():
        dist = geom.euclid_distance(auv.pose[:2], auv1.pose[:2])
        if dist < 20:
            pg.communicate_with_other(pg1)
            pg.measure_other_agent(auv.pose,
                                   auv1.pose,
                                   pg1)

        dist = geom.euclid_distance(auv.pose[:2], auv2.pose[:2])
        if dist < 20:
            pg.communicate_with_other(pg2)
            pg.measure_other_agent(auv.pose,
                                   auv2.pose,
                                   pg2)

    drift = 0
    def run(ticks):
        global drift
        for i in range(ticks):
            auv.update(dt=0.5)
            auv1.update(dt=0.5)
            auv2.update(dt=0.5)

            p = np.array(auv.pose) + [drift, 0, 0]
            drift += 0.05
            pg.append_pose(p)
            pg1.append_pose(auv1.pose)
            pg2.append_pose(auv2.pose)

            if auv.reached_target:
                break

            interact()

    cup = 0
    def hook():
        global cup
        auv.set_target([-mid,cup])
        auv1.set_target([mid,cup])
        auv2.set_target([mid_left, cup])
        run(100)

        cup += up

        auv.set_target([-mid,cup])
        auv1.set_target([mid,cup])
        auv2.set_target([mid_left,cup])
        run(100)

        auv.set_target([left,cup])
        auv1.set_target([right,cup])
        auv2.set_target([left-10,cup])
        run(100)

        cup += up

        auv.set_target([left,cup])
        auv1.set_target([right,cup])
        auv2.set_target([left-10,cup])
        run(100)


    for i in range(4):
        hook()


    pg.pgo.save('test.g2o')
    pg.optimize()


    auv_trace = np.array(auv.pose_trace)
    pg_trace = np.array(pg.pose_trace)
    plt.plot(auv_trace[:,0], auv_trace[:,1], c='g', alpha=0.2)
    plt.scatter(pg_trace[:,0], pg_trace[:,1], c='g', alpha=0.2, marker='+')


    for p1,p2 in pg.all_edges:
        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), c='g', alpha=0.6)

    pg_trace = np.array(pg1.pose_trace)
    plt.plot(pg_trace[:,0], pg_trace[:,1], c='b', alpha=0.2)
    pg_trace = np.array(pg2.pose_trace)
    plt.plot(pg_trace[:,0], pg_trace[:,1], c='r', alpha=0.2)


    plt.axis('equal')



