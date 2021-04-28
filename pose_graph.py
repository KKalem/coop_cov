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
        if self.vertex(v_id) is None:
            v_se3 = g2o.VertexSE2()
            v_se3.set_id(v_id)
            v_se3.set_estimate(g2o.SE2(pose))
            v_se3.set_fixed(fixed)
            super().add_vertex(v_se3)
            self.vertex_ids.append(v_id)


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
            self.log(f'Edge {eid} alredy in graph!')
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
        return self.vertex(id).estimate()

    def get_pose_array(self, id):
        return self.get_pose(id).vector()

    def get_all_poses(self, id_list=None):
        # id_list can be a filter to only get verts in that list
        poses = []
        vert_ids = list(self.vertices().keys())
        vert_ids.sort()
        for vertex_id in vert_ids:
            if id_list is not None and vertex_id in id_list:
                poses.append(self.get_pose_array(vertex_id))
        return np.array(poses)




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

        # independent of the PGOs, keep a trace of poses after corrections
        self.corrected_pose_trace = []
        # and before corrections too
        self.raw_pose_trace = []

        self.prev_pose = None
        self.prev_pose_id = None

        self.min_poses_between_optims = 20
        self.poses_since_last_optimization = 0


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

        if self.prev_pose is None:
            # this is the first pose in the graphs
            # so it is not connected to any previous pose
            # fix this one
            self.pgo.add_vertex(pose_id, pose, fixed=True)
        else:
            # check if the new pose is too close to the previous one
            # this happens if the vehicle is not moving.
            diff = np.abs(pose - self.prev_pose)
            if diff[0] < 0.1 and diff[1] < 0.1:
                return

            self.pgo.add_vertex(pose_id, pose, fixed=False)

            # there is a previous pose we should make an edge with
            edge_id = self.pgo_id_store.get_new_id()

            information = [1., 1., 10.]

            edge = {'vertices':(self.prev_pose_id, pose_id),
                    'poses':(self.prev_pose, pose),
                    'eid':edge_id,
                    'information':information}

            self.pgo.add_edge(**edge)

        self.prev_pose = pose
        self.prev_pose_id = pose_id
        self.poses_since_last_optimization += 1
        self.raw_pose_trace.append(pose)


    @property
    def pose_trace(self):
        return list(self.corrected_pose_trace) + list(self.raw_pose_trace)


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
        self.poses_since_last_optimization = 0
        self.corrected_pose_trace += list(self.pgo.get_all_poses())
        self.raw_pose_trace = []
        self.pgo = PoseGraphOptimization(pgo_id = self.pgo_id)




if __name__=='__main__':
    import matplotlib.pyplot as plt
    try:
        __IPYTHON__
        plt.ion()
    except:
        pass


    dt = 0.5

    from auv import AUV
    auv = AUV(auv_id=0,
              init_pos = [0,0],
              init_heading = 0)

    id_store = PGO_VertexIdStore()

    pg = PoseGraph(pgo_id = 0,
                   pgo_id_store = id_store)

    auv.set_target([30,40])

    for i in range(100):
        auv.update(dt=0.5)
        pg.append_pose(auv.pose)

    pg.optimize()

    auv.set_target([-30,40])

    for i in range(100):
        auv.update(dt=0.5)
        pg.append_pose(auv.pose)

    auv_trace = np.array(auv.pose_trace)
    pg_trace = np.array(pg.pose_trace)
    plt.plot(auv_trace[:,0], auv_trace[:,1], c='g')
    plt.plot(pg_trace[:,0], pg_trace[:,1], c='b')
    plt.axis('equal')


