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


class OdomEdge(object):
    def __init__(self,
                 parent_vertex,
                 child_vertex,
                 edge_id,
                 information,
                 pg_id):

        self.parent_vertex = parent_vertex
        self.child_vertex = child_vertex
        self.edge_id = edge_id
        self.information = information
        # which graph do i belong to?
        self.pg_id = pg_id

    def __repr__(self):
        return f"<OdoE:{self.edge_id}%{self.parent_vid}->{self.child_vid} of pg{self.pg_id}>"

    @property
    def as_dict(self):
        edge = {'vertices':(self.parent_vertex.vid, self.child_vertex.vid),
                'poses':(self.parent_vertex.pose, self.child_vertex.pose),
                'eid':self.edge_id,
                'information':self.information}
        return edge

    @property
    def parent_vid(self):
        return self.parent_vertex.vid

    @property
    def parent_pose(self):
        return np.array(self.parent_vertex.pose)

    @property
    def child_vid(self):
        return self.child_vertex.vid

    @property
    def child_pose(self):
        return np.array(self.child_vertex.pose)


class MeasuredEdge(object):
    def __init__(self,
                 parent_vid,
                 child_vid,
                 parent_pose,
                 child_pose,
                 edge_id,
                 information,
                 pg_id):

        self.parent_vid = parent_vid
        self.child_vid = child_vid
        self.parent_pose = np.array(parent_pose)
        self.child_pose = np.array(child_pose)
        self.edge_id = edge_id
        self.information = information
        # which graph do i belong to?
        self.pg_id = pg_id


    def __repr__(self):
        return f"<MeaE:{self.edge_id}%{self.parent_vid}->{self.child_vid} of pg{self.pg_id}>"

    @property
    def as_dict(self):
        edge = {'vertices':(self.parent_vid, self.child_vid),
                'poses':(self.parent_pose, self.child_pose),
                'eid':self.edge_id,
                'information':self.information}
        return edge


class SummaryEdge(object):
    def __init__(self,
                 parent_vertex,
                 child_vertex,
                 edge_id,
                 information,
                 pg_id):
        """
        basically a copy of the odom edge, but separate in case i need to specialize it
        """

        self.parent_vertex = parent_vertex
        self.child_vertex = child_vertex
        self.edge_id = edge_id
        self.information = information
        # which graph do i belong to?
        self.pg_id = pg_id

    def __repr__(self):
        return f"<SummE:{self.edge_id}%{self.parent_vid}->{self.child_vid} of pg{self.pg_id}>"

    @property
    def as_dict(self):
        edge = {'vertices':(self.parent_vertex.vid, self.child_vertex.vid),
                'poses':(self.parent_vertex.pose, self.child_vertex.pose),
                'eid':self.edge_id,
                'information':self.information}
        return edge

    @property
    def parent_vid(self):
        return self.parent_vertex.vid

    @property
    def parent_pose(self):
        return np.array(self.parent_vertex.pose)

    @property
    def child_vid(self):
        return self.child_vertex.vid

    @property
    def child_pose(self):
        return np.array(self.child_vertex.pose)




class Vertex(object):
    def __init__(self,
                 vid,
                 pose,
                 fixed,
                 pg_id):

        self.vid = vid
        self.pose = pose
        self.fixed = fixed
        # which graph do i belong to?
        self.pg_id = pg_id
        # list of pis that point TO this pose
        self.parent_vids = []
        # that this pose points towards
        self.children_vids = []
        # edges that are shared between this pose and either a parent or a child
        # parent/child id -> edge_id
        self.connected_edge_ids = {}

    def __repr__(self):
        return f"<V:{self.vid}@{self.pose} of pg{self.pg_id}>"

    @property
    def connected_vertex_ids(self):
        return self.parent_vids + self.children_vids


    def add_child(self, child_vid, edge_id):
        if child_vid not in self.connected_edge_ids.keys():
            self.children_vids.append(child_vid)
            self.connected_edge_ids[child_vid] = edge_id


    def add_parent(self, parent_vid, edge_id):
        if parent_vid not in self.connected_edge_ids.keys():
            self.parent_vids.append(parent_vid)
            self.connected_edge_ids[parent_vid] = edge_id


    def update_pose(self, pose):
        self.pose = pose



class PoseGraph(object):
    def __init__(self,
                 pg_id,
                 id_store):

        self.pg_id = pg_id
        self.id_store = id_store

        self.root_vertex = None

        # id -> Vertex/Edge
        self.all_vertices = {}
        self.all_edges = {}

        # the vertex object that is at the end of the odometry chain
        self.odom_tip_vertex = None
        # this vertex should be the tip of the summary graph
        self.summary_tip_vertex = None

        # the tips of other graphs that we interacted with
        self.other_odom_tip_vertices = {}

        # the id of the pose that we last did an optim on
        self.last_optim_vert_id = None

        # debugging stuff
        self._large_fill_verts = []



    def log(self, *args):
        if len(args) == 1:
            args = args[0]
        print(f'[PG:{self.pg_id}]\t{args}')



    @property
    def odom_pose_trace(self):
        poses = []
        # start from the tip and move backwards to create the chain
        vertex  = self.odom_tip_vertex
        while vertex is not None:
            poses.append(vertex.pose)
            parents = vertex.parent_vids
            if len(parents) < 1:
                break
            next_vertex = None
            for parent_vid in parents:
                # out of all the parents (there might be many)
                # find the one that belongs to this pose graph
                parent = self.all_vertices.get(parent_vid)
                if parent is None:
                    continue
                # this parent might be connected to us with a summary edge
                # in that case, dont use that edge because that will skip a bunch
                # of vertices in the middle
                eid_to_parent = vertex.connected_edge_ids.get(parent_vid)
                edge_to_parent = self.all_edges.get(eid_to_parent)
                if parent.pg_id == self.pg_id and type(edge_to_parent) == OdomEdge:
                    next_vertex = parent

            vertex = next_vertex
            next_vertex = None

        return np.array(poses)

    @property
    def fixed_poses(self):
        poses = []
        for vid, vertex in self.all_vertices.items():
            if vertex.fixed:
                poses.append(vertex.pose)
        return np.array(poses)

    @property
    def measured_edges(self):
        edges = []
        for eid, edge in self.all_edges.items():
            if type(edge) == MeasuredEdge:
                edges.append(edge)
        return edges

    @property
    def odom_edges(self):
        edges = []
        for eid, edge in self.all_edges.items():
            if type(edge) == OdomEdge:
                edges.append(edge)
        return edges


    @property
    def self_odom_edges(self):
        edges = []
        for eid, edge in self.all_edges.items():
            if type(edge) == OdomEdge and edge.pg_id == self.pg_id:
                edges.append(edge)
        return edges


    @property
    def foreign_edges(self):
        edges = []
        for eid, edge in self.all_edges.items():
            if edge.pg_id != self.pg_id:
                edges.append(edge)
        return edges


    @property
    def self_summary_edges(self):
        edges = []
        for eid, edge in self.all_edges.items():
            if type(edge) == SummaryEdge and edge.pg_id == self.pg_id:
                edges.append(edge)
        return edges


    @property
    def foreign_poses(self):
        poses = []
        for vid, vertex in self.all_vertices.items():
            if vertex.pg_id != self.pg_id:
                poses.append(vertex.pose)
        return np.array(poses)




    def append_odom_pose(self,
                         pose):
        """
        Add a new odometry pose to the tip of this graphs chain.
        Fixes the first pose in the graph.
        """

        vid = self.id_store.get_new_id()

        if type(pose) != np.ndarray:
            pose = np.array(pose)


        if self.odom_tip_vertex is None:
            # first ever pose in the graph
            # has no parent or children
            v = Vertex(vid = vid,
                       pose = pose,
                       fixed = True,
                       pg_id = self.pg_id)
            self.all_vertices[vid] = v
            self.odom_tip_vertex = v
            self.root_vertex = v
        else:
            # not the first odom, not fixed
            # and has a parent
            v = Vertex(vid = vid,
                       pose = pose,
                       fixed = False,
                       pg_id = self.pg_id)
            self.all_vertices[vid] = v

            # make an edge that connects the current tip
            # of the odom chain to this new pose
            eid = self.id_store.get_new_id()
            e = OdomEdge(parent_vertex = self.odom_tip_vertex,
                         child_vertex = v,
                         edge_id = eid,
                         information = [1., 1., 1000.],
                         pg_id = self.pg_id)
            self.all_edges[eid] = e

            # and tell the current odom tip that it now has a new child
            self.odom_tip_vertex.add_child(child_vid = vid,
                                           edge_id = eid)
            # tell this new pose about its parent
            v.add_parent(parent_vid = self.odom_tip_vertex.vid,
                         edge_id = eid)

            # and finally move the tip to this new pose
            self.odom_tip_vertex = v



    def measure_tip_to_tip(self,
                           self_real_pose,
                           other_real_pose,
                           other_pg):
        """
        Adds a measurement from the tip of this graph to the tip of another graph.
        The real_poses are used to create the measurement, but the vertices are used from
        the respective graph's odometry chain tips.
        """

        own_vert = self.odom_tip_vertex
        other_vert = other_pg.odom_tip_vertex
        if own_vert is None or other_vert is None:
            self.log("a tip was None")
            return

        if self.all_vertices.get(other_vert.vid) is None:
            self.all_vertices[other_vert.vid] = other_vert


        eid = self.id_store.get_new_id()
        e = MeasuredEdge(parent_vid=own_vert.vid,
                         child_vid=other_vert.vid,
                         parent_pose=self_real_pose,
                         child_pose=other_real_pose,
                         edge_id=eid,
                         information=[100.,100.,1000.],
                         pg_id=self.pg_id)

        #XXX this might break a lot of things? maybe.
        own_vert.add_child(other_vert.vid, eid)
        other_vert.add_parent(own_vert.vid, eid)

        self.all_edges[eid] = e



    def get_chain_between_verts(self, start_vertex, end_vertex=None, use_summary=False, reason=None):
        """
        follow the chain from start to end, return a list of vertex objects that make up the chain
        if end_vertex is None, use our odom tip
        """

        if end_vertex is None:
            end_vertex = self.odom_tip_vertex

        if start_vertex is None:
            start_vertex = self.root_vertex

        if start_vertex is None:
            self.log("Start vertex is None")
            return {}, {}, True

        if use_summary:
            self.summarize_to_tip(reason=reason)
            # self.log(f"Summarized into {len(self.self_summary_edges)} summary edges")

        vertex_chain = {}
        edges = {}
        complete = False
        # start from the tip and move backwards to create the chain
        vertex  = end_vertex
        jumped_with_summary = False
        while vertex is not None:

            vertex_chain[vertex.vid] = vertex

            if vertex.vid == start_vertex.vid:
                self.log("Reached start vertex")
                complete = True
                break

            parents = vertex.parent_vids
            next_vertex = None
            for parent_vid in parents:
                # out of all the parents (there might be many)
                # find the one that belongs to this pose graph
                parent = self.all_vertices.get(parent_vid)
                if parent is None:
                    self.log("Parent is none")
                    continue

                if parent.pg_id == self.pg_id:
                    eid_to_parent = vertex.connected_edge_ids.get(parent_vid)
                    edge_to_parent = self.all_edges.get(eid_to_parent)
                    next_vertex = parent
                    edges[eid_to_parent] = edge_to_parent
                    if use_summary and type(edge_to_parent) == SummaryEdge:
                        self.log(f"Jumped with summary edge from {vertex.vid} to {parent.vid}")
                        # early stop to make sure the last "next_vertex" is the one
                        # we got to through a summary edge
                        break


            # if we did find a parent, keep following that
            vertex = next_vertex
            next_vertex = None

        return vertex_chain, edges, complete



    def summarize_to_tip(self, reason=None):
        # from the previous summary tip to the current odom tip,
        # create one edge that summarizes all the edges in between
        # the agent should decide WHEN to do this, not the pose graph
        # should the pose graph make decisions on "summarizing the summary"?
        # probably not honestly. too much complexity for probably no return

        # if we have done a summary before, start there,
        # otherwise start from the root (which is fixed)
        if self.summary_tip_vertex is not None:
            chain_root_vert = self.summary_tip_vertex
        else:
            chain_root_vert = self.root_vertex

        # early quit if the summary tip IS the odom tip
        if chain_root_vert.vid == self.odom_tip_vertex.vid:
            if reason is not None:
                self.log(f"{reason}  Chain root is odom tip~")
            return []

        def follow_until_split(current_tip):
            """
            given a vertex, follow it through the chain of vertices
            that belong to this pg and return the chain and the information matrices
            """
            chain = []
            # in the meantime, collect the covariances of the edges on the chain
            chain_informations = []
            while True:
                chain.append(current_tip)
                if len(current_tip.children_vids) > 1 and len(chain) > 1:
                    # this vert has multiple children.
                    # MOST LIKELY it has a child going to some
                    # other PG.
                    # this is a split in the chain, thus
                    # the summary must be split here too
                    # handle the multiple children in the case that
                    # this is the FIRST vert in the chain
                    # self.log("Tip has many children and not the first in chain")
                    return chain, chain_informations
                # else:
                    # self.log(f"Tip has {len(current_tip.children_vids)} children, chain is {len(chain)} long")

                if len(current_tip.children_vids) < 1:
                    # no children, must be the odom tip
                    # self.log(f"Tip {current_tip} has no children")
                    return chain, chain_informations

                # by this point, we know for sure this vert has at least one child
                # follow its child that belongs to our pg
                children_verts = [self.all_vertices.get(vid) for vid in current_tip.children_vids]
                own_children_verts = list(filter(lambda v: v.pg_id == self.pg_id, children_verts))

                # there should really be exactly one here
                if len(own_children_verts) != 1:
                    # self.log("Either multiple or no children in the same graph")
                    return chain, chain_informations

                child_vert = own_children_verts[0]

                # get the edge between current tip and its selected child
                edge_id = current_tip.connected_edge_ids.get(child_vert.vid)
                edge = self.all_edges.get(edge_id)
                chain_informations.append(edge.information)
                # follow it~
                current_tip = child_vert

        # starting from the root, follow the children until there are multiple children
        # and collect those disjointed parts
        current_tip = chain_root_vert
        chains = []
        while True:
            if current_tip.vid == self.odom_tip_vertex.vid:
                # self.log("Chains reached odom tip")
                break

            chain, informations = follow_until_split(current_tip)
            chains.append(chain)

            # given the chain, now we want a new edge from the first to last
            # vertex in it with the info matrix 1/sum(1/inf).
            # these informatons are just 1d arrays for a diagonal matrix, so 1/ works juuuust fine
            inverted = [1./np.array(information) for information in informations]
            summary_information = 1./np.sum(inverted, axis=0)
            #XXX do i want to keep the orientation cov unchanged? kinda...
            summary_information[2] = informations[0][2]

            # create the edge and add it to the dict of all edges
            new_edge_id = self.id_store.get_new_id()
            edge = SummaryEdge(parent_vertex = chain[0],
                               child_vertex = chain[-1],
                               edge_id = new_edge_id,
                               information = summary_information,
                               pg_id = self.pg_id)
            self.all_edges[new_edge_id] = edge
            # if len(chain) > 10:
                # self.log(f"Added summary edge {edge} for {len(chain)} vert chain")

            # update the vertices themselves that they got a new edge between them now
            chain[0].add_child(chain[-1].vid, new_edge_id)
            chain[-1].add_parent(chain[0].vid, new_edge_id)

            # and start a new chain from the last vertex
            current_tip = chain[-1]

        if reason is not None:
            self.log(f"{reason}  {len(chains)} found")

        # and finally, record our current summarized tip so we dont try
        # to summarize from the root all the time in subsequent summarizations
        self.summary_tip_vertex = self.odom_tip_vertex

        return chains


    def summarize_chain(self, vertices, edges):
        """
        given dicts of {id:vertex} and {id:edge} of a singly linked chain, summarize them by
        creating a single edge between the first and last vertices.
        """

        # if there are no edges, be transparent
        if len(edges) < 2:
            self.log(f"No need to summarize, too few edges={len(edges)}")
            return vertices, edges

        # first find the first and last vertex
        vids = list(vertices.keys())
        first_vert = vertices.get(min(vids))
        last_vert = vertices.get(max(vids))

        self.log(f"Summarizing between {first_vert} and {last_vert}, {len(vertices)},{len(edges)}")
        assert first_vert is not None and last_vert is not None

        # then sum up the info matrices of the edges
        # information matrices are diagonal matrices stored as lists
        # to sum'em, convert to covariances by 1/'ing.
        # then sum the covs
        # then 1/ the cov into info
        infos = [1/np.array(edge.information) for eid,edge in edges.items()]
        new_info = 1/np.sum(np.array(infos), axis=0)
        #XXX keep the orientation information fixed tho, compass is constant
        an_edge = edges.get(list(edges.keys())[0])
        new_info[2] = an_edge.information[2]

        # create the edge and add it to the dict of all edges
        new_edge_id = self.id_store.get_new_id()
        edge = SummaryEdge(parent_vertex = first_vert,
                           child_vertex = last_vert,
                           edge_id = new_edge_id,
                           information = new_info,
                           pg_id = self.pg_id)

        # update the vertices themselves that they got a new edge between them now
        first_vert.add_child(last_vert.vid, new_edge_id)
        last_vert.add_parent(first_vert.vid, new_edge_id)

        vert_dict = {first_vert.vid:first_vert,
                     last_vert.vid:last_vert}
        edge_dict = {new_edge_id:edge}

        return vert_dict, edge_dict





    def fill_in_since_last_interaction(self, other_pg, use_summary=False):
        """
        add all the verts and edges from another pg, between its current tip and
        our last known tip.
        """
        start_vert = self.other_odom_tip_vertices.get(other_pg.pg_id)
        # from the previously known latest vertex to whatever tip the other pg has
        # this is where we can measure bandwidth requirement too
        self.log(f"Getting data from {other_pg.pg_id}")
        vertices, edges, complete = other_pg.get_chain_between_verts(start_vertex = start_vert,
                                                                     end_vertex = None,
                                                                     use_summary=use_summary)

        if use_summary:
            # here, take these verts and edges and further summarize them to be 2 verts and 1 edge in total
            vertices, edges = self.summarize_chain(vertices, edges)

        self.all_vertices.update(vertices)
        self.all_edges.update(edges)

        if len(vertices) > 10:
            self._large_fill_verts.append(self.odom_tip_vertex)
            self.log(f"{len(vertices)}, from {start_vert}, complete={complete}")

        self.other_odom_tip_vertices[other_pg.pg_id] = other_pg.odom_tip_vertex

        return len(vertices), len(edges)



    def make_pgo(self, use_summary=False, start_from=None, save=False):

        if start_from is None:
            start_from = -1

        # the optimizer object
        pgo = PoseGraphOptimization(pgo_id = self.pg_id)
        # add all the vertices
        for vid, vert in self.all_vertices.items():
            if vid >= start_from:
                pgo.add_vertex(v_id = vid,
                               pose = vert.pose,
                               fixed = vert.fixed)

        # add all the edges
        for eid, edge in self.all_edges.items():
            if eid >= start_from:
                # depending on if we want to optimize over the full graph
                # or the summarized graph, but never BOTH
                if use_summary and type(edge) in (SummaryEdge, MeasuredEdge):
                    pgo.add_edge(**edge.as_dict)

                if not use_summary and type(edge) in (MeasuredEdge, OdomEdge):
                    pgo.add_edge(**edge.as_dict)


        if save:
            if self.last_optim_vert_id is None:
                self.last_optim_vert_id = 0
            pgo.save(f"pg_{self.pg_id}_{self.last_optim_vert_id}.g2o")

        return pgo




    def optimize(self, use_summary=False, save_before=False):
        # first, out of our own graph, construct a PGO
        # using vertices that go until the previous optimized tip
        # optimize the PGO
        # from the PGO graph, update our own vertices
        # mark our updated tip as fixed
        # remember our updated tip for next time

        if use_summary:
            # this handles the tips being equal already
            self.summarize_to_tip()

        pgo = self.make_pgo(use_summary=use_summary, save=save_before, start_from=self.last_optim_vert_id)

        success = pgo.optimize(max_iterations=100)

        if not success:
            self.log(f"Optim failed at v:{self.last_optim_vert_id}")
            return success

        # update our own vertices from the pgo
        for vid in pgo.vertices():
            new_pose = pgo.get_pose_array(vid)
            self.all_vertices[vid].update_pose(new_pose)


        self.last_optim_vert_id = self.odom_tip_vertex.vid
        # self.all_vertices[self.last_optim_vert_id].fixed = True

        return success









if __name__=='__main__':
    import matplotlib.pyplot as plt
    from mission_plan import construct_lawnmower_paths
    from auv import AUV

    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch

    try:
        __IPYTHON__
        plt.ion()
    except:
        pass


    dt = 0.5
    id_store = PGO_VertexIdStore()
    comm_dist = 50
    target_threshold = 3



    paths = construct_lawnmower_paths(num_agents = 2,
                                      num_hooks=3,
                                      hook_len=100,
                                      swath=50,
                                      gap_between_rows=-5,
                                      double_sided=False)

    auvs = []
    pgs = []
    for i,path in enumerate(paths):
        auv = AUV(auv_id=i,
                  init_pos = path[0],
                  init_heading = 0,
                  target_threshold=target_threshold)


        pg = PoseGraph(pg_id = i,
                       id_store = id_store)


        auvs.append(auv)
        pgs.append(pg)



    error = 0
    def run(wp_idx):
        for auv,path in zip(auvs, paths):
            auv.set_target(path[wp_idx])

        global error
        for i in range(150):
            # error += 0.1
            for pg, auv in zip(pgs,auvs):
                auv.update(dt=dt)
                if pg.pg_id%2 == 0:
                    scrambled_pose = auv.apose + [error, error, 0]
                else:
                    scrambled_pose = auv.apose - [error, error, 0]

                pg.append_odom_pose(scrambled_pose)

            if i%5==0:
                for pg, auv in zip(pgs,auvs):
                    for pg2, auv2 in zip(pgs,auvs):
                        if auv2.auv_id == auv.auv_id:
                            continue

                        dist = geom.euclid_distance(auv.pose[:2], auv2.pose[:2])
                        if dist < comm_dist:
                            pg.measure_tip_to_tip(self_real_pose=auv.pose,
                                                  other_real_pose=auv2.pose,
                                                  other_pg=pg2)

                            pg.fill_in_since_last_interaction(pg2, use_summary = True)
                            pg.optimize(use_summary=True, save_before=False)





            reached = [auv.reached_target for auv in auvs]
            if all(reached):
                break

    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    for i in range(10):
        run(i+1)


    print("------")
    # chains = pgs[1].summarize_to_tip()

    # for pg in pgs:
        # pg.summarize_to_tip()
        # pg.optimize(use_summary=True, save_before=False)


    # import sys
    # sys.exit(0)


    print('Plotting')
    plt.axis('equal')



    for path in paths:
        p = np.array(path)
        plt.plot(p[:,0], p[:,1], c='k', alpha=0.2, linestyle=':')

    for c,pg in zip(colors,pgs):
        t = pg.odom_pose_trace
        plt.scatter(t[:,0], t[:,1], alpha=0.2, marker='.', s=5, c=c)
        for p in pg.fixed_poses:
            plt.text(p[0], p[1], 'F', c=c)

        for edge in pg.measured_edges:
            p1 = edge.parent_pose
            p2 = edge.child_pose
            diff = p2-p1
            plt.arrow(p1[0], p1[1], diff[0], diff[1],
                      alpha=0.3, color=c, head_width=0.5, shape='left',
                      length_includes_head=True)

        for edge in pg.self_odom_edges:
            p1 = edge.parent_pose
            p2 = edge.child_pose
            diff = p2-p1
            plt.arrow(p1[0], p1[1], diff[0], diff[1],
                      alpha=0.2, color=c, head_width=0.3, length_includes_head=True)

        for edge in pg.self_summary_edges:
            p1 = edge.parent_pose
            p2 = edge.child_pose
            diff = p2-p1
            plt.arrow(p1[0], p1[1], diff[0], diff[1],
                      alpha=0.3, color=c, head_width=2.2, length_includes_head=True)


        pg_marker = ' '*pg.pg_id + 'f'
        for p in pg.foreign_poses:
            tp = TextPath(p[:2], pg_marker, size=0.1)
            plt.gca().add_patch(PathPatch(tp, color=c))

        # for edge in pg.foreign_edges:
            # p1 = edge.parent_pose
            # p2 = edge.child_pose
            # p = (p2+p1)/2
            # tp = TextPath(p[:2], pg_marker, size=0.1)
            # plt.gca().add_patch(PathPatch(tp, color=c))

    # for c,auv in zip(colors, auvs):
        # t = auv.pose_trace
        # plt.scatter(t[:,0], t[:,1], alpha=0.1, marker='.', c=c)


    print('Done plotting')



