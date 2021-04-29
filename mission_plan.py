#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import os
import pickle

import numpy as np
import meshzoo
import itertools

from toolbox import geometry as geom

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
from matplotlib.patches import Polygon, Circle

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

def find_neighbors(points, neighbor_dist):
    """
    brute force check all pairs in points for distance
    and return a dict with indices
    {index:[index, index, index...]}
    """
    neighbors = {}
    print(f"Constructing graph out of {len(points)} points")
    for i,p in enumerate(points):
        neighbors[i] = []
        for i2,p2 in enumerate(points):
            if not all(p==p2):
                diff = p-p2
                is_close, dist = point_is_close(diff, neighbor_dist)
                if is_close:
                    neighbors[i].append(i2)

    return neighbors


def tris_are_neighbors(tri1, tri2):
    moved_vert = tri1.diff_verts(tri2)
    if len(moved_vert) != 1:
        print(f'Multiple jumps on: {tri1} -> {tri2}')
        return False
    return True


class FakeCoven(object):
    def __init__(self, name, vert_neighbors):
        self.path = []
        self.name = name
        self.current_vert = None
        self.vert_neighbors = vert_neighbors

    def __repr__(self):
        return f'Coven:{self.name} pathlen:{len(self.path)}'

    def move(self, new_vert):
        self.path.append(new_vert)
        if new_vert != self.current_vert:
            print(f'[Coven:{self.name}]\t moved from {self.current_vert} to {new_vert}')
        self.current_vert = new_vert



class Tri(object):
    def __init__(self,
                 vert_indices,
                 vert_neighbors,
                 tri_index):

        self.vert_indices = vert_indices
        self.vert_neighbors = vert_neighbors
        self.tri_index = tri_index

        self.verts_to_tris = {}

        self.children = {}
        self.parents = {}

        # which agent is on which vertex
        # this is important when doing flips
        # and we want to flip each _agent_ once
        self.vert_to_agent = {}
        self.agent_to_vert = {}
        self.agents = []
        for vert in self.vert_indices:
            self.put_agent(vert, None)

        self.in_path = False

    def put_agent(self, vert, agent):
        self.vert_to_agent[vert] = agent
        self.agent_to_vert[agent] = vert
        self.agents.append(agent)
        if agent is not None:
            agent.move(vert)

    def remove_agent_from_vert(self, vert):
        agent = self.vert_to_agent[vert]
        self.vert_to_agent[vert] = None
        self.agent_to_vert[agent] = None
        self.agents.remove(agent)


    def diff_verts(self, other_tri):
        verts = set(self.vert_indices)
        other_verts = set(other_tri.vert_indices)
        diff = set.difference(verts, other_verts)
        return list(diff)


    def flip(self, moving_vert_index=None, moving_agent=None, move_agents=True):
        """
        returns the Tri that is a flipped version of this one
        tri, the given vert_index is the only vert that isnt shared
        that one will be the moving vertex
        """
        assert moving_vert_index is not None or moving_agent is not None, "You gotta give me something to move!"
        if moving_vert_index is not None:
            assert moving_vert_index in self.vert_indices, "Given vertex index not found in this tri!"

        if moving_agent is not None:
            assert moving_agent in self.agent_to_vert.keys(), "Given agent not on this Tri"
            moving_vert_index = self.agent_to_vert[moving_agent]

        assert len(self.verts_to_tris)>0, "You must set verts_to_tris from outside before calling this!"



        # first, get the two stationary verts
        s1, s2 = set(self.vert_indices) - set([moving_vert_index])
        # then, find the common neighbors of these stationary vertices
        # that is different from the moving one
        ns1 = self.vert_neighbors[s1]
        ns2 = self.vert_neighbors[s2]
        intersection = set.intersection(set(ns1), set(ns2))
        common_neighbors = list(intersection - set([moving_vert_index]))
        assert len(common_neighbors) >= 1, "TriangleGraph grid size too small!"

        # there should always be just one
        common_neighbor = common_neighbors[0]

        verts = [s1,s2,common_neighbor]
        verts.sort()
        verts = tuple(verts)

        # find the existing tri from its verts
        child = self.verts_to_tris.get(verts)

        # record who our child is etc.
        self.children[moving_vert_index] = child
        child.parents[common_neighbor] = self

        if move_agents:
            # move the agents to the new tri if they were on this tri
            if all(self.vert_to_agent.values()):
                child.put_agent(common_neighbor, self.vert_to_agent[moving_vert_index])
                child.put_agent(s1, self.vert_to_agent[s1])
                child.put_agent(s2, self.vert_to_agent[s2])

                self.remove_agent_from_vert(moving_vert_index)
                self.remove_agent_from_vert(s1)
                self.remove_agent_from_vert(s2)


        tris_are_neighbors(self, child)
        return child




    def flip_thrice(self, agent_flip_order=None, vert_flip_order=None, move_agents=True):
        assert any([agent_flip_order, vert_flip_order]), "Gotta give the order of flips!"

        if agent_flip_order is not None:
            child = self.flip(moving_agent = agent_flip_order[0], move_agents=move_agents)
            child2 = child.flip(moving_agent = agent_flip_order[1], move_agents=move_agents)
            child3 = child2.flip(moving_agent = agent_flip_order[2], move_agents=move_agents)

        if vert_flip_order is not None:
            child = self.flip(moving_vert_index = vert_flip_order[0], move_agents=move_agents)
            child2 = child.flip(moving_vert_index = vert_flip_order[1], move_agents=move_agents)
            child3 = child2.flip(moving_vert_index = vert_flip_order[2], move_agents=move_agents)

        #  return [child, child2, child3]
        return child3




    def __repr__(self):
        return f'tri:{self.tri_index}->{self.vert_indices}'




class TriangleGraph(object):
    def __init__(self,
                 num_edges,
                 triangle_side,
                 rotate=0.):

        # points is an [N,2] array of points, centered on 0,0
        # and is a triangular mesh inside an hexagon.
        # the radius of the hex is always 1.0
        # second value decide the number of edges on the radius
        # with one vertex on the center and one at radius
        # cells are the indices of points that share a facet in the mesh
        verts, self.tri_cell_vert_indices = meshzoo.ngon(6, num_edges)
        # how large an area the covens will cover
        self.triangle_side = triangle_side
        # scale to 0-radius from 0-1 range
        self.radius = num_edges * triangle_side
        self.verts = geom.scale_range(verts, 0, self.radius, 0, 1)
        if abs(rotate)>0:
            self.verts = geom.vec2_rotate(self.verts, np.radians(rotate))
        # add some extra dist to make sure we dont get affected by floats
        self.vert_neighbors = find_neighbors(self.verts, self.radius/num_edges + 0.01)

        self.tris = []
        self.verts_to_tris = {}
        for i,indices in enumerate(self.tri_cell_vert_indices):
            verts = [indices[0], indices[1], indices[2]]
            # we sort the verts so that we _always_ know how to access the tri later
            verts.sort()
            verts = tuple(verts)
            tri = Tri(vert_indices = verts,
                      vert_neighbors = self.vert_neighbors,
                      tri_index = i)
            self.verts_to_tris[verts] = tri
            self.tris.append(tri)

        for tri in self.tris:
            tri.verts_to_tris = self.verts_to_tris


    def get_verts_of_tri(self, tri):
        return np.array([ self.verts[ind] for ind in tri.vert_indices ])


    def find_coven_paths(self,
                         root_tri_index,
                         num_coverage_triangles,
                         target=None,
                         towards_target=True,
                         plot=False,
                         center_x=True,
                         center_y=True):

        root = self.tris[root_tri_index]
        root_center = np.mean(self.get_verts_of_tri(root), axis=0)
        root_coords = self.get_verts_of_tri(root)

        if target is None:
            target = root_center
            towards_target = False
        else:
            target = np.array(target, dtype=float)

        closed_verts = set()
        open_tris = [root]
        tri_path = []
        full_tri_path = []
        head = None
        for i in range(num_coverage_triangles):
            prev_head = head
            head = open_tris[-1]
            tri_path.append(head)
            head.in_path = True
            [closed_verts.add(v) for v in head.vert_indices]
            open_tris.remove(head)

            # also add the current heads parents, going back 2 flips
            # the 3rd flip would already be in the path
            # there should be _one_ parent _at this time_
            try:
                # dig down from the previous head to current head
                # and find the current head in the prev's children
                # loops look scary, but at this point most of these children
                # have 0 or 1 child themselves. 
                prev_head_children = list(prev_head.children.values())
                found = False
                for child1 in prev_head_children:
                    if found:
                        break
                    prev_head_c_children = list(child1.children.values())
                    for child2 in prev_head_c_children:
                        if found:
                            break
                        prev_head_c_c_children = list(child2.children.values())
                        for child3 in prev_head_c_c_children:
                            if found:
                                break
                            if child3.tri_index == head.tri_index:
                                full_tri_path.append(child1)
                                full_tri_path.append(child2)
                                full_tri_path.append(child3)
                                found = True
                                break

            except IndexError:
                full_tri_path.append(head)
                print(f'{head.tri_index} has no parent')
            except AttributeError:
                full_tri_path.append(head)
                print(f'no prev_head yet: {head}')

            try:
                children = [head.flip_thrice(vert_flip_order=perm, move_agents=False)
                            for perm in itertools.permutations(head.vert_indices)]
            except:
                print(f'Tripled flipped {i} times')
                break
            untouched = []
            for child in children:
                verts = child.vert_indices
                if not any([vert in closed_verts for vert in verts]):
                    untouched.append(child)

            centers = [np.mean(self.get_verts_of_tri(tri), axis=0) for tri in untouched]
            dists = [geom.euclid_distance(center, target) for center in centers]
            options = np.array(list(zip(untouched, dists)))
            if towards_target:
                options = options[options[:,1].argsort()[::-1]]
            else:
                options = options[options[:,1].argsort()]
            open_tris = list(options[:,0])



        # now we construct the path from the path
        t = tri_path[0]

        covens  = [
                    FakeCoven('0', self.vert_neighbors),
                    FakeCoven('1', self.vert_neighbors),
                    FakeCoven('2', self.vert_neighbors)
        ]

        t.put_agent(t.vert_indices[0], covens [0])
        t.put_agent(t.vert_indices[1], covens [1])
        t.put_agent(t.vert_indices[2], covens [2])

        for i,current_tri in enumerate(full_tri_path):
            if i+1 < len(full_tri_path):
                next_tri = full_tri_path[i+1]
                moved_vert = current_tri.diff_verts(next_tri)
                #  assert tris_are_neighbors(current_tri, next_tri), f"MORE THAN ONE VERT MOVED!"
                moved_vert = moved_vert[0]
                current_tri.flip(moving_vert_index=moved_vert, move_agents=True)


        # and finally. 
        # we get the world coords for the coven paths
        flower_centers = []
        for coven in covens:
            flower_centers.append( np.array([self.verts[v] for v in coven.path]) )


        # center flower_centers and root_coords for efficient painting later
        mid_x, mid_y = 0, 0
        if center_x or center_y:
            mid_x, mid_y = np.mean(np.vstack(flower_centers), axis=0)
            for path in flower_centers:
                if center_x:
                    path[:,0] -= mid_x
                if center_y:
                    path[:,1] -= mid_y
            if center_x:
                root_coords[:,0] -= mid_x
            if center_y:
                root_coords[:,1] -= mid_y


        if plot:
            for i,tri in enumerate(tri_path):
                pts = self.get_verts_of_tri(tri)
                pts[:,0] -= mid_x
                pts[:,1] -= mid_y
                plt.gca().add_patch(Polygon(pts, facecolor='r', alpha=0.5))

                if i+1 < len(tri_path):
                    next_pts = self.get_verts_of_tri(tri_path[i+1])
                    next_pts[:,0] -= mid_x
                    next_pts[:,1] -= mid_y
                    center = np.mean(pts, axis=0)
                    next_center = np.mean(next_pts, axis=0)
                    diff = next_center - center
                    plt.arrow(center[0], center[1], diff[0], diff[1])

            for i,tri in enumerate(full_tri_path):
                pts = self.get_verts_of_tri(tri)
                pts[:,0] -= mid_x
                pts[:,1] -= mid_y
                plt.gca().add_patch(Polygon(pts, facecolor='y', alpha=0.5))
                # center = np.mean(pts, axis=0)
                #  plt.text(center[0], center[1], tri.tri_index)

                if i+1 < len(full_tri_path):
                    next_pts = self.get_verts_of_tri(full_tri_path[i+1])
                    next_pts[:,0] -= mid_x
                    next_pts[:,1] -= mid_y
                    center = np.mean(pts, axis=0)
                    next_center = np.mean(next_pts, axis=0)
                    diff = next_center - center
                    plt.arrow(center[0], center[1], diff[0], diff[1], color='r')

            for tri in open_tris:
                pts = self.get_verts_of_tri(tri)
                pts[:,0] -= mid_x
                pts[:,1] -= mid_y
                plt.gca().add_patch(Polygon(pts, facecolor='b', alpha=0.3))
                center = np.mean(pts, axis=0)



        return covens, flower_centers, root_coords




def construct_lawnmower_paths(num_agents,
                              num_hooks,
                              hook_len,
                              swath,
                              double_sided = True,
                              center_x = True,
                              center_y = True):
    assert num_agents%2==0 or not double_sided, "There must be even number of agents for a double-sided lawnmower plan!"

    def make_hook(flip_y = False, flip_x = False):
        side = 1
        if flip_y:
            side = -1

        direction = 1
        if flip_x:
            direction = -1

        p0 = np.array((0.,0.))
        p1 = [0, side*hook_len]
        p2 = [direction*swath, side*hook_len]
        p3 = [direction*swath, 0]
        p4 = [direction*2*swath, 0]
        return np.array([p0, p1, p2, p3, p4])

    def make_lawnmower_path(starting_pos, flip_y=False, flip_x=False):
        path = []
        for hook_i in range(num_hooks):
            if hook_i>=1:
                starting_pos = path[-1]

            hook = make_hook(flip_y, flip_x)
            hook += starting_pos
            path.extend(hook)

        return np.array(path)

    paths = []
    flip_x = False
    if double_sided:
        num_agents_side = int(num_agents/2)
        num_sides = 2
    else:
        num_agents_side = int(num_agents)
        num_sides = 1

    for side in range(num_sides):
        flip_x = side%2==1
        for agent_i in range(num_agents_side):
            flip_y = True
            if agent_i%2 == 0:
                flip_y = False

            if agent_i >= 1:
                pos  = paths[-1][0].copy()
                if flip_y:
                    pos[1] += 2*hook_len+swath
                else:
                    pos[1] += swath
            else:
                pos  = np.array([swath, swath])

            # the other agents copy the position of the
            # first one, so we just need to adjust the first one
            if double_sided and agent_i == 0:
                if flip_x:
                    pos[0] -= swath/2
                else:
                    pos[0] += swath/2

            print(f'{agent_i}:{pos}')

            path = make_lawnmower_path(pos , flip_y, flip_x)
            paths.append(path)

    paths = np.array(paths)
    # and then center the paths on x and y
    # makes map painting more efficient
    if center_x or center_y:
        mid_x, mid_y = np.mean(np.vstack(paths), axis=0)
        for path in paths:
            if center_x:
                path[:,0] -= mid_x
            if center_y:
                path[:,1] -= mid_y


    return paths



def make_triangle_graph(num_edges, triangle_side, rotate=0, save_dir=''):

    if save_dir is None:
        tg = TriangleGraph(num_edges = num_edges,
                           triangle_side = triangle_side,
                           rotate = rotate)
        return tg


    try:
        filename = os.path.join(save_dir, f"triangle_graph_{int(num_edges)}_{int(triangle_side)}_{int(rotate)}.pickle")
        tg = pickle.load(open(filename, 'rb'))
        print("Found saved graph!")
        return tg
    except:
        tg = TriangleGraph(num_edges = num_edges,
                           triangle_side = triangle_side,
                           rotate = rotate)
        pickle.dump(tg, open(filename, 'wb'))
        print(f"Saved graph to:{filename}")
        return tg


if __name__=='__main__':
    plt.ion()
    plt.axis('equal')

    try:
        plt.cla()
    except:
        pass

    # paths = construct_lawnmower_paths(num_agents = 6,
                                      # num_hooks = 10,
                                      # hook_len = 100,
                                      # swath = 50)

    # for path in paths:
        # plt.plot(path[:,0], path[:,1])


    swath = 50
    coven_radius = 2*swath
    triangle_side = np.sqrt(3)*coven_radius

    tg = make_triangle_graph(num_edges=20,
                             triangle_side=triangle_side,
                             rotate=30,
                             save_dir=None)


    # print("Plotting graph")
    # plt.scatter(tg.verts[:,0], tg.verts[:,1], alpha=0.1, c='b', marker='.')
    # for i,p in enumerate(tg.verts):
        # for i2 in tg.vert_neighbors[i]:
            # p2 = tg.verts[i2]
            # plt.plot([p[0], p2[0]], [p[1], p2[1]], color='b', alpha=0.1)


    print("Finding coven paths")
    vert_paths, flower_centers, root_coords  = tg.find_coven_paths(root_tri_index=0,
                                                                   num_coverage_triangles=50,
                                                                   target=[-500,0],
                                                                   towards_target=False,
                                                                   plot=True,
                                                                   center_x=True,
                                                                   center_y=True)




    print("Plotting path")
    colors = ['r', 'g', 'b']
    for color, path in zip(colors, flower_centers):
        for i in range(len(path)):
            if i+1 < len(path):
                c1 = path[i]
                c2 = path[i+1]
                diff = c2-c1
                if any(abs(diff)>0):
                    plt.arrow(c1[0], c1[1], diff[0], diff[1],
                              color=color,
                              head_width=40,
                              length_includes_head=True)
                    plt.gca().add_patch( Circle( (c1[0], c1[1]), radius=coven_radius, facecolor=color, alpha=0.5) )





