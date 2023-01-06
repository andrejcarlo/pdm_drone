"""
Copied from "3D RRT.ipynb" to be imported in other python files
"""
# imports

import math
import time
from random import random, randrange, uniform, choice

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
# line class with some metrics to ease graph creation
class Line():
    # initialize some metrics
    def __init__(self, p0, p1):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.direction = np.array(p1) - np.array(p0)
        self.distance = np.linalg.norm(self.direction)
        self.direction = self.direction/self.distance # normalize

    def path(self, t):
        return self.p0 + t * self.dirn

# graph class
class Graph:
    # initialize some metrix
    def __init__(self, startposition, endposition):
        self.startposition = startposition
        self.endposition = endposition
        self.found_path = False
        # size of searchbox
        self.searchboxsize_x = (startposition[0] - endposition[0])
        self.searchboxsize_y = (startposition[1] - endposition[1])
        self.searchboxsize_z = (startposition[2] - endposition[2])
        # location of seachbox startpoint: two times size of box between start and end position
        self.searchbegin_x = self.startposition[0] + (self.searchboxsize_x/2)
        self.searchbegin_y = self.startposition[1] + (self.searchboxsize_y/2)
        self.searchbegin_z = self.startposition[2] + (self.searchboxsize_z/2)

        self.vertices = [startposition]
        self.edges = [] # contains the indices of the vertices
        self.indices = {startposition:0} # dictonary with indices for the vertices
        self.distances = {0:0.} # dictonary with distances for each index
        self.connections = {0:[]} # dictonary with connecting node(s)

    # creating a random position within search frame
    def randpos(self, obstacles = None, rand_radius = None, bias = None, obstacle_bias = False):
        if obstacle_bias == False:
            # create random values between 0 and 1
            x = random()
            y = random()
            z = random()
            # convert to value within searchbox
            posx = self.searchbegin_x - x*self.searchboxsize_x*2
            posy = self.searchbegin_y - y*self.searchboxsize_y*2
            posz = self.searchbegin_z - z*self.searchboxsize_z*2

        else:
            select = random()
            if select > bias:
                # create random values between 0 and 1
                x = random()
                y = random()
                z = random()
                # convert to value within searchbox
                posx = self.searchbegin_x - x*self.searchboxsize_x*2
                posy = self.searchbegin_y - y*self.searchboxsize_y*2
                posz = self.searchbegin_z - z*self.searchboxsize_z*2
            else:
                rand_obs = randrange(0,len(obstacles))
                direction = np.array((uniform(-1,1), uniform(-1,1), uniform(-1,1)))
                length = np.linalg.norm(direction)
                rand_pos = (direction/length)*(obstacles[rand_obs][3] + uniform(0.,rand_radius))
                posx = obstacles[rand_obs][0]+ rand_pos[0]
                posy = obstacles[rand_obs][1]+ rand_pos[1]
                posz = obstacles[rand_obs][2]+ rand_pos[2]

        return posx, posy, posz
    def add_vertex(self, vertex):
        try: # check if vertex already exists
            idx = self.indices[vertex]
        except:# otherwise add to list and dictionaries
            idx = len(self.vertices)
            self.vertices.append(vertex)
            self.indices[vertex] = idx
            self.connections[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2)) # add edge based on indices
        self.connections[idx1].append((idx2, cost)) #add the connecting nodes and costs
        self.connections[idx2].append((idx1, cost))

# RRT algorithm
def RRT(startposition, endposition, obstacles, iterations, threshold, rand_radius, bias, obstacle_bias, stepsize, goal):
    RRT_Graph = Graph(startposition, endposition)
    optimal = distance(startposition,endposition)
    if goal is not None: iterations = 10000
    for i in range(iterations):
        randpos = RRT_Graph.randpos(obstacles, rand_radius, bias, obstacle_bias)

        if RRT_Graph.found_path and goal is not None:
            found_path = dijkstra(RRT_Graph)
            length = 0.
            for path in range(len(found_path)-1):
                length = length + distance((found_path[path][0],found_path[path][1],found_path[path][2]),(found_path[path+1][0],found_path[path+1][1],found_path[path+1][2]))

            if goal is not None and (length/optimal) <= goal:
                break

        if in_obstacle(obstacles, randpos):
            continue

        near_vex, near_idx = nearest_node(RRT_Graph, randpos, obstacles)
        if near_vex is None:
            continue

        if stepsize is not None:
            randpos = reshape(randpos, near_vex, stepsize)

        new_idx = RRT_Graph.add_vertex(randpos)
        dist = distance(near_vex,randpos)
        RRT_Graph.add_edge(new_idx,near_idx,dist)

        end_distance = distance(randpos,RRT_Graph.endposition)
        if end_distance < threshold:
            end_index = RRT_Graph.add_vertex(RRT_Graph.endposition)
            RRT_Graph.add_edge(new_idx,end_index,end_distance)
            RRT_Graph.found_path = True
    return RRT_Graph

# RRT* algorithm
def RRT_s(startposition, endposition, obstacles, iterations, threshold, rand_radius, bias, obstacle_bias, stepsize, goal):
    RRT_Graph = Graph(startposition, endposition)
    gamma = 10*stepsize*pow((1+1/3),(1/3))
    optimal = distance(startposition,endposition)
    if goal is not None: iterations = 5000
    for i in range(iterations):
        randpos = RRT_Graph.randpos(obstacles, rand_radius, bias, obstacle_bias)

        if RRT_Graph.found_path and goal is not None:
            found_path = dijkstra(RRT_Graph)
            length = 0.
            for path in range(len(found_path)-1):
                length = length + distance((found_path[path][0],found_path[path][1],found_path[path][2]),(found_path[path+1][0],found_path[path+1][1],found_path[path+1][2]))

            if goal is not None and (length/optimal) <= goal:
                break

        if in_obstacle(obstacles, randpos):
            continue

        near_vex, near_idx = nearest_node(RRT_Graph, randpos, obstacles)
        if near_vex is None:
            continue

        if stepsize is not None:
            randpos = reshape(randpos, near_vex, stepsize)

        new_idx = RRT_Graph.add_vertex(randpos)
        dist = distance(near_vex,randpos)
        RRT_Graph.add_edge(new_idx,near_idx,dist)
        RRT_Graph.distances[new_idx] = RRT_Graph.distances[near_idx] + dist #update the distances
        n = len(RRT_Graph.vertices)

        # update nearby vertices distance (if shorter)
        for vertex in RRT_Graph.vertices:
            if vertex == randpos:
                continue

            dist = distance(vertex, randpos)
            if dist > gamma*pow((np.log(n)/n),1/3):
                continue

            line = Line(vertex, randpos)
            if trough_obstacle(obstacles, line):
                continue

            idx = RRT_Graph.indices[vertex]
            if (RRT_Graph.distances[new_idx] + dist) < (RRT_Graph.distances[idx]):
                RRT_Graph.add_edge(idx, new_idx, dist)
                RRT_Graph.distances[idx] = RRT_Graph.distances[new_idx] + dist


        end_distance = distance(randpos,RRT_Graph.endposition)
        if end_distance < threshold:
            end_index = RRT_Graph.add_vertex(RRT_Graph.endposition)
            RRT_Graph.add_edge(new_idx,end_index,end_distance)
            try:
                RRT_Graph.distances[end_index] = min(RRT_Graph.distances[end_index], RRT_Graph.distances[new_idx]+end_distance)
            except:
                RRT_Graph.distances[end_index] = RRT_Graph.distances[new_idx]+end_distance

            RRT_Graph.found_path = True
    return RRT_Graph

# Informed RRT* algorithm
def iRRT_s(startposition, endposition, obstacles, iterations, threshold, rand_radius, bias, obstacle_bias, stepsize, goal):
    RRT_Graph = Graph(startposition, endposition)
    gamma = 10*stepsize*pow((1+1/3),(1/3))
    optimal = distance(startposition,endposition)
    if goal is not None: iterations = 5000
    for i in range(iterations):
        randpos = RRT_Graph.randpos(obstacles, rand_radius, bias, obstacle_bias)

        if RRT_Graph.found_path:
            found_path = dijkstra(RRT_Graph)
            length = 0.
            for path in range(len(found_path)-1):
                length = length + distance((found_path[path][0],found_path[path][1],found_path[path][2]),(found_path[path+1][0],found_path[path+1][1],found_path[path+1][2]))

            if goal is not None and (length/optimal) <= goal:
                break

            if not in_ellipsoid(startposition, endposition, randpos, length):
                continue



        if in_obstacle(obstacles, randpos):
            continue

        near_vex, near_idx = nearest_node(RRT_Graph, randpos, obstacles)
        if near_vex is None:
            continue

        if stepsize is not None:
            randpos = reshape(randpos, near_vex, stepsize)

        new_idx = RRT_Graph.add_vertex(randpos)
        dist = distance(near_vex,randpos)
        RRT_Graph.add_edge(new_idx,near_idx,dist)
        RRT_Graph.distances[new_idx] = RRT_Graph.distances[near_idx] + dist #update the distances

        n = len(RRT_Graph.vertices)
        # update nearby vertices distance (if shorter)
        for vertex in RRT_Graph.vertices:
            if vertex == randpos:
                continue

            dist = distance(vertex, randpos)
            if dist > gamma*pow((np.log(n)/n),1/3):
                continue

            line = Line(vertex, randpos)
            if trough_obstacle(obstacles, line):
                continue

            idx = RRT_Graph.indices[vertex]
            if (RRT_Graph.distances[new_idx] + dist) < (RRT_Graph.distances[idx]):
                RRT_Graph.add_edge(idx, new_idx, dist)
                RRT_Graph.distances[idx] = RRT_Graph.distances[new_idx] + dist


        end_distance = distance(randpos,RRT_Graph.endposition)
        if end_distance < threshold:
            end_index = RRT_Graph.add_vertex(RRT_Graph.endposition)
            RRT_Graph.add_edge(new_idx,end_index,end_distance)
            try:
                RRT_Graph.distances[end_index] = min(RRT_Graph.distances[end_index], RRT_Graph.distances[new_idx]+end_distance)
            except:
                RRT_Graph.distances[end_index] = RRT_Graph.distances[new_idx]+end_distance

            RRT_Graph.found_path = True




    return RRT_Graph

# Plotting function
def plot_graph(Graph, obstacles, startposition, endposition, RRT_time, found_path = None, dijkstra_time = None):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection ='3d')
    vertices = np.asarray(Graph.vertices)

    size_x = max(vertices[:,0:1]) - min(vertices[:,0:1])
    size_y = max(vertices[:,1:2]) - min(vertices[:,1:2])
    size_z = max(vertices[:,2:]) - min(vertices[:,2:])

    vex_x = [x for x, y, z in Graph.vertices]
    vex_y = [y for x, y, z in Graph.vertices]
    vex_z = [z for x, y, z in Graph.vertices]

    for obstacle in range(len(obstacles)):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = obstacles[obstacle][0]+obstacles[obstacle][3]*np.cos(u)*np.sin(v)
        y = obstacles[obstacle][1]+obstacles[obstacle][3]*np.sin(u)*np.sin(v)
        z = obstacles[obstacle][2]+obstacles[obstacle][3]*np.cos(v)
        ax.plot_surface(x, y, z, color="r")

    ax.scatter(vex_x,vex_y,vex_z, s=10, color="b")
    ax.scatter(Graph.startposition[0],Graph.startposition[1],Graph.startposition[2],s=50, color="y")
    ax.scatter(Graph.endposition[0],Graph.endposition[1],Graph.endposition[2],s=50, color="y")

    for edge in Graph.edges:
        x = np.array([Graph.vertices[edge[0]][0],Graph.vertices[edge[1]][0]])
        y = np.array([Graph.vertices[edge[0]][1],Graph.vertices[edge[1]][1]])
        z = np.array([Graph.vertices[edge[0]][2],Graph.vertices[edge[1]][2]])
        ax.plot3D(x, y, z, color="b",linewidth=0.3)

    if found_path is not None:
        length = 0.
        for path in range(len(found_path)-1):
            x = np.array([found_path[path][0],found_path[path+1][0]])
            y = np.array([found_path[path][1],found_path[path+1][1]])
            z = np.array([found_path[path][2],found_path[path+1][2]])
            length = length + distance((found_path[path][0],found_path[path][1],found_path[path][2]),(found_path[path+1][0],found_path[path+1][1],found_path[path+1][2]))
            ax.plot3D(x, y, z, color="r",linewidth=3, alpha=0.5)
        print('A path was found, with length:', round(length,2))
        print('The direct path is of length:', round(distance(startposition,endposition),2))
        print('The RRT generation took: '+str(round(RRT_time,2))+ 's')
        print('The Dijkstra path finding took: '+str(dijkstra_time)+'s')

    if found_path is None:
        print('No path was found.')
        print('The RRT generation took: '+str(round(RRT_time,2))+ 's')

    ax.set_box_aspect(aspect = (size_x/size_z,size_y/size_z,1))
    #ax.set_aspect('equal', adjustable='box')
    plt.show()




# calculate intersection between line and obstacle
# still needs some work, only works for spheres
def intersection(obstacle, line):
    r = obstacle[3]
    C = obstacle[:3]
    P = line.p0
    U = line.direction
    Q=P-C
    a = np.dot(U,U)
    b= 2*np.dot(U,Q)
    c = np.dot(Q,Q)-r*r
    d = np.dot(b,b)-4*np.dot(a,c)
    if (line.p0[0] > (obstacle[0]+r) and line.p1[0] > (obstacle[0]+r)) or (line.p0[0] < (obstacle[0]-r) and line.p1[0] < (obstacle[0]-r)) or (line.p0[1] > (obstacle[1]+r) and line.p1[1] > (obstacle[1]+r)) or (line.p0[1] < (obstacle[1]-r) and line.p1[1] < (obstacle[1]-r)) or (line.p0[2] > (obstacle[2]+r) and line.p1[2] > (obstacle[2]+r)) or (line.p0[2] < (obstacle[2]-r) and line.p1[2] < (obstacle[2]-r)):
        return False
    if d < 0:
        return False
    return True

# determine distance between 2 points
def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

# determine whether a vertex (point) is in one of the obstacles
def in_obstacle(obstacles, vertex):
    for obstacle in obstacles:
        if distance(obstacle[:3],vertex) < obstacle[3]:
            return True
    return False

# determine whether a edge (line) goes through one of the obstacles
def trough_obstacle(obstacles,line):
    for obstacle in obstacles:
        if intersection(obstacle, line):
            return True
    return False

#find the nearest node
def nearest_node(Graph, vertex, obstacles):
    near_vex = None
    near_idx = None
    min_distance = 10.e10
    for idx, graph_vertex in enumerate(Graph.vertices):
        line = Line(graph_vertex, vertex)
        if trough_obstacle(obstacles,line):
            continue

        dist = distance(graph_vertex, vertex)
        if dist < min_distance:
            min_distance = dist
            near_idx = idx
            near_vex = graph_vertex
    return near_vex, near_idx

def reshape(randpos, near_vex, stepsize):
    direction = np.array(randpos) - np.array(near_vex)
    length = np.linalg.norm(direction)
    direction = (direction / length) * min (stepsize, length) # find whether stepsize or length is smallest

    shortend_vertex = (near_vex[0]+direction[0], near_vex[1]+direction[1], near_vex[2]+direction[2]) # create a shorter line
    return shortend_vertex

def average(lst):
    try:
        average = sum(lst) / len(lst)
    except:
        average = None
    return average

#define rotated ellipsoid for informed RRT*
def in_ellipsoid(startposition, endposition, pos, path_length):
    startposition = np.array(startposition)
    endposition = np.array(endposition)
    c_min = np.linalg.norm(endposition - startposition)
    c_best = path_length
    a_squared = 0.25 * (np.square(c_best)+np.square(c_min))
    c_squared = np.square(0.5*c_best)
    r = (endposition-startposition)/np.linalg.norm(endposition-startposition)
    p = 0.5 * endposition + startposition
    z_squared = np.square((np.dot((pos-p),r)))
    xy_squared = np.dot((pos-p),(pos-p))-z_squared

    if (xy_squared/a_squared + z_squared/c_squared) < 1:
        return True
    else:
        return False


def dijkstra(Graph):

    # Dijkstra algorithm for finding shortest path from start position to end, from MIT License Copyright (c) 2019 Fanjin Zeng

    srcIdx = Graph.indices[Graph.startposition]
    dstIdx = Graph.indices[Graph.endposition]

    # build dijkstra
    nodes = list(Graph.connections.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in Graph.connections[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(Graph.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(Graph.vertices[curNode])
    return list(path)
