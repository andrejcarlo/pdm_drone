from src.graph import Graph, Line
from src.utils import distance, nearest_node, reshape, dijkstra, in_obstacle, trough_obstacle, in_ellipsoid

import numpy as np

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