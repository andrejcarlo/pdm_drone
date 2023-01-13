from src.graph import Graph, Line
from src.utils import (
    distance,
    nearest_node,
    reshape,
    dijkstra,
    in_obstacle,
    trough_obstacle,
    in_ellipsoid,
)

import numpy as np
from tqdm import tqdm

# RRT algorithm
def RRT(
    startposition,
    endposition,
    obstacles,
    iterations,
    threshold,
    rand_radius,
    bias,
    obstacle_bias,
    stepsize,
    goal,
    fix_room_size=False,
):
    # initialize graph
    RRT_Graph = Graph(startposition, endposition, fix_room_size=fix_room_size)
    optimal = distance(startposition, endposition)
    # for a length goal, continue for many iterations
    if goal is not None:
        iterations = 10000
    for i in tqdm(range(iterations)):
        # sample a random position
        randpos = RRT_Graph.randpos(obstacles, rand_radius, bias, obstacle_bias)

        # check whether goal length is reached if specified
        if RRT_Graph.found_path and goal is not None:
            found_path = dijkstra(RRT_Graph)
            length = 0.0
            for path in range(len(found_path) - 1):
                length = length + distance(
                    (found_path[path][0], found_path[path][1], found_path[path][2]),
                    (
                        found_path[path + 1][0],
                        found_path[path + 1][1],
                        found_path[path + 1][2],
                    ),
                )
            # stop if goal length is reached
            if goal is not None and (length / optimal) <= goal:
                break
        # continue if the point is in an obstacle
        if in_obstacle(obstacles, randpos):
            continue
        # find the nearest vertex
        near_vex, near_idx = nearest_node(RRT_Graph, randpos, obstacles)
        if near_vex is None:
            continue
        # resize if stepsize is specified
        if stepsize is not None:
            randpos = reshape(randpos, near_vex, stepsize)
            if in_obstacle(obstacles, randpos):
                continue
        # add vertices and edges to graph
        new_idx = RRT_Graph.add_vertex(randpos)
        dist = distance(near_vex, randpos)
        RRT_Graph.add_edge(new_idx, near_idx, dist)

        end_distance = distance(randpos, RRT_Graph.endposition)
        # if a path is found within threshold value, mark path as found
        if end_distance < threshold:
            end_index = RRT_Graph.add_vertex(RRT_Graph.endposition)
            RRT_Graph.add_edge(new_idx, end_index, end_distance)
            RRT_Graph.found_path = True
    return RRT_Graph


# RRT* algorithm
def RRT_s(
    startposition,
    endposition,
    obstacles,
    iterations,
    threshold,
    rand_radius,
    bias,
    obstacle_bias,
    stepsize,
    goal,
    fix_room_size=False,
):
    # initialize graph
    RRT_Graph = Graph(startposition, endposition, fix_room_size=fix_room_size)
    # determine gamma factor for rewiring radius
    gamma = 10 * stepsize * pow((1 + 1 / 3), (1 / 3))
    optimal = distance(startposition, endposition)
    # for a length goal, continue for many iterations
    if goal is not None:
        iterations = 5000
    for i in tqdm(range(iterations)):
        # sample a random position
        randpos = RRT_Graph.randpos(obstacles, rand_radius, bias, obstacle_bias)

        # check whether goal length is reached if specified
        if RRT_Graph.found_path and goal is not None:
            found_path = dijkstra(RRT_Graph)
            length = 0.0
            for path in range(len(found_path) - 1):
                length = length + distance(
                    (found_path[path][0], found_path[path][1], found_path[path][2]),
                    (
                        found_path[path + 1][0],
                        found_path[path + 1][1],
                        found_path[path + 1][2],
                    ),
                )
            # stop if goal length is reached
            if goal is not None and (length / optimal) <= goal:
                break
        # continue if the point is in an obstacle
        if in_obstacle(obstacles, randpos):
            continue
        # find the nearest vertex
        near_vex, near_idx = nearest_node(RRT_Graph, randpos, obstacles)
        if near_vex is None:
            continue
        # resize if stepsize is specified
        if stepsize is not None:
            randpos = reshape(randpos, near_vex, stepsize)
            if in_obstacle(obstacles, randpos):
                continue
        # add vertices and edges to graph
        new_idx = RRT_Graph.add_vertex(randpos)
        dist = distance(near_vex, randpos)
        RRT_Graph.add_edge(new_idx, near_idx, dist)
        RRT_Graph.distances[new_idx] = (
            RRT_Graph.distances[near_idx] + dist
        )  # update the distances
        n = len(RRT_Graph.vertices)

        # update nearby vertices distance (if shorter)
        for vertex in RRT_Graph.vertices:
            if vertex == randpos:
                continue
            # check if node is within radius. The radius reduces as more nodes are sampled
            dist = distance(vertex, randpos)
            if dist > gamma * pow((np.log(n) / n), 1 / 3):
                continue
            # check whether rewiring does not go through obstacle
            line = Line(vertex, randpos)
            if trough_obstacle(obstacles, line):
                continue
            # rewire if distance is shorter
            idx = RRT_Graph.indices[vertex]
            if (RRT_Graph.distances[new_idx] + dist) < (RRT_Graph.distances[idx]):
                RRT_Graph.add_edge(idx, new_idx, dist)
                RRT_Graph.distances[idx] = RRT_Graph.distances[new_idx] + dist

        end_distance = distance(randpos, RRT_Graph.endposition)
        # if a path is found within threshold value, mark path as found
        if end_distance < threshold:
            end_index = RRT_Graph.add_vertex(RRT_Graph.endposition)
            RRT_Graph.add_edge(new_idx, end_index, end_distance)
            try:
                RRT_Graph.distances[end_index] = min(
                    RRT_Graph.distances[end_index],
                    RRT_Graph.distances[new_idx] + end_distance,
                )
            except:
                RRT_Graph.distances[end_index] = (
                    RRT_Graph.distances[new_idx] + end_distance
                )

            RRT_Graph.found_path = True
    return RRT_Graph


# Informed RRT* algorithm
def iRRT_s(
    startposition,
    endposition,
    obstacles,
    iterations,
    threshold,
    rand_radius,
    bias,
    obstacle_bias,
    stepsize,
    goal,
    fix_room_size=False,
):
    # initialize graph
    RRT_Graph = Graph(startposition, endposition, fix_room_size=fix_room_size)
    # determine gamma factor for rewiring radius
    gamma = 10 * stepsize * pow((1 + 1 / 3), (1 / 3))
    optimal = distance(startposition, endposition)
    # for a length goal, continue for many (infinite) iterations
    if goal is not None:
        iterations = 5000
    for i in tqdm(range(iterations)):
        # sample a random position
        randpos = RRT_Graph.randpos(obstacles, rand_radius, bias, obstacle_bias)

        # check what the shortest path length is
        if RRT_Graph.found_path:
            found_path = dijkstra(RRT_Graph)
            length = 0.0
            for path in range(len(found_path) - 1):
                length = length + distance(
                    (found_path[path][0], found_path[path][1], found_path[path][2]),
                    (
                        found_path[path + 1][0],
                        found_path[path + 1][1],
                        found_path[path + 1][2],
                    ),
                )
            # stop if goal length is reached when specified
            if goal is not None and (length / optimal) <= goal:
                break
            # ensure that the newly sampled point is within the ellipsoid as defined for informed RRT*
            if not in_ellipsoid(startposition, endposition, randpos, length):
                continue
        # continue if the point is in an obstacle
        if in_obstacle(obstacles, randpos):
            continue
        # find the nearest vertex
        near_vex, near_idx = nearest_node(RRT_Graph, randpos, obstacles)
        if near_vex is None:
            continue
        # resize if stepsize is specified
        if stepsize is not None:
            randpos = reshape(randpos, near_vex, stepsize)
            if in_obstacle(obstacles, randpos):
                continue
        # add vertices and edges to graph
        new_idx = RRT_Graph.add_vertex(randpos)
        dist = distance(near_vex, randpos)
        RRT_Graph.add_edge(new_idx, near_idx, dist)
        RRT_Graph.distances[new_idx] = (
            RRT_Graph.distances[near_idx] + dist
        )  # update the distances

        n = len(RRT_Graph.vertices)
        # update nearby vertices distance (if shorter)
        for vertex in RRT_Graph.vertices:
            if vertex == randpos:
                continue
            # check if node is within radius. The radius reduces as more nodes are sampled
            dist = distance(vertex, randpos)
            if dist > gamma * pow((np.log(n) / n), 1 / 3):
                continue
            # check whether rewiring does not go through obstacle
            line = Line(vertex, randpos)
            if trough_obstacle(obstacles, line):
                continue
            # rewire if distance is shorter
            idx = RRT_Graph.indices[vertex]
            if (RRT_Graph.distances[new_idx] + dist) < (RRT_Graph.distances[idx]):
                RRT_Graph.add_edge(idx, new_idx, dist)
                RRT_Graph.distances[idx] = RRT_Graph.distances[new_idx] + dist
        # if a path is found within threshold value of end goal, mark path as found
        end_distance = distance(randpos, RRT_Graph.endposition)
        if end_distance < threshold:
            end_index = RRT_Graph.add_vertex(RRT_Graph.endposition)
            RRT_Graph.add_edge(new_idx, end_index, end_distance)
            try:
                RRT_Graph.distances[end_index] = min(
                    RRT_Graph.distances[end_index],
                    RRT_Graph.distances[new_idx] + end_distance,
                )
            except:
                RRT_Graph.distances[end_index] = (
                    RRT_Graph.distances[new_idx] + end_distance
                )

            RRT_Graph.found_path = True

    return RRT_Graph
