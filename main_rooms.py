import math
import time
from random import random, randrange, uniform, choice

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from src.prm import PRM
from src.visualisation import plot_graph, plot_obstacle_map
from src.rrt import iRRT_s
from src.utils import dijkstra

if __name__ == '__main__':
    # startposition = (-3., -2., -3.)
    # endposition = (10., 5., 5.)
    endposition = (19., 0., 0.)
    startposition = (2.5, 8.6, 7.7)

    # DEFINE MAPS HERE
    map0 = [
        (5., 2.5, 2.5, 3, 'sphere'),
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 'cube'),
        ([2.0, 0.0, 0.0], [4.0, 4.0, 3.0], 'cube')
    ]

    map1 = [
        (7.0, 5.5, 5., 1, 'sphere'),  # O3
        (5.0, 10., 5., 2, 'sphere'),  # O4
        ([1.0, 2.0, 1.0], [9.0, 4.0, 9.0], 'cube'),  # O1
        ([11.0, 6.0, 1.0], [17.0, 12.0, 9.0], 'cube'),  # O2
        ([5.0, 14.0, 1.0], [7.5, 14.5, 9.0], 'cube'),  # O5
        ([20.0, 0.0, 0.0], [0.0, 15.0, 0.5], 'cube'),  # bottom_plate
        ([20.0, 0.0, 10.0], [0.0, 15.0, 10.5], 'cube'),  # top_plate
    ]

    map2 = [
        (7.0, 6.5, 5., 1, 'sphere'),  # O3
        (5.0, 10., 5., 2, 'sphere'),  # O4
        (8.5, 8.0, 8., 1.5, 'sphere'),  # O6
        (13.5, 3.5, 3., 2, 'sphere'),  # O7
        (18.5, 2.0, 2., 1.5, 'sphere'),  # O8
        ([9.5, 8.5, 2.0], [11.5, 4.5, 6.5], 'cube'),  # O9
        ([2.0, 3.0, 2.0], [8.0, 5.0, 8.0], 'cube'),  # O1
        ([12.0, 7.0, 2.0], [16.0, 11.0, 8.0], 'cube'),  # O2
        ([6.0, 14.0, 2.0], [8.0, 14.5, 8.0], 'cube'),  # O5
        ([20.0, 0.0, 0.0], [0.0, 15.0, 0.5], 'cube'),  # bottom_plate
        ([20.0, 0.0, 10.0], [0.0, 15.0, 10.5], 'cube'),  # top_plate
    ]


    map3 = [
        (7.0, 6.5, 5., 1, 'sphere'),  # O3
        (10.0, 12.5, 5., 2, 'sphere'),  # O4
        (8.5, 8.0, 8., 1.5, 'sphere'),  # O6
        (13.5, 3.5, 3., 2, 'sphere'),  # O7
        (18.5, 2.0, 2., 1.5, 'sphere'),  # O8
        ([9.5, 8.5, 2.0], [11.5, 4.5, 6.5], 'cube'),  # O9
        ([2.0, 3.0, 2.0], [8.0, 5.0, 8.0], 'cube'),  # O1
        ([12.0, 7.0, 2.0], [16.0, 11.0, 8.0], 'cube'),  # O2
        ([6.0, 14.0, 2.0], [8.0, 14.5, 8.0], 'cube'),  # O5
        ([20.0, 0.0, 0.0], [0.0, 15.0, 0.5], 'cube'),  # bottom_plate
        ([20.0, 0.0, 10.0], [0.0, 15.0, 10.5], 'cube'),  # top_plate
        ([0.5, 10.0, 0.5], [3.5, 14.0, 4.0], 'cube'),  # O11
        ([15.0, 12.0, 2.0], [18.0, 14.5, 9.0], 'cube'),  # O12
        ([9.0, 0.5, 0.5], [11.0, 4.0, 3.5], 'cube'),  # O13
        ([17.0, 4.0, 5.5], [20.0, 7.0, 9.0], 'cube'),  # O14
        ([4.0, 9.0, 2.5], [7.0, 12.0, 6.5], 'cube'),  # O15
    ]

    obstacles = map3
    plot_obstacle_map(obstacles)

    # min, max, 'cube' ( y, x, z)
    # x, y, z, radius

    # RRT, RRT Star, Biased RRT*
    iterations = 2000
    threshold = 4.  # for marking the end position as found
    stepsize = 15.  # stepsize of newly generated vertices

    # some parameters for the obstacle bias attempt
    obstacle_bias = True
    bias = 0.8
    rand_radius = 0.5

    # path length/optimal length
    goal = None

    # UNCOMMENT TO VIEW MAP
    # plot_obstacle_map(obstacles)

    use_prm = False
    use_rrt = True

    if use_prm:

        # UNCOMMENT TO RUN PLANNER
        start_prm = time.time() #record start time
        prm_planner = PRM(200, obstacles= obstacles, start= startposition, destination = endposition)
        prm_planner.runPRM()
        prm_time = time.time() - start_prm

        if prm_planner.solutionFound:
            print("Found a path!")
            print('The vertices of the path are:')
            print(prm_planner.found_path)
            print()
            plot_graph(prm_planner.graph, obstacles, startposition, endposition, prm_time, prm_planner.found_path, visualize_all=True)
        else:
            print("No path found!")
            plot_graph(prm_planner.graph, obstacles, startposition, endposition, prm_time, prm_planner.found_path, visualize_all=True)

    elif use_rrt:

        start_RRT = time.time() #record start time
        # G = rrt_andrei(startposition, endposition, obstacles, iterations, threshold, rand_radius, bias, obstacle_bias, stepsize, goal)
        #G = RRT_s(startposition, endposition, obstacles, iterations, threshold, rand_radius, bias, obstacle_bias, stepsize, goal)
        G = iRRT_s(startposition= startposition,
                    endposition= endposition,
                    obstacles= obstacles,
                    iterations= iterations,
                    threshold= threshold,
                    rand_radius= rand_radius,
                    bias = bias,
                    obstacle_bias= obstacle_bias,
                    stepsize= stepsize,
                    goal= goal)

        end_RRT = time.time()
        RRT_time = end_RRT - start_RRT
        if G.found_path:
            start_dijkstra = time.time()
            path = dijkstra(G)
            end_dijkstra = time.time()
            dijkstra_time = end_dijkstra - start_dijkstra
            print("Found a path!")
            print('The vertices of the path are:')
            print(path)
            print()
            plot_graph(G, obstacles, startposition, endposition, RRT_time, path, dijkstra_time, visualize_all=True)
        else:
            print(f"No path found in {iterations}")
            plot_graph(G, obstacles, startposition, endposition, RRT_time, visualize_all=True)

