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

    startposition = (10., 10., 3.)
    endposition = (55., 55., 3.)

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

    basic_cube = ([0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 'cube')

    map2 = []

    # print([[basic_cube[0][0]+x, basic_cube[0][1]+x, basic_cube[0][2]] for x in np.linspace(0, 10, 1)])
    for i in np.arange(0,50,15):
        for j in np.arange(0,50,15):

            map2.append(([basic_cube[0][0]+i, basic_cube[0][1]+j, basic_cube[0][2]], 
                    [basic_cube[1][0]+i, basic_cube[1][1]+j, basic_cube[1][2]], 'cube'))

    # print(map2)

    obstacles = map2

    iterations = 500
    threshold = 1.  # for marking the end position as found
    stepsize = 5.  # stepsize of newly generated vertices

    # some parameters for the obstacle bias attempt
    obstacle_bias = True
    bias = 0.9
    rand_radius = 5

    # path length/optimal length
    goal = None

    # UNCOMMENT TO VIEW MAP
    plot_obstacle_map(obstacles, startposition, endposition, set_limits=True)

    use_prm = False
    use_rrt = True

    if use_prm:

        # UNCOMMENT TO RUN PLANNER
        start_prm = time.time() #record start time
        prm_planner = PRM(5000, obstacles= obstacles, start= startposition, destination = endposition)
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
                    goal= goal,
                    fix_room_size=False)

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

