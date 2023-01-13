import math
import time
from random import random, randrange, uniform, choice

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from src.prm import PRM
from src.visualisation import plot_graph, plot_obstacle_map

if __name__ == "__main__":

    startposition = (0.0, 0.0, 0.0)
    endposition = (10.0, 5.0, 5.0)
    # obstacles = []
    obstacles = [(5.0, 2.5, 2.5, 3, "sphere")]
    # obstacles = [(1.,1.,1.,.5),(3.,4.,5.,1.),(4,2,3,1),(6,3,1,2),(6,1,1,2),(8,1,4,1)]
    # obstacles = [(5,-1,-1,1.5),(5,1,-1,1.5),(5,3,-1,1.5),(5,5,-1,1.5),(5,-1,1,1.5),(5,1,1,1.5),(5,3,1,1.5),(5,5,1,1.5),(5,-1,3,1.5),(5,3.5,3,1.5),(5,0,3,1.5),(5,5,3,1.5),(5,-1,5,1.5),(5,1,5,1.5),(5,3,5,1.5),(5,5,5,1.5)] #hole
    # obstacles = [(5,2,3,0.5),(5,2,2,0.5),(5,3,2,0.5),(5,3,3,0.5),(5,1,2.5,1),(5,4,2.5,1),(5,2.5,1,1),(5,2.5,4,1),(5,1,1,1),(5,1,4,1),(5,4,1,1),(5,4,4,1)]

    map_1 = [
        (7.0, 5.5, 5.0, 1, "sphere"),  # O3
        (5.0, 10.0, 5.0, 2, "sphere"),  # O4
        ([1.0, 2.0, 1.0], [9.0, 4.0, 9.0], "cube"),  # O1
        ([11.0, 6.0, 1.0], [17.0, 12.0, 9.0], "cube"),  # O2
        ([5.0, 14.0, 1.0], [7.5, 14.5, 9.0], "cube"),  # O5
        ([20.0, 0.0, 0], [0.0, 15.0, 0.5], "cube"),  # bottom_plate
        ([20.0, 0.0, 10.0], [0.0, 15.0, 10.5], "cube"),  # top_plate
    ]

    obstacles = map_1
    endposition = (19.0, 0.0, 2)
    startposition = (2.5, 8.6, 7.7)

    iterations = 5
    threshold = 2.0  # for marking the end position as found
    stepsize = 1.0  # stepsize of newly generated vertices

    # some parameters for the obstacle bias attempt
    obstacle_bias = True
    bias = 0.8
    rand_radius = 0.5

    # path length/optimal length
    goal = None

    # Run PRM implementation
    start_prm = time.time()
    prm_planner = PRM(
        200,
        obstacles=obstacles,
        start=startposition,
        destination=endposition,
        iterations=iterations,
    )
    prm_planner.runPRM()
    prm_time = time.time() - start_prm

    if prm_planner.solutionFound:
        print("Found a path!")
        print("The vertices of the path are:")
        print(prm_planner.found_path)
    else:
        print("No path found!")

    plot_graph(
        prm_planner.graph,
        obstacles,
        startposition,
        endposition,
        prm_time,
        prm_planner.found_path,
        visualize_all=True,
        set_aspect=False,
    )
