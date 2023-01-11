import math
import time
from random import random, randrange, uniform, choice

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from src.prm import PRM
from src.visualisation import plot_graph, plot_obstacle_map

if __name__ == '__main__':
       
    startposition = (-3.,-2.,-3.)
    endposition = (10.,5.,5.)

    # DEFINE MAPS HERE
    obstacles = [(5.,2.5,2.5,3, 'sphere'), ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 'cube'), ([2.0, 0.0, 0.0], [4.0, 4.0, 3.0], 'cube')]
    
    # min, max, 'cube'
    # x, y, z, radius

    # RRT, RRT Star, Biased RRT*

    # obstacles = [(1.,1.,1.,.5),(3.,4.,5.,1.),(4,2,3,1),(6,3,1,2),(6,1,1,2),(8,1,4,1)]
    # obstacles = [(5,-1,-1,1.5),(5,1,-1,1.5),(5,3,-1,1.5),(5,5,-1,1.5),(5,-1,1,1.5),(5,1,1,1.5),(5,3,1,1.5),(5,5,1,1.5),(5,-1,3,1.5),(5,3.5,3,1.5),(5,0,3,1.5),(5,5,3,1.5),(5,-1,5,1.5),(5,1,5,1.5),(5,3,5,1.5),(5,5,5,1.5)] #hole
    # obstacles = [(5,2,3,0.5),(5,2,2,0.5),(5,3,2,0.5),(5,3,3,0.5),
                # (5,1,2.5,1),(5,4,2.5,1),(5,2.5,1,1),(5,2.5,4,1),(5,1,1,1),(5,1,4,1),(5,4,1,1),(5,4,4,1)]
    
    iterations = 1000
    threshold = 2. #for marking the end position as found
    stepsize = 1. # stepsize of newly generated vertices
    
    #some parameters for the obstacle bias attempt
    obstacle_bias = True
    bias = 0.8
    rand_radius = 0.5
    
    #path length/optimal length
    goal = None

    # UNCOMMENT TO VIEW MAP
    plot_obstacle_map(obstacles)


    # UNCOMMENT TO RUN PLANNER
    # start_prm = time.time() #record start time
    # prm_planner = PRM(200, obstacles= obstacles, start= startposition, destination = endposition)
    # prm_planner.runPRM()
    # prm_time = time.time() - start_prm

    # if prm_planner.solutionFound:
    #     print("Found a path!")
    #     print('The vertices of the path are:')
    #     print(prm_planner.found_path)
    #     print()
    #     plot_graph(prm_planner.graph, obstacles, startposition, endposition, prm_time, prm_planner.found_path)
    # else:
    #     print("No path found!")
    #     plot_graph(prm_planner.graph, obstacles, startposition, endposition, prm_time, prm_planner.found_path)




    
    
   