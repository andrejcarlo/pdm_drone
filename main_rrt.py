import math
import time
from random import random, randrange, uniform, choice

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from src.utils import dijkstra, distance
from src.graph import Graph, Line
from src.rrt import iRRT_s, RRT, RRT_s
from src.visualisation import plot_graph, plot_graph_simple
from src.prm import rrt_andrei

if __name__ == '__main__':
       
    startposition = (0.,0.,0.)
    endposition = (10.,5.,5.)
    #obstacles = []
    obstacles = [(5.,2.5,2.5,3)]
    #obstacles = [(1.,1.,1.,.5),(3.,4.,5.,1.),(4,2,3,1),(6,3,1,2),(6,1,1,2),(8,1,4,1)]
    #obstacles = [(5,-1,-1,1.5),(5,1,-1,1.5),(5,3,-1,1.5),(5,5,-1,1.5),(5,-1,1,1.5),(5,1,1,1.5),(5,3,1,1.5),(5,5,1,1.5),(5,-1,3,1.5),(5,3.5,3,1.5),(5,0,3,1.5),(5,5,3,1.5),(5,-1,5,1.5),(5,1,5,1.5),(5,3,5,1.5),(5,5,5,1.5)] #hole
    #obstacles = [(5,2,3,0.5),(5,2,2,0.5),(5,3,2,0.5),(5,3,3,0.5),(5,1,2.5,1),(5,4,2.5,1),(5,2.5,1,1),(5,2.5,4,1),(5,1,1,1),(5,1,4,1),(5,4,1,1),(5,4,4,1)]
    iterations = 1000
    threshold = 2. #for marking the end position as found
    stepsize = 1. # stepsize of newly generated vertices
    
    #some parameters for the obstacle bias attempt
    obstacle_bias = True
    bias = 0.8
    rand_radius = 0.5
    
    #path length/optimal length
    goal = None
    
    start_RRT = time.time() #record start time
    G = rrt_andrei(startposition, endposition, obstacles, iterations, threshold, rand_radius, bias, obstacle_bias, stepsize, goal)
    #G = RRT_s(startposition, endposition, obstacles, iterations, threshold, rand_radius, bias, obstacle_bias, stepsize, goal)
    # G = iRRT_s(startposition= startposition,
    #             endposition= endposition,
    #             obstacles= obstacles,
    #             iterations= iterations,
    #             threshold= threshold,
    #             rand_radius= rand_radius,
    #             bias = bias,
    #             obstacle_bias= obstacle_bias,
    #             stepsize= stepsize,
    #             goal= goal)

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
        plot_graph(G, obstacles, startposition, endposition, RRT_time, path, dijkstra_time)
    else:
        print(f"No path found in {iterations}")
        plot_graph(G, obstacles, startposition, endposition, RRT_time)
    
   