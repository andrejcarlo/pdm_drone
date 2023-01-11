import numpy as np
from random import random, uniform, randrange

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
    def __init__(self, startposition, endposition, fix_room_size=True):
        self.startposition = startposition
        self.endposition = endposition
        self.found_path = False
        self.fixed_room_size = fix_room_size
        # size of searchbox
        if fix_room_size:
            self.searchboxsize_x = 20. #(startposition[0] - endposition[0])
            self.searchboxsize_y = 10. #(startposition[1] - endposition[1])
            self.searchboxsize_z = 10. #(startposition[2] - endposition[2])
            # location of seachbox startpoint: two times size of box between start and end position
            self.searchbegin_x = 0.
            self.searchbegin_y = 0.
            self.searchbegin_z = 0.
        else:
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
    def randpos(self, obstacles = None, rand_radius = None, bias = None, obstacle_bias = False) -> tuple:
        fix_room_factor = (1 if self.fixed_room_size else -1)
        if obstacle_bias == False:
            # create random values between 0 and 1
            x = random()
            y = random()
            z = random()
            # convert to value within searchbox
            posx = self.searchbegin_x + fix_room_factor*x*self.searchboxsize_x*2
            posy = self.searchbegin_y + fix_room_factor*y*self.searchboxsize_y*2
            posz = self.searchbegin_z + fix_room_factor*z*self.searchboxsize_z*2
            
        else:
            select = random()
            if select > bias:
                # create random values between 0 and 1
                x = random()
                y = random()
                z = random()
                # convert to value within searchbox
                posx = self.searchbegin_x + fix_room_factor*x*self.searchboxsize_x*2
                posy = self.searchbegin_y + fix_room_factor*y*self.searchboxsize_y*2
                posz = self.searchbegin_z + fix_room_factor*z*self.searchboxsize_z*2
            else:
                rand_obs = randrange(0,len(obstacles))
                direction = np.array((uniform(-1,1), uniform(-1,1), uniform(-1,1)))
                length = np.linalg.norm(direction)
                if obstacles[rand_obs][-1] == 'sphere':
                    rand_pos = (direction/length)*(obstacles[rand_obs][3] + uniform(0.,rand_radius))
                    posx = obstacles[rand_obs][0]+ rand_pos[0]
                    posy = obstacles[rand_obs][1]+ rand_pos[1]
                    posz = obstacles[rand_obs][2]+ rand_pos[2]
                if obstacles[rand_obs][-1] == 'cube':
                    # resample around cube 
                    # get center of cube and sample either in x,y,z based on cube width, height, depth
                    obs = obstacles[rand_obs]
                    center = np.median(np.array([obs[0], obs[1]]), axis=0)
                    direction = np.array((uniform(-1,1), uniform(-1,1), uniform(-1,1)))
                    length = np.linalg.norm(direction)
                    dimensions_cube = np.array(obs[1]) - np.array(obs[0])
                    # center base_point, pick a direction, mulitply with width/2, height/2, depth/2 and add offset rand_radius
                    rand_pos = center + np.multiply((direction/length),dimensions_cube/2) + uniform(0.,rand_radius)
                    posx = rand_pos[0]
                    posy = rand_pos[1]
                    posz = rand_pos[2]
                    
        return posx, posy, posz

    def add_vertex(self, vertex):
        vertex = tuple(vertex)
        try: # check if vertex already exists
            idx = self.indices[vertex]
        except:# otherwise add to list and dictionaries
            idx = len(self.vertices)
            self.vertices.append(vertex)
            self.indices[vertex] = idx
            self.connections[idx] = []
        return idx

    def add_edge(self, start_node, end_node, cost):
        self.edges.append((start_node, end_node)) # add edge based on indices
        self.connections[start_node].append((end_node, cost)) #add the connecting nodes and costs
        self.connections[end_node].append((start_node, cost))
