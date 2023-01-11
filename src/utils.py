from collections import deque
import numpy as np
from src.graph import Line

# determine distance between 2 points
def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

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

# adapted from intersectCube in https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
# compute the near and far intersections of the cube (stored in the x and y components) using the slab method
# no intersection means vec.x > vec.y (really tNear > tFar)
def intersectAABB(rayOrigin, rayDir, rayEnd, cubeMin, cubeMax):
    """
    rayOrigin, rayDir, cubeMin, cubeMax: vec3
    """
    tMin = (cubeMin - rayOrigin) / rayDir; # vec3
    tMax = (cubeMax - rayOrigin) / rayDir; # vec3
    t1 = np.minimum(tMin, tMax); # vec3
    t2 = np.maximum(tMin, tMax); # vec3
    tNear = max(max(t1[0], t1[1]), t1[2]); # float
    tFar = min(min(t2[0], t2[1]), t2[2]);  # float

    # if tNear < tFar and not ((distance(t1,rayOrigin) > distance(rayOrigin, rayEnd)) or (distance(t2,rayOrigin) > distance(rayOrigin, rayEnd))):
    if tNear < tFar:
      return True
        
    return False

def pointInAABB(vecPoint, cubeMin, cubeMax):

    # Check if the point is less than max and greater than min
    if(vecPoint[0] >= cubeMin[0] and vecPoint[0] <= cubeMax[0] and 
        vecPoint[1] >= cubeMin[1] and vecPoint[1] <= cubeMax[1] and 
        vecPoint[2] >= cubeMin[2] and vecPoint[2] <= cubeMax[2]):

        return True

    # If not, then return false
    return False


# determine whether a vertex (point) is in one of the obstacles
def in_obstacle(obstacles, vertex):
    for obstacle in obstacles:
        if obstacle[-1] == 'sphere':
            if distance(obstacle[:3],vertex) < obstacle[3]:
                return True
        elif obstacle[-1] == 'cube':
            if pointInAABB(vertex, np.array(obstacle[0]), np.array(obstacle[1])):
                return True
    return False

# determine whether a edge (line) goes through one of the obstacles
def trough_obstacle(obstacles,line):
    for obstacle in obstacles:
        if obstacle[-1] == 'sphere':
            if intersection(obstacle, line):
                return True
        elif obstacle[-1] == 'cube':
            line_intersects_cube = intersectAABB(line.p0, line.direction, line.p1, np.array(obstacle[0]), np.array(obstacle[1]))
            if line_intersects_cube:
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