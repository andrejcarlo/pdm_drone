# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 21:26:12 2022

@author: Gebra
"""
import math
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class Line():
  ''' Define line '''
  def __init__(self, p0, p1):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn = self.dirn/self.dist # normalize

  def path(self, t):
        return self.p + t * self.dirn

def intersection(obstacle, line):
    r = obstacle[3]
    C = obstacle[:3]
    P = line.p0
    U = line.dirn
    Q=P-C
    a = np.dot(U,U)
    b= 2*np.dot(U,Q)
    c = np.dot(Q,Q)
    d = np.dot(b,b)-4*np.dot(a,c)
    #if (line.p0[0] and line.p1[0] > (obstacle[0]+r)) or (line.p0[0] and line.p1[0] < (obstacle[0]-r)) or (line.p0[1] and line.p1[1] > (obstacle[1]+r)) or (line.p0[1] and line.p1[1] < (obstacle[1]-r)) or (line.p0[2] and line.p1[2] > (obstacle[2]+r)) or (line.p0[2] and line.p1[2] < (obstacle[2]-r)):
     #   return False
    if d < 0:
        return False
    return True
fig = plt.figure()
ax = plt.axes(projection ='3d')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
obstacles = [(1.,1.,1.,2.),(3.,4.,5.,1.),(4,2,3,1)]
lines = [(0.,0.,0.),(1.,1.,1.),(5.,5.,5.),(3.,3.,3.)]
# defining coordinates for the 2 points.
x = np.array([1, 1])
y = np.array([1, 1])
z = np.array([-2, 5])
l = Line((1,1,-2),(1,1,5))
print(intersection((1.,1.,1.,2.),l))

# plotting
ax.plot3D(x, y, z)
for obstacle in range(len(obstacles)):
    x = obstacles[obstacle][0]+obstacles[obstacle][3]*np.cos(u)*np.sin(v)
    y = obstacles[obstacle][1]+obstacles[obstacle][3]*np.sin(u)*np.sin(v)
    z = obstacles[obstacle][2]+obstacles[obstacle][3]*np.cos(v)
    ax.plot_surface(x, y, z, color="r")

plt.show()
#for line in range(len(lines)):
 #   ax.plot(lines[line][0],lines[line][1],lines[line][2])    
