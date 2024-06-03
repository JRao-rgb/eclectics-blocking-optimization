# -*- coding: utf-8 -*-
"""
Created on Thu May 30 06:01:49 2024

@author: jraos
"""

import os
os.chdir("C:\\Users\\jraos\\OneDrive - Stanford\\Documents\\Stanford\\EC\\Eclectics Blocking Optimization\\eclectics-blocking-optimization")
import metrics
import optimize
import computational_geometry as geo
import formations
import numpy as np
import matplotlib.pyplot as plt
import time

num_dancers = 28
num_formations = 3

sf1 = 1 # weighting of the intersection
sf2 = 1 # weighting of the maximum distance
sf3 = 4 # weighting to the average distance

start_time = time.time()
np.random.seed(42) # set the seed that we will use so the tests are repeatable

X = np.zeros((num_dancers,num_formations),dtype=np.float16)
Y = np.zeros((num_dancers,num_formations),dtype=np.float16)

max_iteration = np.int16(np.ceil(num_dancers/2)) # number of maximum iterations to go over one permutation

# begin the optimization procedure. Let'd o one optimmization procedure first.
# let's start by defining the variables we need for this.

P = np.full((num_dancers,num_formations),0,dtype=np.int8) # permutation array. The entire point
# of the algorithm is to figure out what numbers to put in here
C = np.full((num_dancers,num_formations),-1,dtype=np.int8) # constraint array. This array
# specifies which dancer assignments are multable and immutable. If -1, it means
# there is no constraint for the circle at that x[i,j], y[i,j]
M = np.full(num_dancers, -1, dtype=np.int8) # mapping array that goes from physical location mappings
# in the P array to actual dancer numbers specified in the C array
# initialize the permutation matrix
P[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

X[:,0], Y[:,0] = formations.ring(num_dancers = num_dancers)
X[:,1], Y[:,1] = formations.ring(num_dancers = num_dancers, radius = 3)
X[:,2], Y[:,2] = formations.ring(num_dancers = num_dancers, radius = 5)

# adding in some constraints
# C[0,0] = 2; C[17,1] = 2;
C[14,1] = 5; C[0,2] = 5;

P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P, max_iteration, sf1, sf2, sf3, display_metrics=True)
geo.plot_movement(X[:,0], X[:,1], Y[:,0], Y[:,1], P_inv[:,0], P_inv[:,1], plot_title="formation 0 to 1",display_numbers=True)
geo.plot_movement(X[:,1], X[:,2], Y[:,1], Y[:,2], P_inv[:,1], P_inv[:,2], plot_title="formation 1 to 2",display_numbers=True)

#%%

geo.animate_movement(
    X, Y, P_inv, filename="6-2-24_animations/r-r-r_benchmark_c7.gif")