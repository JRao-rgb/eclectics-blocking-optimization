# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:31:11 2024

@author: jraos
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 30 06:01:49 2024

@author: jraos
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
os.chdir("C:\\Users\\jraos\\OneDrive - Stanford\\Documents\\Stanford\\EC\\Eclectics Blocking Optimization\\eclectics-blocking-optimization")
import formations
import computational_geometry as geo

num_dancers = 50
num_formations = 2

start_time = time.time()
np.random.seed(42) # set the seed that we will use so the tests are repeatable

X = np.zeros((num_dancers,num_formations),dtype=np.float16)
Y = np.zeros((num_dancers,num_formations),dtype=np.float16)

max_iterations = 3 # number of maximum iterations to go over one permutation

# begin the optimization procedure. Let'd o one optimmization procedure first.
# let's start by defining the variables we need for this.

P = np.full((num_dancers,num_formations),0,dtype=np.int8) # permutation array. The entire point
# of the algorithm is to figure out what numbers to put in here
C = np.full((num_dancers,num_formations),-1,dtype=np.int8) # constraint array. This array
# specifies which dancer assignments are multable and immutable. If -1, it means
# there is no constraint for the circle at that x[i,j], y[i,j]
M = np.full(num_dancers, np.NAN) # mapping array that goes from physical location mappings
# in the P array to actual dancer numbers specified in the C array

# ======================= suitable break point ============================

X[:,0], Y[:,0] = formations.ring(num_dancers = num_dancers, radius = 1)
X[:,1], Y[:,1] = formations.ring(num_dancers = num_dancers, radius = 2, offset = [4,4])
# X[:,2], Y[:,2] = formations.ring(num_dancers = num_dancers)

# pre-allocate space for our arrays
base_array = np.ones((num_dancers,2)) # something to help with vectorizing
positions_initial = np.ones((num_dancers,2))
positions_final   = np.ones((num_dancers,2))

# initialize the permutation matrix
P[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]
P[:,1] = np.random.permutation(num_dancers)
# P[:,1] = [5,3,4,2,1,0]
P_ = np.copy(P)

# loop over formations-1
for formation in range(num_formations-1):
    i = formation
    # initialize a permutation for each formation based on Euclidean distance
    idx_available = np.ones((1,num_dancers))
    for dancer in range(num_dancers):
        Euclidean_distance = (np.power(X[dancer,i]-np.multiply(X[:,i+1],idx_available),2) +\
                              np.power(Y[dancer,i]-np.multiply(Y[:,i+1],idx_available),2))
        idx = np.nanargmin(Euclidean_distance)
        P[dancer,i+1] = idx
        idx_available[:,idx] = np.array([np.nan])
    # loop over iterations per formation
    positions_initial[:,0] = X[:,i]
    positions_initial[:,1] = Y[:,i]
    positions_final[:,0]   = X[:,i+1]
    positions_final[:,1]   = Y[:,i+1]
    # now for the actual optimization steps
    for z in range(max_iterations): # replace with max_iterations
        # loop through each initial position
        current_loop_permutation = np.random.permutation(num_dancers)
        for y in range(num_dancers): # replace with num_dancers
            # for each dancer, make a switch in the path assignmment with the path of
            # another dancer.
            j = current_loop_permutation[y]
            ideal_switch_index = j
            current_intersection_record = 2000 # some ridiculous large number
            for x in range(y+1,num_dancers): # replace with num_dancers - 1
                # now we are at the meat of it. We will count the number of intersections
                # in the path of dancer j and dancer k in the current case. Then,
                # we will swap the destinations of dancer j and dancer k and see
                # if that results in less intersections. If it does, we keep the
                # swap, and this is the new target number of intersections to shoot
                # for
                k = current_loop_permutation[x]
                p1_j = np.multiply(positions_initial[j,:],base_array) # the starting position of path for dancer j
                p1_k = np.multiply(positions_initial[k,:],base_array) # the starting position of path for dancer k
                p2_j_original = np.multiply(positions_final[P[j,i+1],:],base_array) # the original destination of a path for dancer j
                p2_k_original = np.multiply(positions_final[P[k,i+1],:],base_array) # the original destination of a path for dancer k
                p2_j_swapped  = np.multiply(positions_final[P[k,i+1],:],base_array) # the swapped  destination of a path for dancer j
                p2_k_swapped  = np.multiply(positions_final[P[j,i+1],:],base_array) # the swapped  destination of a path for dancer k
                
                # now, for each of these two paths, we will count the number of 
                # intersections they each have with all other paths
                # check if the intersection exists
                intersection_original_j = np.sum(geo.intersect2(p1_j,p2_j_original,positions_initial,positions_final[P[:,i+1],:]))
                intersection_swapped_j  = np.sum(geo.intersect2(p1_j,p2_j_swapped,positions_initial,positions_final[P[:,i+1],:]))
                intersection_original_k = np.sum(geo.intersect2(p1_k,p2_k_original,positions_initial,positions_final[P[:,i+1],:]))
                intersection_swapped_k  = np.sum(geo.intersect2(p1_k,p2_k_swapped,positions_initial,positions_final[P[:,i+1],:]))
                     
                # if the intersection calculations turn out to be favorable...
                if intersection_original_j + intersection_original_k > intersection_swapped_j + intersection_swapped_k \
                    and intersection_swapped_j + intersection_swapped_k < current_intersection_record:
                    ideal_switch_index = k
                    current_intersection_record = intersection_swapped_j + intersection_swapped_k
                    # print("new record",current_intersection_record,"index",k)
                
            # now, we make the switch
            # print(ideal_switch_index, end = ' ')
            P[j,i+1], P[ideal_switch_index,i+1] = P[ideal_switch_index,i+1], P[j,i+1]
            
geo.plot_movement(X[:,0], X[:,1], Y[:,0], Y[:,1], P[:,0], P[:,1])
end_time = time.time()
print("original code w/ multiple formations: ",end_time-start_time)