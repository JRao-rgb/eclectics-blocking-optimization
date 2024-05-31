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

num_dancers = 30
num_formations = 3

start_time = time.time()
np.random.seed(42) # set the seed that we will use so the tests are repeatable

X = np.zeros((num_dancers,num_formations),dtype=np.float16)
Y = np.zeros((num_dancers,num_formations),dtype=np.float16)

max_iteration = np.int16(np.round(num_dancers/3)) # number of maximum iterations to go over one permutation

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

X[:,0], Y[:,0] = formations.lines_and_windows(num_dancers = num_dancers)
X[:,1], Y[:,1] = formations.pyramid(num_dancers = num_dancers)
X[:,2], Y[:,2] = formations.ring(num_dancers = num_dancers)

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
        Euclidean_distance = (np.power(X[P[dancer,i],i]-np.multiply(X[:,i+1],idx_available),2) +\
                              np.power(Y[P[dancer,i],i]-np.multiply(Y[:,i+1],idx_available),2))
        idx = np.nanargmin(Euclidean_distance)
        P[dancer,i+1] = idx
        idx_available[:,idx] = np.array([np.nan])
    # loop over iterations per formation
    for iteration in range(max_iteration):
        # randomize the order at which we scan through the dancers here
        current_iter_permutation = np.random.permutation(num_dancers)
        p = current_iter_permutation        
        # loop over each dancer-1 (since need at least 2 dancers to make a switch)
        for dancer1 in range(num_dancers-1): # range(num_dancers-1)
            k = current_iter_permutation[dancer1]
            # loop over the rest of the dancers. Now we are swapping dancer k and l
            k_p1_x = np.full((num_dancers,num_dancers-dancer1-1),X[P[k,i],i])
            k_p1_y = np.full((num_dancers,num_dancers-dancer1-1),Y[P[k,i],i])
            l_p1_x = np.transpose(np.tile(X[P[p[dancer1+1::],i],i][...,None],num_dancers))
            l_p1_y = np.transpose(np.tile(Y[P[p[dancer1+1::],i],i][...,None],num_dancers))
            
            # absolute magic fuckery. Good luck figuring this out in 2 weeks.
            k_p2_x_original = np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            k_p2_y_original = np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            k_p2_x_swapped =  np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            k_p2_y_swapped =  np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            l_p2_x_original = np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            l_p2_y_original = np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            l_p2_x_swapped =  np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            l_p2_y_swapped =  np.full((num_dancers,num_dancers-dancer1-1),0,dtype=np.float16)
            
            for dancer2 in range(dancer1+1,num_dancers):
                l = current_iter_permutation[dancer2]
                k_p2_x_original[:,dancer2-dancer1-1] = np.transpose(np.tile(X[P[k,i+1],i+1],num_dancers))
                k_p2_y_original[:,dancer2-dancer1-1] = np.transpose(np.tile(Y[P[k,i+1],i+1],num_dancers))
                k_p2_x_swapped[:,dancer2-dancer1-1]  = np.transpose(np.tile(X[P[l,i+1],i+1],num_dancers))
                k_p2_y_swapped[:,dancer2-dancer1-1]  = np.transpose(np.tile(Y[P[l,i+1],i+1],num_dancers))
                l_p2_x_original[:,dancer2-dancer1-1] = np.transpose(np.tile(X[P[l,i+1],i+1],num_dancers))
                l_p2_y_original[:,dancer2-dancer1-1] = np.transpose(np.tile(Y[P[l,i+1],i+1],num_dancers))
                l_p2_x_swapped[:,dancer2-dancer1-1]  = np.transpose(np.tile(X[P[k,i+1],i+1],num_dancers))
                l_p2_y_swapped[:,dancer2-dancer1-1]  = np.transpose(np.tile(Y[P[k,i+1],i+1],num_dancers))
                                
            # performing the intersection calculations
            arr_intersection_original = geo.intersect(k_p1_x, k_p1_y, 
                                                      k_p2_x_original, k_p2_y_original, 
                                                      X[P[:,i],i][...,None], Y[P[:,i],i][...,None], 
                                                      X[P[:,i+1],i+1][...,None], Y[P[:,i+1],i+1][...,None]) + \
                                        geo.intersect(l_p1_x, l_p1_y, 
                                                      l_p2_x_original, l_p2_y_original, 
                                                      X[P[:,i],i][...,None], Y[P[:,i],i][...,None], 
                                                      X[P[:,i+1],i+1][...,None], Y[P[:,i+1],i+1][...,None])
            arr_intersection_swapped  = geo.intersect(k_p1_x, k_p1_y, 
                                                      k_p2_x_swapped, k_p2_y_swapped, 
                                                      X[P[:,i],i][...,None], Y[P[:,i],i][...,None], 
                                                      X[P[:,i+1],i+1][...,None], Y[P[:,i+1],i+1][...,None]) + \
                                        geo.intersect(l_p1_x, l_p1_y, 
                                                      l_p2_x_swapped, l_p2_y_swapped, 
                                                      X[P[:,i],i][...,None], Y[P[:,i],i][...,None], 
                                                      X[P[:,i+1],i+1][...,None], Y[P[:,i+1],i+1][...,None])
                                        
            num_intersection_original = np.sum(arr_intersection_original,axis=0)
            num_intersection_swapped  = np.sum(arr_intersection_swapped,axis=0)
            temp = num_intersection_swapped.copy()
            
            if all(num_intersection_original <= num_intersection_swapped):
                ideal_swap_idx = -1
            else:
                # if np.max(num_intersection_original - num_intersection_swapped) <= 0: continue
                num_intersection_swapped[num_intersection_original <= num_intersection_swapped]=1000
                ideal_swap_idx = np.argmin(num_intersection_swapped)
            
            # print(p[dancer1+ideal_swap_idx+1], end = ' ')
            # geo.plot_movement(X[:,0], X[:,1], Y[:,0], Y[:,1], P[:,0], P[:,1])
            P[k,i+1], P[p[dancer1+ideal_swap_idx+1],i+1] = P[p[dancer1+ideal_swap_idx+1],i+1], P[k,i+1]
            # geo.plot_movement(X[:,0], X[:,1], Y[:,0], Y[:,1], P[:,0], P[:,1])
            
    geo.plot_movement(X[:,i], X[:,i+1], Y[:,i], Y[:,i+1], P[:,i], P[:,i+1])
end_time = time.time()
print("Optimization Completed. Time elapsed: ",end_time-start_time)

#%%
geo.animate_movement(X,Y,P)