# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:21:00 2024

@author: jraos
"""
import os
import numpy as np
os.chdir("C:\\Users\\jraos\\OneDrive - Stanford\\Documents\\Stanford\\EC\\Eclectics Blocking Optimization\\eclectics-blocking-optimization")
import computational_geometry as geo

def num_intersections(X,Y,P):
    num_intersections = 0
    for i in range(np.shape(X)[1]-1):
        for j in range(np.shape(X)[0]):
            num_intersections += np.sum(geo.intersect(X[P[j,i],i],Y[P[j,i],i],
                                                       X[P[j,i+1],i+1],Y[P[j,i+1],i+1],
                                                       X[P[j::,i],i],Y[P[j::,i],i],
                                                       X[P[j::,i+1],i+1],Y[P[j::,i+1],i+1]))
    return num_intersections

def max_distance(X,Y,P):
    pass

def avg_distance(X,Y,P):
    pass