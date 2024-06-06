# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:21:00 2024

@author: jraos
"""
import os
import numpy as np
os.chdir("C:\\Users\\jraos\\OneDrive - Stanford\\Documents\\Stanford\\EC\\Eclectics Blocking Optimization\\eclectics-blocking-optimization")
import computational_geometry as geo

def num_intersections(X,Y,P_inv):
    num_intersections = 0
    for i in range(np.shape(X)[1]-1):
        for j in range(np.shape(X)[0]):
            num_intersections += np.sum(geo.intersect(X[P_inv[j,i],i],Y[P_inv[j,i],i],
                                                      X[P_inv[j,i+1],i+1],Y[P_inv[j,i+1],i+1],
                                                      X[P_inv[j::,i],i],Y[P_inv[j::,i],i],
                                                      X[P_inv[j::,i+1],i+1],Y[P_inv[j::,i+1],i+1]))
    return num_intersections

def max_distance(X,Y,P_inv):
    distance = np.zeros(np.shape(X))
    for i in range(np.shape(X)[1]-1):
        distance[:,i] = np.power(np.power(X[P_inv[:,i+1],i+1]-X[P_inv[:,i],i],2) + np.power(Y[P_inv[:,i+1],i+1]-Y[P_inv[:,i],i],2),0.5)
    return np.max(distance)

def average_distance(X,Y,P_inv):
    distance = np.zeros(np.shape(X))
    for i in range(np.shape(X)[1]-1):
        distance[:,i] = np.power(np.power(X[P_inv[:,i+1],i+1]-X[P_inv[:,i],i],2) + np.power(Y[P_inv[:,i+1],i+1]-Y[P_inv[:,i],i],2),0.5)
    return np.average(distance)