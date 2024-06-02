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

num_dancers = 5
num_formations = 2

sf1 = 1  # weighting of the intersection
sf2 = 1  # weighting of the maximum distance
sf3 = 1  # weighting to the average distance

# number of maximum iterations to go over one permutation
max_iteration = 3

# number of repeats to do
num_repeats = 3

# a variable to keep track of if we've passed all tests
tests_passed = 0

np.random.seed(42)  # set the seed that we will use so the tests are repeatable

# test 1 ----------------------------------------------------------------------
X = np.transpose(np.array([[-2,-1,0,1,2], [-2,-1,0,1,2]]))
Y = np.transpose(np.array([[-1,-1,-1,-1,-1], [0,0,0,0,0]]))
C = np.transpose(np.array([[-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1]]))
P_init = np.full((num_dancers, num_formations), 0, dtype=np.int8)
P_init[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

M_expected = np.array([-1,-1,-1,-1,-1])
P_expected = np.transpose(np.array([[0,1,2,3,4], [0,1,2,3,4]]))

try:
    for i in range(num_repeats):
        # P_init[:,1] = np.random.permutation(num_dancers)
        P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P_init, max_iteration, sf1, sf2, sf3)
        assert(np.all(np.equal(M,M_expected)))
        assert(np.all(np.equal(P,P_expected)))
    print("test 1 (No constraints in f0 or f1) with", num_repeats,"repeats passed!")
    tests_passed += 1
except:
    print("test 1 failed")
    
# test 2 ----------------------------------------------------------------------
X = np.transpose(np.array([[-2,-1,0,1,2], [-2,-1,0,1,2]]))
Y = np.transpose(np.array([[-1,-1,-1,-1,-1], [0,0,0,0,0]]))
C = np.transpose(np.array([[2,-1,-1,-1,-1], [-1,-1,-1,-1,-1]]))
P_init = np.full((num_dancers, num_formations), 0, dtype=np.int8)
P_init[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

M_expected = np.array([2,-1,-1,-1,-1])
P_expected = np.transpose(np.array([[0,1,2,3,4], [0,1,2,3,4]]))

try:
    for i in range(num_repeats):
        # P_init[:,1] = np.random.permutation(num_dancers)
        P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P_init, max_iteration, sf1, sf2, sf3)
        assert(np.all(np.equal(M,M_expected)))
        assert(np.all(np.equal(P,P_expected)))
    print("test 2 (Single constraint in f0; no constraints in f1) with", num_repeats,"repeats passed!")
    tests_passed += 1
except:
    print("test 2 failed")
    
# test 3 ----------------------------------------------------------------------
X = np.transpose(np.array([[-2,-1,0,1,2], [-2,-1,0,1,2]]))
Y = np.transpose(np.array([[-1,-1,-1,-1,-1], [0,0,0,0,0]]))
C = np.transpose(np.array([[2,-1,-1,-1,-1], [-1,4,-1,-1,-1]]))
P_init = np.full((num_dancers, num_formations), 0, dtype=np.int8)
P_init[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

M_expected = np.array([2,4,-1,-1,-1])
P_expected = np.transpose(np.array([[0,1,2,3,4], [0,1,2,3,4]]))

try:
    for i in range(num_repeats):
        # P_init[:,1] = np.random.permutation(num_dancers)
        P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P_init, max_iteration, sf1, sf2, sf3)
        assert(np.all(np.equal(M,M_expected)))
        assert(np.all(np.equal(P,P_expected)))
    print("test 3 (Single constraint in f0, single constraint in f1 (different dancerID, different coordinateID)) with", num_repeats,"repeats passed!")
    tests_passed += 1
except:
    print("test 3 failed")
    
# test 4 ----------------------------------------------------------------------
X = np.transpose(np.array([[-2,-1,0,1,2], [-2,-1,0,1,2]]))
Y = np.transpose(np.array([[-1,-1,-1,-1,-1], [0,0,0,0,0]]))
C = np.transpose(np.array([[1,-1,-1,-1,-1], [3,-1,-1,-1,-1]]))
P_init = np.full((num_dancers, num_formations), 0, dtype=np.int8)
P_init[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

M_expected = np.array([1,3,-1,-1,-1])
P_expected = np.transpose(np.array([[0,1,2,3,4], [1,0,2,3,4]]))

try:
    for i in range(num_repeats):
        # P_init[:,1] = np.random.permutation(num_dancers)
        P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P_init, max_iteration, sf1, sf2, sf3)
        assert(np.all(np.equal(M,M_expected)))
        assert(np.all(np.equal(P,P_expected)))
    print("test 4 (Single constraint in f0, single constraint in f1 (different dancerID, same coordinateID)) with", num_repeats,"repeats passed!")
    tests_passed += 1
except:
    print("test 4 failed")

# test 5 ----------------------------------------------------------------------
X = np.transpose(np.array([[-2,-1,0,1,2], [-2,-1,0,1,2]]))
Y = np.transpose(np.array([[-1,-1,-1,-1,-1], [0,0,0,0,0]]))
C = np.transpose(np.array([[2,-1,-1,-1,-1], [-1,-1,-1,2,-1]]))
P_init = np.full((num_dancers, num_formations), 0, dtype=np.int8)
P_init[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

M_expected = np.array([2,-1,-1,-1,-1])
P_expected = np.transpose(np.array([[0,1,2,3,4], [1,2,3,0,4]]))

try:
    for i in range(num_repeats):
        # P_init[:,1] = np.random.permutation(num_dancers)
        P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P_init, max_iteration, sf1, sf2, sf3)
        assert(np.all(np.equal(M,M_expected)))
        assert(np.all(np.equal(P,P_expected)))
    print("test 5 (Single constraint in f0, single constraint in f1 (same dancerID, different coordinateID)) with", num_repeats,"repeats passed!")
    tests_passed += 1
except:
    print("test 5 failed")
   
# test 6 ----------------------------------------------------------------------
X = np.transpose(np.array([[-2,-1,0,1,2], [-2,-1,0,1,2]]))
Y = np.transpose(np.array([[-1,-1,-1,-1,-1], [0,0,0,0,0]]))
C = np.transpose(np.array([[2,-1,-1,-1,-1], [2,-1,-1,-1,-1]]))
P_init = np.full((num_dancers, num_formations), 0, dtype=np.int8)
P_init[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

M_expected = np.array([2,-1,-1,-1,-1])
P_expected = np.transpose(np.array([[0,1,2,3,4], [0,1,2,3,4]]))

try:
    for i in range(num_repeats):
        # P_init[:,1] = np.random.permutation(num_dancers)
        P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P_init, max_iteration, sf1, sf2, sf3)
        assert(np.all(np.equal(M,M_expected)))
        assert(np.all(np.equal(P,P_expected)))
    print("test 6 (Single constraint in f0, single constraint in f1 (same dancerID, same coordinateID) with", num_repeats,"repeats passed!")
    tests_passed += 1
except:
    print("test 6 failed")
    
# test 7 ----------------------------------------------------------------------
X = np.transpose(np.array([[-2,-1,0,1,2], [-2,-1,0,1,2]]))
Y = np.transpose(np.array([[-1,-1,-1,-1,-1], [0,0,0,0,0]]))
C = np.transpose(np.array([[2,4,-1,-1,-1], [-1,3,-1,2,-1]]))
P_init = np.full((num_dancers, num_formations), 0, dtype=np.int8)
P_init[:,] = np.linspace(0,num_dancers-1,num_dancers)[...,None]

M_expected = np.array([2,4,3,-1,-1])
P_expected = np.transpose(np.array([[0,1,2,3,4], [1,2,3,0,4]]))

try:
    for i in range(num_repeats):
        # P_init[:,1] = np.random.permutation(num_dancers)
        P, P_inv, M, M_inv = optimize.optimize_formation(X, Y, C, P_init, max_iteration, sf1, sf2, sf3)
        assert(np.all(np.equal(M,M_expected)))
        assert(np.all(np.equal(P,P_expected)))
    print("test 7 (Mixed constraints (d0, d1 constrained in f0, d1, d2 constrained in f1) with", num_repeats,"repeats passed!")
    tests_passed += 1
except:
    print("test 7 failed")
    
# =============================================================================
# DISPLAY FINAL STATUS
if tests_passed == 7:
    print("all tests successful!")