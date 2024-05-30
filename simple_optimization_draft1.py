# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:36:25 2024

@author: jraos
"""
#%% import statement

import numpy as np
import matplotlib.pyplot as plt

stage_width = 10 # stage width in meters
stage_length = 20 # stage length in meters

#%% formation change generator -- n dancers, uniformly shifting right

num_dancers = 6     # number of dancers in the formation
total_spacing = 7   # total spacing, in meters

positions_initial = np.zeros((num_dancers,2)); positions_final = np.zeros((num_dancers,2))

positions_initial[:,1] = 0; positions_initial[:,0] = np.linspace(0,-total_spacing,num_dancers)
positions_final[:,1] = 1; positions_final[:,0] = np.linspace(0,total_spacing,num_dancers)

#%% formation change generator -- n dancers in a circle, uniformly expanding

num_dancers = 6     # number of dancers in the formation
total_expansion = 2 # how much the "circle" of dancers expands by

positions_initial = np.zeros((num_dancers,2)); positions_final = np.zeros((num_dancers,2))
theta_array = np.linspace(0,2*np.pi,num_dancers+1)

positions_initial[:,0] = np.cos(-theta_array[0:-1]); positions_initial[:,1] = np.sin(-theta_array[0:-1])
positions_final[:,0] = total_expansion * np.cos(theta_array[0:-1]); positions_final[:,1] = total_expansion * np.sin(theta_array[0:-1])

#%% formation change generator -- n dancers in a circle, uniformly expanding

num_dancers = 20     # number of dancers in the formation
total_expansion = 2 # how much the "circle" of dancers expands by
translation = 4     # how much we are translating the whole setup by

positions_initial = np.zeros((num_dancers,2)); positions_final = np.zeros((num_dancers,2))
theta_array = np.linspace(0,2*np.pi,num_dancers+1)

positions_initial[:,0] = np.cos(-theta_array[0:-1]); positions_initial[:,1] = np.sin(-theta_array[0:-1])
positions_final[:,0] = total_expansion * np.cos(theta_array[0:-1]); positions_final[:,1] = total_expansion * np.sin(theta_array[0:-1])
positions_final = positions_final + translation

#%% define objective functions

def max_distance_objective(positions_initial, positions_final, indices_final):
    euclidean_distance = np.power(positions_initial - positions_final[indices_final,:],2)
    euclidean_distance = np.power(np.sum(euclidean_distance,1),0.5)
    return max(euclidean_distance)

def intersection_objective(positions_initial, positions_final, indices_final):
    pass

def ccw(A,B,C):
    return (C[:,1]-A[:,1])*(B[:,0]-A[:,0]) > (B[:,1]-A[:,1])*(C[:,0]-A[:,0])

def intersect(l1p1,l1p2,l2p1,l2p2):
    return np.logical_and(ccw(l1p1,l2p1,l2p2) != ccw(l1p2,l2p1,l2p2), ccw(l1p1,l1p2,l2p1) != ccw(l1p1,l1p2,l2p2))

#%% plotting functions

def plot_movement(positions_initial,positions_final,indices_final,iteration=0):
    plt.ion()
    plt.figure()
    plt.scatter(positions_initial[:,0],positions_initial[:,1])
    plt.scatter(positions_final[:,0],positions_final[:,1])
    for i in range(np.shape(positions_initial)[0]):
        x = positions_initial[i,0]
        y = positions_initial[i,1]
        dx = positions_final[indices_final[i],0] - x
        dy = positions_final[indices_final[i],1] - y
        plt.arrow(x,y,dx,dy)
    plt.title("movement pattern, iteration = %d"%iteration)

def plot_objective_values(objective_function_values):
    plt.figure()
    plt.plot(objective_function_values)
    plt.xlabel("number of interations")
    plt.ylabel("objective function value")

#%% optimization attempt 2, given initial and final positions
# first, find the shortest path that connects all of the dancers. Start with this
# contrived example, but graduatlly move onto more elaborate "training data"

np.random.seed(42) # set the seed that we will use so the tests are repeatable

indices_final = np.random.permutation(num_dancers)
indices_available = np.ones((num_dancers,1))

max_iterations = 3
number_of_changes_befor_quitting = 10

objective_function_values = np.zeros((num_dancers+max_iterations,1))

base_array = np.ones(np.shape(positions_initial)) # something to help with vectorizing

# setting up the initial condition (for each circle in initial position, find
# the circle in final position such that Euclidean distance between these two 
# circles are minimized)
for i in range(num_dancers):
    euclidean_distance = np.power(positions_initial[i,:] - np.multiply(positions_final,indices_available),2)
    euclidean_distance = np.power(np.sum(euclidean_distance,1),0.5)
    idx = np.nanargmin(euclidean_distance)
    indices_final[i] = idx
    indices_available[idx] = np.array([np.nan])
    objective_function_values[i,:] = max_distance_objective(positions_initial, positions_final, indices_final)

plot_movement(positions_initial,positions_final,indices_final,iteration=-1)

# now for the actual optimization steps
for z in range(max_iterations): # replace with max_iterations
    # loop through each initial position
    current_loop_permutation = np.random.permutation(num_dancers)
    for y in range(num_dancers): # replace with num_dancers
        # for each dancer, make a switch in the path assignmment with the path of
        # another dancer.
        j = current_loop_permutation[y]
        ideal_switch_index = j
        current_intersection_record = 1000 # some ridiculous large number
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
            p2_j_original = np.multiply(positions_final[indices_final[j],:],base_array) # the original destination of a path for dancer j
            p2_k_original = np.multiply(positions_final[indices_final[k],:],base_array) # the original destination of a path for dancer k
            p2_j_swapped  = np.multiply(positions_final[indices_final[k],:],base_array) # the swapped  destination of a path for dancer j
            p2_k_swapped  = np.multiply(positions_final[indices_final[j],:],base_array) # the swapped  destination of a path for dancer k
            
            # now, for each of these two paths, we will count the number of 
            # intersections they each have with all other paths
            # check if the intersection exists
            intersection_original_j = np.sum(intersect(p1_j,p2_j_original,positions_initial,positions_final[indices_final,:]))
            intersection_swapped_j  = np.sum(intersect(p1_j,p2_j_swapped,positions_initial,positions_final[indices_final,:]))
            intersection_original_k = np.sum(intersect(p1_k,p2_k_original,positions_initial,positions_final[indices_final,:]))
            intersection_swapped_k  = np.sum(intersect(p1_k,p2_k_swapped,positions_initial,positions_final[indices_final,:]))
                 
            # if the intersection calculations turn out to be favorable...
            if intersection_original_j + intersection_original_k > intersection_swapped_j + intersection_swapped_k \
                and intersection_swapped_j + intersection_swapped_k < current_intersection_record:
                ideal_switch_index = k
                current_intersection_record = intersection_swapped_j + intersection_swapped_k
                
            # print(k, indices_final)
        # now, we make the switch
        indices_final[j], indices_final[ideal_switch_index] = indices_final[ideal_switch_index], indices_final[j]
    
    # print(indices_final)
    plot_movement(positions_initial,positions_final,indices_final,iteration=z)

plot_movement(positions_initial,positions_final,indices_final,iteration=z)

#%% optimization attempt 1, given initial and final positions
# in this attempt, we will use local search. Perform a random switch per round until switching
# no longer yields a more favorable outcome

import matplotlib.animation as animation

np.random.seed(0) # set the seed that we will use so the tests are repeatable

indices_final = np.random.permutation(np.arange(0,num_dancers,1))
indices_temp = indices_final.copy()

max_iterations = 1000
number_of_changes_befor_quitting = 10

objective_function_values = np.zeros((max_iterations,1))

# Initialize parameters related for animating the movement
plt.ioff(); fig, ax = plt.subplots(); frames = []

for i in range(max_iterations):
    idx1 = np.random.randint(0,num_dancers)
    idx2 = np.random.randint(0,num_dancers)
    indices_temp[idx1], indices_temp[idx2] = indices_temp[idx2], indices_temp[idx1]
    if max_distance_objective(positions_initial, positions_final,indices_temp) <= \
        max_distance_objective(positions_initial, positions_final,indices_final):
            indices_final = indices_temp.copy()
    objective_function_values[i,:] = max_distance_objective(positions_initial, positions_final, indices_final)
    
# Create an animation
ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=100, blit=True, repeat_delay=1000)

# Save the animation as a video
ani.save("output_video1.gif", writer="pillow")

plot_objective_values(objective_function_values)