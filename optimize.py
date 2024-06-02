# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:14:39 2024

@author: jraos
"""

import os
os.chdir("C:\\Users\\jraos\\OneDrive - Stanford\\Documents\\Stanford\\EC\\Eclectics Blocking Optimization\\eclectics-blocking-optimization")

import numpy as np
import computational_geometry as geo
import metrics


def optimize_formation(X, Y, C, P, max_iteration, sf1, sf2, sf3, display_metrics=False):
    num_dancers = np.shape(X)[0]
    num_formations = np.shape(X)[1]
    
    P_inv = np.full((num_dancers,num_formations), 0, dtype=np.int8)
    P_inv[:, ] = np.linspace(0, num_dancers-1, num_dancers)[..., None]
    # of the algorithm is to figure out what numbers to put in here
    # constraint array. This array
    M = np.full(num_dancers, -1, dtype=np.int8)
    # in the P array to actual dancer numbers specified in the C array. M[positionID] = dancerID
    # mapping that's the inverse of M:
    M_inv = np.full(num_dancers, -1, dtype=np.int8)
    # you give it a dancer ID and it tells you the posiiton ID that it's mapped to. M_inv[dancerID] = positionID
    
    # initializing any constraints that are present in formation 0 (the starting formation):
    # indices of the coordinates that are constrained (i.e. already have dancers assigned)
    constrained_coordinates_mask = C[:, 0] != -1
    # dancerIDs of the circles that are constrained (i.e. have specified dancers)
    constrained_dancerIDs = C[constrained_coordinates_mask, 0]
    # positionIDs of the circles that are constrained
    constrained_positionIDs = P[constrained_coordinates_mask, 0]
    # initial mapping from positionIDs to dancerIDs
    M[constrained_positionIDs] = constrained_dancerIDs
    # initial mapping from dancerIDs to positionIDs
    M_inv[constrained_dancerIDs] = constrained_positionIDs
    
    # pre-allocate arrays that are needed to keep track of which dancers can go where. Call this A.
    A = np.full((num_dancers, num_dancers), False)
    
    # loop over formations-1
    for formation in range(num_formations-1):
        i = formation
    
        # -------------------------------------------------------------------------
        # ===================== START BY ENFORCING CONSTRAINTS ====================
        # -------------------------------------------------------------------------
        # next, fill out values of A according to known values of M, C
        A = np.full((num_dancers, num_dancers), False)
        for c1, p in enumerate(P[:, i]):
    
            for c2, dancerID in enumerate(C[:, i+1]):
                # check if p has somewhere it needs to be
                if M[p] != -1:
                    does_p_have_somewhere_it_needs_to_be = np.any(
                        C[:, i+1] == M[p])
                else:
                    does_p_have_somewhere_it_needs_to_be = False
    
                # check if someone else is supposed to be at c2
                if dancerID != -1:  # if a constraint exists:
                    if M[p] != -1 and dancerID == M[p]:
                        is_someone_else_supposed_to_be_here = False
                    elif M[p] != -1 and dancerID != M[p]:
                        is_someone_else_supposed_to_be_here = True
                    elif M_inv[dancerID] != -1 and p == M_inv[dancerID]:
                        is_someone_else_supposed_to_be_here = False
                    elif M_inv[dancerID] != -1 and p != M_inv[dancerID]:
                        is_someone_else_supposed_to_be_here = True
                    elif M_inv[dancerID] == -1:
                        is_someone_else_supposed_to_be_here = False
                else:
                    is_someone_else_supposed_to_be_here = False
    
                # check to see if this is where p is supposed to end up
                if M[p] != -1 and M[p] == dancerID:
                    p_is_supposed_to_end_up_here = True
                else:
                    p_is_supposed_to_end_up_here = False
    
                A[p, c2] = (does_p_have_somewhere_it_needs_to_be == False and
                            is_someone_else_supposed_to_be_here == False) or \
                    p_is_supposed_to_end_up_here == True
    
        # initialize a permutation for each formation based on Euclidean distance
        # depends on if we want to normalize the formations before optimization...
        idx_available = np.ones((num_dancers))
    
        # initialize a permutation for each formation based on Euclidean distance
        # and the constraints available.
        number_of_allowed_pos_per_position  = np.sum(A, axis=1)
        sorted_indices_of_constraints       = np.argsort(number_of_allowed_pos_per_position)
        number_of_allowed_pos_after_sorting = np.sum(A[sorted_indices_of_constraints,:],axis = 1) # should be strictly ascending array
        
        # print(P)
        for coordinateID, unassigned_position in enumerate(P[:,i]):
            available_coordinate_IDs = np.float16(A[sorted_indices_of_constraints[unassigned_position],:].flatten())
            available_coordinate_IDs[np.logical_not(available_coordinate_IDs)] = np.nan
            
            euclidean_distance = np.power(X[coordinateID,i] - X[:,i+1],2) + \
                                 np.power(Y[coordinateID,i] - Y[:,i+1],2)
                                 
            euclidean_distance = np.multiply(np.multiply(idx_available,euclidean_distance),available_coordinate_IDs)
            
            idx_to_assign = np.nanargmin(euclidean_distance)

            P[idx_to_assign,i+1] = sorted_indices_of_constraints[unassigned_position]
            idx_available[idx_to_assign] = np.nan
        
        # insert statements to check to make sure we've done the reassignment properly
        # if this was done properly, we would see that we end up with a valid permutation
        # for P[:,i+1]
        assert(np.shape(np.unique(P[:,i+1]))[0]==num_dancers)
    
        # initialize an additional matrix that's similar to A, but gets modified during
        # the optimization process to reflects what's still available
        A_dynamic = A.copy()
    
        # -------------------------------------------------------------------------
        # ======================= START ACTUAL OPTIMIZATION =======================
        # -------------------------------------------------------------------------
        # loop over iterations per formation
        for iteration in range(max_iteration):
            # randomize the order at which we scan through the dancers here
            current_iter_permutation = np.random.permutation(num_dancers)
            
            # reset A_dynamic, since it gets altered every iteration
            A_dynamic = A.copy()
            
            # loop over each number of available positions:
            for number_of_originally_available_coordinateIDs in range(1,num_dancers+1): # allowed positions goes from 1 to num_dancers, 
            # so we must alter this for-loop accordingly
            
                # looping over each positionID:
                for index in range(num_dancers):  # range(num_dancers-1)
                    # trying to find the best c2 in formation i+1 to match positionID p to, given 
                    # that positionID p was assigned to c1 in formation i
                    coordinateID_formation_i         = current_iter_permutation[index]
                    positionID_to_be_assigned        = P[coordinateID_formation_i,i]
                    previously_assigned_coordinateID = int(np.where(P[:,i+1]==positionID_to_be_assigned)[0][0])
                    available_coordinate_IDs  = A[positionID_to_be_assigned,:].flatten()
                    number_of_available_coordinateIDs_for_this_positionID = np.sum(np.int8(available_coordinate_IDs))
                                    
                    if number_of_available_coordinateIDs_for_this_positionID != number_of_originally_available_coordinateIDs: continue # skip over this if it's not right
                    
                    # initialize the values of P_inv based on the values of P:
                    for coordinateID in range(num_dancers):
                        P_inv[P[coordinateID,i+1],i+1] = coordinateID
                    
                    # now we update the available IDs using the dynamic array. The
                    # previous step was just to make sure we are incrementing up
                    # the initially checked array to make sure we are guaranteed
                    # to still have spots
                    available_coordinate_IDs  = A_dynamic[positionID_to_be_assigned,:].flatten()
                    number_of_available_coordinateIDs_for_this_positionID = np.sum(np.int8(available_coordinate_IDs))
                    num_cID_avail = number_of_available_coordinateIDs_for_this_positionID
                    
                    # print("available positions for pID", positionID_to_be_assigned, ":",available_coordinate_IDs)
                    
                    # begin the "swapping" process to determine which one is the best
                    # fit. Clearly, the size of the number of "swaps" only corresponds
                    # to the number of allowed positions
                    
                    # loop over the rest of the dancers. Now we are swapping dancer k and l
                    positionID_to_be_assigned_path_start_x = \
                        np.full((num_dancers, num_cID_avail), X[coordinateID_formation_i, i])
                    positionID_to_be_assigned_path_start_y = \
                        np.full((num_dancers, num_cID_avail), Y[coordinateID_formation_i, i])
                    positionID_to_be_assigned_path_end_x = \
                        np.full((num_dancers, num_cID_avail), 0, dtype=np.float16)
                    positionID_to_be_assigned_path_end_y = \
                        np.full((num_dancers, num_cID_avail), 0, dtype=np.float16)
                    
                    positionID_that_got_replaced_path_start_x = \
                        np.full((num_dancers, num_cID_avail), 0, dtype=np.float16)
                    positionID_that_got_replaced_path_start_y = \
                        np.full((num_dancers, num_cID_avail), 0, dtype=np.float16)
                    positionID_that_got_replaced_path_end_x = \
                        np.transpose(np.tile(X[available_coordinate_IDs, i+1][..., None], num_dancers))
                    positionID_that_got_replaced_path_end_y = \
                        np.transpose(np.tile(Y[available_coordinate_IDs, i+1][..., None], num_dancers))
        
                    # arrays to help with the Euclidean distance calculation
                    # print(P_inv[:,i])
                    # print(X)
                    # print(X[P_inv[:,i],i])
                    X_initial = np.tile(X[P_inv[:, i], i][..., None], number_of_available_coordinateIDs_for_this_positionID)
                    Y_initial = np.tile(Y[P_inv[:, i], i][..., None], number_of_available_coordinateIDs_for_this_positionID)
                    X_final = np.tile(X[P_inv[:, i+1], i+1][..., None], number_of_available_coordinateIDs_for_this_positionID)
                    Y_final = np.tile(Y[P_inv[:, i+1], i+1][..., None], number_of_available_coordinateIDs_for_this_positionID)
        
                    idx = 0 
                    for coordinateID, availability in enumerate(available_coordinate_IDs):
                        if availability == False: continue # skip the coordinateIDs that cannot be assigned to our positionID_to_be_assigned
        
                        positionID_previously_assigned_to_coordinateID = P[coordinateID,i+1]
                        coordinateID_of_this_position_in_formation_f0  = int(np.where(P[:,i]==positionID_previously_assigned_to_coordinateID)[0][0])
        
                        P_swap = P[:,i+1].copy()
                        P_swap[coordinateID], P_swap[previously_assigned_coordinateID] = \
                            P_swap[previously_assigned_coordinateID], P_swap[coordinateID]
                        
                        P_inv_swap = P_inv[:,i+1].copy()
                        P_inv_swap[positionID_to_be_assigned], P_inv_swap[positionID_previously_assigned_to_coordinateID] = \
                            P_inv_swap[positionID_previously_assigned_to_coordinateID], P_inv_swap[positionID_to_be_assigned]
    
                        positionID_to_be_assigned_path_end_x[:, idx] = np.transpose(np.tile(X[coordinateID, i+1], num_dancers))
                        positionID_to_be_assigned_path_end_y[:, idx] = np.transpose(np.tile(Y[coordinateID, i+1], num_dancers))
                        positionID_that_got_replaced_path_start_x[:, idx] = np.transpose(np.tile(X[coordinateID_of_this_position_in_formation_f0, i], num_dancers))
                        positionID_that_got_replaced_path_start_y[:, idx] = np.transpose(np.tile(Y[coordinateID_of_this_position_in_formation_f0, i], num_dancers))
                        positionID_that_got_replaced_path_end_x[:, idx] = np.transpose(np.tile(X[previously_assigned_coordinateID, i+1], num_dancers))
                        positionID_that_got_replaced_path_end_y[:, idx] = np.transpose(np.tile(Y[previously_assigned_coordinateID, i+1], num_dancers))
                        
                        X_final[:, idx] = X[P_inv_swap, i+1]
                        Y_final[:, idx] = Y[P_inv_swap, i+1]
                        
                        idx += 1
        
                    # performing the intersection calculations
                    arr_intersection =  geo.intersect(positionID_to_be_assigned_path_start_x, 
                                                      positionID_to_be_assigned_path_start_y,
                                                      positionID_to_be_assigned_path_end_x, 
                                                      positionID_to_be_assigned_path_end_y,
                                                      X[:, i][...,None], 
                                                      Y[:, i][..., None],
                                                      X[:, i+1][..., None], 
                                                      Y[:, i+1][..., None]) + \
                                        geo.intersect(positionID_that_got_replaced_path_start_x, 
                                                      positionID_that_got_replaced_path_start_y,
                                                      positionID_that_got_replaced_path_end_x, 
                                                      positionID_that_got_replaced_path_end_y,
                                                      X[:, i][..., None], 
                                                      Y[:, i][..., None],
                                                      X[:, i+1][..., None], 
                                                      Y[:, i+1][..., None])
        
                    num_intersection = np.sum(arr_intersection, axis=0)
        
                    cost = sf1 * num_intersection + \
                        sf2 * np.max(geo.euclidean_distance(X_initial,
                                                            X_final,
                                                            Y_initial,
                                                            Y_final), axis=0) + \
                        sf3 * np.average(geo.euclidean_distance(X_initial,
                                                                X_final,
                                                                Y_initial,
                                                                Y_final), axis=0)

                    
                    ideal_assignment_idx = np.argmin(cost) # this is in terms of unknown IDs!!!
                    ideal_assignment_coordinateID = np.int8(np.linspace(0,num_dancers-1,num_dancers)[available_coordinate_IDs][ideal_assignment_idx])
        
                    P[previously_assigned_coordinateID, i+1], P[ideal_assignment_coordinateID, i+1] =\
                        P[ideal_assignment_coordinateID, i+1], P[previously_assigned_coordinateID, i+1]
                    A_dynamic[:,ideal_assignment_coordinateID] = False
            
                    # print("Pa_start",positionID_to_be_assigned_path_start_x[0,:])
                    # print("Pa_end  ",positionID_to_be_assigned_path_end_x[0,:])
                    # print("Pi_start",positionID_that_got_replaced_path_start_x[0,:])
                    # print("Pi_end  ",positionID_that_got_replaced_path_end_x[0,:])
                    # print("X_init  ")
                    # print(X_initial)
                    # print("X_fin   ")
                    # print(X_final)
                    # print("Euclidean Distance", geo.euclidean_distance(X_initial,
                    #                                         X_final,
                    #                                         Y_initial,
                    #                                         Y_final))
                    # print("averaged dist", np.average(geo.euclidean_distance(X_initial,
                    #                                         X_final,
                    #                                         Y_initial,
                    #                                         Y_final),axis=0))
                    # print("Cost   ",cost)
            if display_metrics:
                print("formation", formation, "iteration", iteration, "intersections count",
                      metrics.num_intersections(X[:, i:i+2], Y[:, i:i+2], P_inv[:, i:i+2]),
                      "\tmax dist", metrics.max_distance(X[:, i:i+2], Y[:, i:i+2], P_inv[:, i:i+2]),
                      "\tavg dist", metrics.average_distance(X[:, i:i+2], Y[:, i:i+2], P_inv[:, i:i+2]),
                      "\tcost", np.min(cost))
        
        # assigning new values of M so that optimization in the next step can use this:
        # print(M, M_inv)
        # print(C)
        for coordinateID in range(num_dancers):
            if C[coordinateID,i+1] != -1 and M_inv[C[coordinateID,i+1]] == -1:
                M_inv[C[coordinateID,i+1]] = P[coordinateID,i+1]
                M[P[coordinateID,i+1]] = C[coordinateID,i+1]
        # print(M, M_inv)
        
        # finalize the value of P_inv (note to self: we might not need this?)
        for coordinateID in range(num_dancers):
            P_inv[P[coordinateID,i+1],i+1] = coordinateID
            
    return P, P_inv, M, M_inv