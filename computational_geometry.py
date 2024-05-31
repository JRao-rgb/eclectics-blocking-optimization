# -*- coding: utf-8 -*-
"""
Created on Thu May 30 06:19:02 2024

@author: jraos
"""

import numpy as np
import matplotlib.pyplot as plt

def onSegment(A,B,C):
    return np.logical_and(np.logical_and(A[:,0] <= np.maximum(B[:,0],C[:,0]), A[:,0] >= np.minimum(B[:,0],C[:,0])), \
                          np.logical_and(A[:,1] <= np.maximum(B[:,1],C[:,1]), A[:,1] >= np.minimum(B[:,1],C[:,1])))

def ccw(Ax,Ay,Bx,By,Cx,Cy):
    return np.multiply((Cy-Ay),(Bx-Ax)) > np.multiply((By-Ay),(Cx-Ax))

def intersect(l1p1x,l1p1y,l1p2x,l1p2y,l2p1x,l2p1y,l2p2x,l2p2y):
    o1 = ccw(l1p1x,l1p1y,l2p1x,l2p1y,l2p2x,l2p2y)
    o2 = ccw(l1p2x,l1p2y,l2p1x,l2p1y,l2p2x,l2p2y)
    o3 = ccw(l1p1x,l1p1y,l1p2x,l1p2y,l2p1x,l2p1y)
    o4 = ccw(l1p1x,l1p1y,l1p2x,l1p2y,l2p2x,l2p2y)
    
    # General Case
    intersect_arr = np.logical_and(np.not_equal(o1, o2), np.not_equal(o3, o4))
    
    # # Special Cases
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o1, False), onSegment(l1p1,l2p1,l1p2)))
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o2, False), onSegment(l1p1,l2p2,l1p2)))
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o3, False), onSegment(l2p1,l1p1,l2p2)))
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o4, False), onSegment(l2p1,l1p2,l2p2)))
        
    return np.int8(intersect_arr)

def plot_movement(X_initial,X_final,Y_initial,Y_final,indices_initial,indices_final):
    plt.ion()
    plt.figure()
    plt.scatter(X_initial,Y_initial)
    plt.scatter(X_final,Y_final)
    for i in range(np.shape(X_initial)[0]):
        x = X_initial[indices_initial[i]]
        y = Y_initial[indices_initial[i]]
        dx = X_final[indices_final[i]] - x
        dy = Y_final[indices_final[i]] - y
        plt.arrow(x,y,dx,dy)
    plt.title("movement")
        
def ccw2(A,B,C):
    return (C[:,1]-A[:,1])*(B[:,0]-A[:,0]) > (B[:,1]-A[:,1])*(C[:,0]-A[:,0])

def intersect2(l1p1,l1p2,l2p1,l2p2):
    o1 = ccw2(l1p1,l2p1,l2p2)
    o2 = ccw2(l1p2,l2p1,l2p2)
    o3 = ccw2(l1p1,l1p2,l2p1)
    o4 = ccw2(l1p1,l1p2,l2p2)
    
    # General Case
    intersect_arr = np.logical_and(np.not_equal(o1, o2), np.not_equal(o3, o4))
    
    # # Special Cases
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o1, False), onSegment(l1p1,l2p1,l1p2)))
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o2, False), onSegment(l1p1,l2p2,l1p2)))
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o3, False), onSegment(l2p1,l1p1,l2p2)))
    # intersect_arr = np.logical_or(intersect_arr,np.logical_and(np.equal(o4, False), onSegment(l2p1,l1p2,l2p2)))
        
    return intersect_arr
    
#%% animating the formations as each dancer moves from one to another


# position_matrix = -1*np.ones((60,20,2)) # array that contains the dancerID in the rows and the 
# # timed position in the columns. So #rows = #dancers and #cols = #formations
# # dancer0 is a test object because Python is 0-indexed

# for i, formation in enumerate(text_positions):
#     for j, dancer in enumerate(formation):
#         position_matrix[int(dancer[0])][i][0] = dancer[1]/slide_width
#         position_matrix[int(dancer[0])][i][1] = dancer[2]/slide_height

# # now, make it so that each dancer's position is padded and it kind of animates
# # the movement of the dancers across the stage

# fps = 10 # 10 frames between each formation in the animation
# freeze_frames = 10 # number of frames the current positions will be frozen for
# animated_position_matrix = -1 * np.ones((60,(20-1)*fps+20*freeze_frames,2))

# for i in range(np.shape(position_matrix)[0]-1):
#     # create the initial freeze frames
#     for k in range(freeze_frames):
#         animated_position_matrix[i][k][0] = position_matrix[i,0,0]
#         animated_position_matrix[i][k][1] = position_matrix[i,0,1]
        
#     for j in range(np.shape(position_matrix)[1]-1):
#         x1 = position_matrix[i,j,0]
#         x2 = position_matrix[i,j+1,0]
#         y1 = position_matrix[i,j,1]
#         y2 = position_matrix[i,j+1,1]
        
#         if int(y1) == -1:
#             x1 = int(x2 > 0.5)
#             y1 = y2
#         elif int(y2) == -1:
#             x2 = int(x1 > 0.5)
#             y2 = y1
        
#         for k in range(fps):
#             animated_position_matrix[i][j*(fps+freeze_frames)+freeze_frames+k][0] = x1 + k * (x2-x1)/fps
#             animated_position_matrix[i][j*(fps+freeze_frames)+freeze_frames+k][1] = y1 + k * (y2-y1)/fps
        
#         for k in range(freeze_frames):
#             animated_position_matrix[i][j*(fps+freeze_frames)+fps+freeze_frames+k][0] = x2
#             animated_position_matrix[i][j*(fps+freeze_frames)+fps+freeze_frames+k][1] = y2
            
# #%% making a video of the whole thing

# import matplotlib.animation as animation

# # Create an array of images (for demonstration purposes)
# # You can replace this with your actual images
# num_frames = (20-1)*fps+1

# # Create a figure
# plt.ioff()
# fig, ax = plt.subplots()

# # Initialize an empty list to store frames
# frames = []

# # Add images to frames
# for i in range(num_frames):
#     container = ax.scatter(animated_position_matrix[:,i,0], animated_position_matrix[:,i,1], animated=True, color='b')
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     frames.append([container])

# # Create an animation
# ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=100, blit=True, repeat_delay=1000)

# # Save the animation as a video
# ani.save("output_video.gif", writer="pillow")