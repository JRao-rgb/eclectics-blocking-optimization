# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:34:33 2024

@author: jraos
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# - num_dancers is the number of dancers
# - offset = (x, y) away from the center that the "center of mass" of the
#   formation is located at

def lines_and_windows(num_dancers=28,
                      x_size=10,
                      y_size=5,
                      offset=[0,0],
                      num_dancers_per_row=6,
                      initial_offset=1):
    
    # create a grid first, then offset every other row, then delete extraneous values
    num_dancers_per_col = math.ceil(num_dancers/num_dancers_per_row)
    
    # generate the inidial x and y coordinates
    x_coords = np.linspace(-x_size/2+offset[0], x_size/2+offset[0], num_dancers_per_row)
    y_coords = np.linspace(-y_size/2+offset[1], y_size/2+offset[1], num_dancers_per_col)
    
    # find the windows spacing -- used to offset the rows later
    x_spacing = x_coords[1] - x_coords[0]
    
    # generating the actual x and y coordinates
    x, y = np.meshgrid(x_coords,y_coords)
    
    # offseting every other row
    x[0::2, :] += initial_offset*x_spacing/4
    x[1::2, :] -= initial_offset*x_spacing/4

    # symmetrize
    total_dancers = num_dancers_per_col * num_dancers_per_row # this is how many
    # dancers we would have if the grid was completely filled
    
    d = num_dancers_per_row
    l = np.int8(num_dancers - (total_dancers - num_dancers_per_row))
    f = np.int8(total_dancers - num_dancers_per_row)
    offset_idx = np.int8(math.floor((d-l)/2))
    
    # flatten just converts a 2D array into a 1D array -- it "flattens" it hehe  
    x = x.flatten(); y = y.flatten()
    
    x = np.concatenate([x[0:f],x[f+offset_idx:f+offset_idx+l]])
    y = np.concatenate([y[0:f],y[f+offset_idx:f+offset_idx+l]])
  
    return x.flatten(), y.flatten()

def pyramid(num_dancers=28,
            base=10,
            height=5,
            offset=[0,0]):
    
    # create a full triangle first, then flatten and delete the unused ones
    row = 1
    num_dancers_per_row = 1
    total_dancers = 1
    while True:
        row += 1
        num_dancers_per_row = row
        total_dancers += num_dancers_per_row
        if total_dancers >= num_dancers:
            break
    
    x = np.zeros((total_dancers,1))
    y = np.zeros((total_dancers,1))
    
    # generate the initial ordered pyramid array
    row = 1
    num_dancers_per_row = 1
    total_dancers = 1
    x[total_dancers-num_dancers_per_row:total_dancers] = np.linspace(0,num_dancers_per_row-1,num_dancers_per_row)[..., None] - (num_dancers_per_row-1)/2
    y[total_dancers-num_dancers_per_row:total_dancers] = row-1
    while True:
        row += 1
        num_dancers_per_row = row
        total_dancers += num_dancers_per_row
        x[total_dancers-num_dancers_per_row:total_dancers] = np.linspace(0,num_dancers_per_row-1,num_dancers_per_row)[..., None] - (num_dancers_per_row-1)/2
        y[total_dancers-num_dancers_per_row:total_dancers] = row-1
        if total_dancers >= num_dancers:
            break
        
    # normalize the pyramid array to height of 1, base of 1
    x = x / (num_dancers_per_row - 1) * 2
    y = y / (row - 1) - 0.5
    
    # scale and offset the pramid
    x = x * base + offset[0]
    y = y * height + offset[0]
    
    # symmetrizing the formation
    d = num_dancers_per_row
    l = np.int8(num_dancers - (total_dancers - num_dancers_per_row))
    f = np.int8(total_dancers - num_dancers_per_row)
    offset_idx = np.int8(np.floor((d-l)/2))
    
    x = np.concatenate([x[0:f],x[f+offset_idx:f+offset_idx+l]])
    y = np.concatenate([y[0:f],y[f+offset_idx:f+offset_idx+l]])
    
    # return the arryas, flattening again due to numpy datatypes
    return x.flatten(), y.flatten()

def grid(num_dancers=28,
        x_size=10,
        y_size=5,
        offset=[0,0],
        num_dancers_per_row=6,
        initial_offset=1):
    
    # create a grid first, then delete extraneous values that correspond to void people
    num_dancers_per_col = math.ceil(num_dancers/num_dancers_per_row)
    
    # generate the inidial x and y coordinates
    x_coords = np.linspace(-x_size/2+offset[0], x_size/2+offset[0], num_dancers_per_row)
    y_coords = np.linspace(-y_size/2+offset[1], y_size/2+offset[1], num_dancers_per_col)
    
    # generating the actual x and y coordinates
    x, y = np.meshgrid(x_coords,y_coords)
    
    # symmetrize
    total_dancers = num_dancers_per_col * num_dancers_per_row # this is how many
    # dancers we would have if the grid was completely filled
    
    d = num_dancers_per_row
    l = np.int8(num_dancers - (total_dancers - num_dancers_per_row))
    f = np.int8(total_dancers - num_dancers_per_row)
    offset_idx = np.int8(math.floor((d-l)/2))
    
    x = x.flatten(); y = y.flatten()
    
    x = np.concatenate([x[0:f],x[f+offset_idx:f+offset_idx+l]])
    y = np.concatenate([y[0:f],y[f+offset_idx:f+offset_idx+l]])

    # flatten just converts a 2D array into a 1D array -- it "flattens" it hehe    
    return x.flatten(), y.flatten()

def horizontal_line(num_dancers=28,
                    length=10,
                    offset=[0,0]):
    x_coords = np.linspace(-length/2+offset[0], length/2+offset[0], num_dancers)
    y_coords = np.linspace(offset[1],offset[1], num_dancers)
    
    x, y = np.meshgrid(x_coords,y_coords)
    
    return x.flatten(), y.flatten()

def vertical_line(num_dancers=28,
                  length=10,
                  offset=[0,0]):
    
    x_coords = np.linspace(offset[0], offset[0], num_dancers)
    y_coords = np.linspace(-length/2+offset[1], length/2+offset[1], num_dancers)
    
    x, y = np.meshgrid(x_coords,y_coords)
    
    return x.flatten(), y.flatten()

def ring(num_dancers=28,
         radius=5,
         offset=[0,0]):
    
    theta_array = np.linspace(0,2*np.pi,num_dancers+1)
    x = radius * np.cos(theta_array) + offset[0]
    y = radius * np.sin(theta_array) + offset[1]

    return x[0:-1], y[0:-1]

def clump(n):
    pass

def diagonals(n):
    pass

def v(n):
    pass

# x, y = lines_and_windows(num_dancers = 33)
# plt.scatter(x,y)

# x, y = grid(num_dancers = 24,num_dancers_per_row=8)
# plt.scatter(x,y)

# x, y = ring()
# plt.scatter(x,y)

# x, y = pyramid()
# plt.scatter(x,y)