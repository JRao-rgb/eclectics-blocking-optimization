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

def lines_and_windows(num_dancers=28,\
                      x_size=10,\
                      y_size=5,\
                      offset=[0,0],\
                      num_dancers_per_row=6,\
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

    # flatten just converts a 2D array into a 1D array -- it "flattens" it hehe    
    return x.flatten()[0:num_dancers], y.flatten()[0:num_dancers]

def pyramid(num_dancers=28,\
            base=10,\
            height=5,\
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
    while True:
        print(row, num_dancers_per_row, total_dancers)
        print(x[total_dancers-num_dancers_per_row:total_dancers])
        print(np.linspace(0,num_dancers_per_row,num_dancers_per_row)[..., None])
        x[total_dancers-num_dancers_per_row:total_dancers] = np.linspace(0,num_dancers_per_row-1,num_dancers_per_row)[..., None] - (num_dancers_per_row-1)/2
        y[total_dancers-num_dancers_per_row:total_dancers] = row-1
        row += 1
        num_dancers_per_row = row
        total_dancers += num_dancers_per_row
        if total_dancers > num_dancers:
            break
        
    # normalize the pyramid array to height of 1, base of 1
    x = x / (num_dancers_per_row - 2) * 2
    y = y / (row - 2)
    
    # scale and offset the pramid
    x = x * base + offset[0]
    y = y * height + offset[0]
    
    # return the arryas, truncated based on num_dancers
    return x[0:num_dancers], y[0:num_dancers]

def grid(num_dancers=28,\
        x_size=10,\
        y_size=5,\
        offset=[0,0],\
        num_dancers_per_row=6,\
        initial_offset=1):
    
    # create a grid first, then delete extraneous values that correspond to void people
    num_dancers_per_col = math.ceil(num_dancers/num_dancers_per_row)
    
    # generate the inidial x and y coordinates
    x_coords = np.linspace(-x_size/2+offset[0], x_size/2+offset[0], num_dancers_per_row)
    y_coords = np.linspace(-y_size/2+offset[1], y_size/2+offset[1], num_dancers_per_col)
    
    # generating the actual x and y coordinates
    x, y = np.meshgrid(x_coords,y_coords)

    # flatten just converts a 2D array into a 1D array -- it "flattens" it hehe    
    return x.flatten()[0:num_dancers], y.flatten()[0:num_dancers]

def horizontal_line(num_dancers=28,\
                    length=10,\
                    offset=[0,0]):
    x_coords = np.linspace(-length/2+offset[0], length/2+offset[0], num_dancers)
    y_coords = np.linspace(offset[1],offset[1], num_dancers)
    
    x, y = np.meshgrid(x_coords,y_coords)
    
    return x.flatten(), y.flatten()

def vertical_line(num_dancers=28,\
                  length=10,\
                  offset=[0,0]):
    
    x_coords = np.linspace(offset[0], offset[0], num_dancers)
    y_coords = np.linspace(-length/2+offset[1], length/2+offset[1], num_dancers)
    
    x, y = np.meshgrid(x_coords,y_coords)
    
    return x.flatten(), y.flatten()

def ring(num_dancers=28,\
         radius=10,\
         offset=[0,0]):
    
    theta_array = np.linspace(0,2*np.pi,num_dancers)
    x = np.cos(theta_array) 
    y = np.sin(theta_array)

    return x, y

def clump(n):
    pass

def diagonals(n):
    pass

def v(n):
    pass

x, y = lines_and_windows()
plt.scatter(x,y)

x, y = ring()
plt.scatter(x,y)

x, y = pyramid()
plt.scatter(x,y)