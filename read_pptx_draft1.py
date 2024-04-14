# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:26:03 2024

@author: jraos
"""

import pptx as ppt
import os

os.chdir("C:/Users/jraos/OneDrive - Stanford/Documents/Stanford/EC/Eclectics Blocking Optimization")

data_dir = "data"
blocking_diagram = ppt.Presentation(data_dir + "/wake-up-cleaned.pptx")

#%% extracting all the text from the slides

text_runs = []
text_positions = []

for slide in blocking_diagram.slides:
    text_positions_per_slide = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                text_runs.append(run.text)
                text_positions_per_slide.append([run.text,shape.left.inches, shape.top.inches])
    text_positions.append(text_positions_per_slide)
                
#%% displaying the text at the correct locations

import matplotlib.pyplot as plt

plt.figure()
for formation in text_positions:
    for dancer in formation:
        plt.text(dancer[1]/16,dancer[2]/9,dancer[0],size='large')
    break
    
#%% extracting all the shapes fromm the slides

shape_list = []

for slide in blocking_diagram.slides:
    shape_list.append(slide.shapes)
    for shape in slide.shapes:
        print(shape.top, shape.left, shape.height, shape.width)