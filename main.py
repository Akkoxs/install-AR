# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 21:52:33 2025
@author: kai-s
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as mat

mat.close()

"Step  1: Data Processing"
"----------------------------------------------------------------------------"
df = pd.read_csv("Project 1 Data.csv")

print("DATASAMPLE")
print(df.head())
"----------------------------------------------------------------------------"

"Step  2: Data Visualization"
"----------------------------------------------------------------------------"

#Histogram
hist = df['Step'].hist()
hist.set_title("Step vs # of Data Points")
hist.set_xlabel("Step")
hist.set_ylabel("# of Data Points")


#Grouped Bar Chart - Grouped by avg val of coordinate for each step
grouped_avg_df = df.groupby('Step').mean() #groups dataframe by Step and the avg coordinates per step (XYZ)

steps = np.arange(len(grouped_avg_df))
mult = 0 #no. of bars
bar_width = 0.25 #of each bar 

step_coords= {
    'X': (grouped_avg_df['X']),
    'Y': (grouped_avg_df['Y']),
    'Z': (grouped_avg_df['Z'])}

fig, barplt = mat.subplots(layout ='constrained')

for coord, avg_val in step_coords.items():
    bar_pos = bar_width*mult
    rects = barplt.bar(steps + bar_pos, avg_val, bar_width, label = coord)
    barplt.bar_label(rects, padding = 3)
    mult += 1

barplt.set_ylabel("Mean of XYZ Coordinates")
barplt.set_xlabel("Step and XYZ")
barplt.legend()

mat.show()


#3D Scatterplot

fig = mat.figure(layout = 'constrained')
threeD_plt = fig.add_subplot(1, 1, 1, projection = '3d')
threeD_plt.scatter(df['X'], df['Y'], df['Z'])

#for step, coords in df

threeD_plt_xlabel = 'X Coordinate'
threeD_plt_ylabel = 'Y Coordinate'
threeD_plt_zlabel = 'Z Coordinate'
barplt.legend()

mat.show()


#Plan for tomorrow
#Group data by row Step, X, Y, Z 
#Based on Step, give it a colour/shape
#Plot it on 3D scatterplot



"----------------------------------------------------------------------------"

"Step  3: Correlation Analysis"
"----------------------------------------------------------------------------"

"----------------------------------------------------------------------------"