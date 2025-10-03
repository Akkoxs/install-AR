# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 21:52:33 2025
@author: kai-s
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as mat
import seaborn as sea
from sklearn.model_selection import StratifiedShuffleSplit


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

#group by outputs (key, group) pairs
grouped_by_step = df.groupby('Step')

colours = [ #there are 13 colours here, one for each step
    'blue',
    'green',
    'red',
    'cyan',
    'magenta',
    'yellow',
    'black',
    'white',
    'orange',
    'purple',
    'brown',
    'pink',
    'gray']

fig = mat.figure(layout = 'constrained')
threeD_plt = fig.add_subplot(1, 1, 1, projection = '3d')

#unpack those key, group pairs, key is the step and the group we can access for the coords
for i, (step_label, coords) in enumerate(grouped_by_step): #unpack tuple
    threeD_plt.scatter(coords['X'],
                       coords['Y'], 
                       coords['Z'], 
                       color = colours[i], 
                       label = f'Step {step_label}'
                       )

threeD_plt_xlabel = 'X Coordinate'
threeD_plt_ylabel = 'Y Coordinate'
threeD_plt_zlabel = 'Z Coordinate'
threeD_plt.legend(loc ='center left', bbox_to_anchor = (1.1, 0.5))

mat.show()
"----------------------------------------------------------------------------"

"Step  3: Correlation Analysis"
"----------------------------------------------------------------------------"
#firstly, let us split our dataset to avoid data leakage 
splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 23)



#pearson correlation
# include correlation plot
#explain correlation between features and target vars (REPORT)

"----------------------------------------------------------------------------"