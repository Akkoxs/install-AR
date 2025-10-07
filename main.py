# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 21:52:33 2025
@author: kai-s
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as mat
import seaborn as sea
import warnings
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression

#supress warnings - CHECK
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

"Step  1: Data Processing"
"----------------------------------------------------------------------------"
df = pd.read_csv("Project 1 Data.csv")

print("DATASAMPLE")
print(df.head())
"----------------------------------------------------------------------------"

"Step  2: Data Visualization"
"----------------------------------------------------------------------------"
#Start splitting to avoid data leakage & embedding in data snooping bias 

#returns the # of split.shuffle ops
splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 23)

for train_index, test_index in splitter.split(df, df["Step"]):    
    train_set = df.loc[train_index].reset_index(drop=True) #total
    train_dat = train_set.drop(columns=["Step"], axis = 1)
    train_ans = train_set["Step"]
    
    test_set = df.loc[test_index].reset_index(drop=True)  #total
    test_dat = test_set.drop(columns=["Step"], axis = 1)
    test_ans = test_set["Step"] 

#Histogram
hist = train_set['Step'].hist()
hist.set_title("Step vs # of Data Points")
hist.set_xlabel("Step")
hist.set_ylabel("# of Data Points")


#Grouped Bar Chart - Grouped by avg val of coordinate for each step
grouped_avg_df = train_set.groupby('Step').mean() #groups dataframe by Step and the avg coordinates per step (XYZ)

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
grouped_by_step = train_set.groupby('Step')

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
#create figure of 4x4 and flatten dataset into 16x1 array for iteration 
#Correlation Matrix for XYZ Coordinates of Each Step
fig, hmp = mat.subplots(4, 4, figsize=(16,16))
hmp = hmp.flatten()

for i, (step, group) in enumerate(grouped_by_step):
    corr_matrix = group[['X', 'Y', 'Z']].corr(method='pearson') #pariwise corr. b/w colns. 
    hm_plot = hmp[i] #iterate through the 4x4 greater plot and place subplots.And below. 
    sea.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=hm_plot)
    hm_plot.set_title(f'Step {step}')
    
#cleans up unused plots on grid
for j in range(i + 1, len(hmp)):
    fig.delaxes(hmp[j])

mat.show()
"----------------------------------------------------------------------------"

"Step  4: Classification Model Development/Engineering"
"----------------------------------------------------------------------------"

#stratify training set again for cross validation
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 23)
#scoring = ['f1_macro', 'f1_micro', 'f1_weighted', 'accuracy', 'precision_macro', 'recall_macro']

"Model 1 - SVM w/ GridSearchCV"

#parameter grid for svm gridsearch
svm_grid = {
    'C': [0.01, 0.1, 1 ,10, 100], #regularization
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], #kernel types
    'gamma': ['scale'], #kernel coeff.
    'degree': [3, 4, 5, 6] #poly kernel degree
    }

#instance
svm = SVC()

#gridsearchCV for SVM
print("--SVM--")
svm_gridSearch = GridSearchCV(
    estimator = svm,
    param_grid = svm_grid,
    cv = cv,
    scoring = 'f1_macro', #improvement using f1_macro
    return_train_score = True, #diagnostics
    n_jobs = -1, #use all avilable CPU cores
    verbose = 1 #minimal
    )

svm_gridSearch.fit(train_dat, train_ans)

"Model 2 - Random Forest w/ GridSearchCV"

#param grid for RF gridsearch
rf_grid = {
    'n_estimators': [100, 200, 300], #total number of trees
    'max_depth': [None, 10, 20, 30], # tree depth
    'min_samples_split': [2, 5, 10], # min req. for splitting
    'min_samples_leaf': [1, 2, 4], # min req to be at a leaf node
    'criterion': ['gini', 'entropy'], #function for measuring qual of split
    'max_features': ['sqrt', 'log2'], #no. of features to consider for each split
    'bootstrap': [True, False] # use bootstrap samples?
    }

#instance 
rf = RandomForestClassifier(random_state = 23)

#gridsearchCV for RF
print("--RF--")
rf_gridSearch = GridSearchCV(
    estimator = rf,
    param_grid = rf_grid,
    cv = cv,
    scoring = 'f1_macro',
    return_train_score = True,
    n_jobs = -1,
    verbose = 1
)

rf_gridSearch.fit(train_dat, train_ans)

"Model 3 Decision Tree w/ GridSearchCV "

dt_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None],
    'splitter': ['best', 'random'] #splitting strategy
    }

#instance
dt = DecisionTreeClassifier(random_state = 23)

print("--DT--")
dt_gridSearch = GridSearchCV(
    estimator = dt,
    param_grid = dt_grid,
    cv = cv, 
    scoring = 'f1_macro',
    return_train_score = True,
    n_jobs = -1,
    verbose = 1
    )

dt_gridSearch.fit(train_dat, train_ans)

"Model 4 - Logistic Regression w/ RandomizedSearchCV"

lreg_grid = {
    'C': np.logspace(-4, 4, 20),  #regularization strength, testing across range of log spaced vals
    'penalty': ['l1', 'l2', 'elasticnet', 'none'], #the type of regularization applied
    'solver': ['saga'],  # optimization alg. saga supports above penalties
    'l1_ratio': np.linspace(0, 1, 10)  # mixing for elasticnet regularization
}

lreg = LogisticRegression(multi_class = 'multinomial', max_iter = 5000, random_state = 23) #5000 is set to avoid convergence warnings

print("--LOGISTIC REGRESSION--")
lreg_randoSearch = RandomizedSearchCV(
    estimator = lreg,
    param_distributions = lreg_grid,
    n_iter = 50,  #number of rando combos to try
    cv = cv,
    scoring = 'f1_macro', 
    random_state = 23, #this time its rando, so save rando state for reproducibility 
    n_jobs = -1,
    verbose = 1
)

lreg_randoSearch.fit(train_dat, train_ans)

"----------------------------------------------------------------------------"

"Step  5: Model Performance Analysis"
"----------------------------------------------------------------------------"

"----------------------------------------------------------------------------"

"Step  6: Stacked Model Performance Analysis"
"----------------------------------------------------------------------------"

"----------------------------------------------------------------------------"

"Step  7: Model Evaluation"
"----------------------------------------------------------------------------"

"----------------------------------------------------------------------------"
