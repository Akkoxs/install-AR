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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import joblib

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
print("SVM Done!")
print("Best SVM params:", svm_gridSearch.best_params_)

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
print("RandomForest Done!")
print("Best RF params:", rf_gridSearch.best_params_)

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
print("DecisionTree Done!")
print("Best DT params:", dt_gridSearch.best_params_)

"Model 4 - Logistic Regression w/ RandomizedSearchCV"

lreg_grid = {
    'C': np.logspace(-4, 4, 20),  #regularization strength, testing across range of log spaced vals
    'penalty': ['l1', 'l2', 'elasticnet', 'none'], #the type of regularization applied
    'solver': ['saga'],  # optimization alg. saga supports above penalties
    'l1_ratio': np.linspace(0, 1, 10)  # mixing for elasticnet regularization
}

lreg = LogisticRegression(max_iter = 5000, random_state = 23) #5000 is set to avoid convergence warnings

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
print("LogisticRegression Done!")
print("Best LReg params:", lreg_randoSearch.best_params_)

"----------------------------------------------------------------------------"

"Step  5: Model Performance Analysis"
"----------------------------------------------------------------------------"
all_models = {
    "SVM": svm_gridSearch.best_estimator_,
    "Random Forest": rf_gridSearch.best_estimator_,
    "Decision Tree": dt_gridSearch.best_estimator_,
    "Logistic Regression": lreg_randoSearch.best_estimator_
    }

#storage loc for all eval results
results = {}

for name, model in all_models.items():
    #generate predictions based on test data 
    preds = model.predict(test_dat)
    
    #metrics
    f1 = f1_score(test_ans, preds, average='macro')
    precision = precision_score(test_ans, preds, average='macro')
    accuracy = accuracy_score(test_ans, preds)
    mae = mean_absolute_error(test_ans, preds)
    
    results[name] = {
        "F1 Score (macro)": f1,
        "Precision (macro)": precision,
        "Accuracy": accuracy,
        "Mean Absolute Error": mae,
        "Predictions": preds 
    }

#visualize results
print("Model Performance Summary:")
perf_df = pd.DataFrame(results).T.drop(columns="Predictions") #drop just for visualization
print(perf_df)


#confusion matrix creation
fig, axes = mat.subplots(1, len(all_models), figsize=(20, 5))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(test_ans, res["Predictions"], labels=sorted(test_ans.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(test_ans.unique()))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name)

mat.show()

"----------------------------------------------------------------------------"

"Step  6: Stacked Model Performance Analysis"
"----------------------------------------------------------------------------"
print("STACKED MODEL: SVM + RF")

stacked_base_models = [
    ('svm', svm_gridSearch.best_estimator_),
    ('rf', rf_gridSearch.best_estimator_)
]

#using lreg "meta" classifier
stacked_meta_model = LogisticRegression(multi_class='multinomial', max_iter=5000, random_state=23) 

#stacking classifier def
stacked_model = StackingClassifier(
    estimators = stacked_base_models,
    final_estimator = stacked_meta_model,
    cv = cv,
    n_jobs = -1
)

stacked_model.fit(train_dat, train_ans)

#generate predictions
stacked_preds = stacked_model.predict(test_dat)

#metrics 
stacked_f1 = f1_score(test_ans, stacked_preds, average='macro')
stacked_precision = precision_score(test_ans, stacked_preds, average='macro')
stacked_accuracy = accuracy_score(test_ans, stacked_preds)
stacked_mae = mean_absolute_error(test_ans, stacked_preds)

# output performance 
print("Stacked Model Performance Metrics:")
print(f"F1 Score (macro):      {stacked_f1:.4f}")
print(f"Precision (macro):     {stacked_precision:.4f}")
print(f"Accuracy:              {stacked_accuracy:.4f}")

#save to same results dictionary
results["Stacked Model"] = {
    "F1 Score (macro)": stacked_f1,
    "Precision (macro)": stacked_precision,
    "Accuracy": stacked_accuracy,
    "Mean Absolute Error": stacked_mae,
    "Predictions": stacked_preds
    }


#confusion matrix
stacked_cm = confusion_matrix(test_ans, stacked_preds, labels = sorted(test_ans.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix = stacked_cm, display_labels = sorted(test_ans.unique()))
disp.plot(cmap='Purples', colorbar=False)
disp.ax_.set_title("Stacked Model Confusion Matrix")

mat.show()

"----------------------------------------------------------------------------"

"Step  7: Model Evaluation"
"----------------------------------------------------------------------------"
 
#save in joblib format
joblib.dump(stacked_model, "stacked_model.joblib")
    
new_dat = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
],
    columns=["X", "Y", "Z"])

#generate new prediction based on new data given & add to new df
new_preds = stacked_model.predict(new_dat)
new_dat["Predicted Step"] = new_preds

print("----FINAL----")
print("Predicted Step for New Data:")
print(new_dat)

"----------------------------------------------------------------------------"
