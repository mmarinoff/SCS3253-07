# -*- coding: utf-8 -*-
"""Copy of PROJECT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17DHyX4-8TYBYWF7NezkhkjJlqcqnI2tn
"""

import os
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
import matplotlib as plt
#from metrics import plot_roc_curve, model_update_results

from google.colab import files
import pandas as pd
from google.colab import drive
drive.mount('/content/gdrive')

!gdown --id 1CLtos7Lpc82ufDrTtKtyC7dyUXeRztFm

#downloaded = drive.CreateFile({'id':id}) 
#downloaded.GetContentFile('train.csv')  
df = pd.read_csv("/content/gdrive/MyDrive/DS-PROJECT/bank/train.csv")

df.head()

df.head()

df.info

df.describe



#define target and predictors
y = df['target']
X = df.drop(['target', 'ID_code'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1364, stratify=y)
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
pca_training_X = pca.fit_transform(X_train)

#import the libraries for all the classifiers.

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#Design the random forest classifier

# Random Forest Classifier

n_estimators = [10, 200]
max_features = [0.1, 0.5]
max_depth = [2, 10, 20] 
oob_score = [True, False]
min_samples_split = [0.1, 0.5]
min_samples_leaf = [0.1, 0.5]
max_leaf_nodes = [2, 10, 100]

parameter_random_forest = {'n_estimators' : n_estimators, 'max_features' : max_features,
                     'max_depth' : max_depth, 'min_samples_split' : min_samples_split,
                    'oob_score' : oob_score, 'min_samples_leaf': min_samples_leaf, 
                     'max_leaf_nodes' : max_leaf_nodes}
             
Random_Forest_Classifier = RandomForestClassifier(random_state = 42)

#use grid search to tune the model

grid_search_RndmForest = GridSearchCV(Random_Forest_Classifier,parameter_random_forest, cv = 3, scoring='roc_auc', refit = True,
                                     n_jobs = -1, verbose=2)

grid_search_RndmForest.fit(pca_training_X,y_train)
             
forest_best_params_ = grid_search_RndmForest.best_params_
forest_best_estimators_ = grid_search_RndmForest.best_estimator_

print(forest_best_params_)
print(forest_best_estimators_)

