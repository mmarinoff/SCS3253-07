import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
import matplotlib.pyplot as plt

# best results 'SAMME.R', learning_rate': 0.5, n_estimators': 100}
# accuracy 0.9059649122807018

# import data
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\kmeans_reduced.csv'), header=0)

# split into x & y, drop ID column and target column from X
data_y = data['target']
data_x = data.drop(['target', 'ID_code'], axis=1)

# std = StandardScaler()
# data_x = pd.DataFrame(std.fit_transform(data_x))
#
# feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
# data_x = feature_cluster.fit_transform(data_x)  # reduce samples to

# Test/Train Split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=42, stratify=data_y)

# Processing Pipeline

pipe = Pipeline(steps=[
    # ('stdscale', StandardScaler())
    ('adaboost', AdaBoostClassifier())
])

param_grid = {
    'adaboost__n_estimators': [50, 100],
    'adaboost__algorithm': ['SAMME', 'SAMME.R'],
    'adaboost__learning_rate': [0.1, 0.5, 0.7, 1, 1.2, 1.5],
}

pipe.fit(x_train, y_train)  # apply scaling on training data

# Performance Metrics
search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
search.fit(x_train, y_train.ravel())
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

y_pred = search.predict(x_test)  # test predictions

acc = accuracy_score(y_test, y_pred)
print(acc)
y_proba = search.predict_log_proba(x_test)  # test probabilities
precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred)
print(precision, recall, fbeta)

# model_update_results(mse, auc, score, precision[1], recall[1], fbeta[1], modelname='Baseline Logistic Regression')
# model_update_results(mse, auc, score, precision[1], recall[1], fbeta[1], ID=1)
