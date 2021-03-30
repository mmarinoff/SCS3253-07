import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# best result 0.90574, clf1-4 w/ hard
# 0.90834 c1, cl3, cl4 soft
# import data
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

# split into x & y, drop ID column and target column from X
data_y = data['target']
data_x = data.drop(['target', 'ID_code'], axis=1)

# Feature Reduction
data_minmax = MinMaxScaler().fit_transform(data_x)
kbest = SelectKBest(chi2, k=20).fit(data_minmax, data_y)

kbest_scores = pd.DataFrame(kbest.scores_)
kbest_columns = kbest_scores >= 15
data_x = data_x.iloc[:, kbest_columns.values.flatten()]

# Feature Agglo
# feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
# data_x = feature_cluster.fit_transform(data_x)  # reduce samples to

# scale data
std = StandardScaler()
data_x = pd.DataFrame(std.fit_transform(data_x))

# Test/Train Split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=42, stratify=data_y)

clf1 = LogisticRegression(solver='sag', C=0.7)
clf2 = RandomForestClassifier(ccp_alpha=0, criterion='gini', max_features='sqrt', min_samples_split=5)
clf3 = AdaBoostClassifier(algorithm='SAMME.R', learning_rate=0.5, n_estimators=100)
clf4 = GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.1, min_samples_leaf=2,
                                  min_samples_split=10, n_estimators=200, subsample=0.7)

vote = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf3), ('rf', clf4)], voting='soft')  # ('rf', clf2)
vote.fit(x_train, y_train)

y_pred = vote.predict(x_test)  # test predictions

acc = accuracy_score(y_test, y_pred)
print(acc)
precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred)
print(precision, recall, fbeta)

# model_update_results(mse, auc, score, precision[1], recall[1], fbeta[1], modelname='Baseline Logistic Regression')
# model_update_results(mse, auc, score, precision[1], recall[1], fbeta[1], ID=1)