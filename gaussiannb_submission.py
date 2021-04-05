import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from utils import import_data, process_data, model_update_results, import_clusters
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

X_train, y_train, X_test = import_data()

kbest = SelectKBest(f_classif, k='all').fit(X_train, y_train)

kbest_scores = pd.DataFrame(kbest.scores_)
kbest_columns = kbest_scores >= 150
X_train = X_train.iloc[:, kbest_columns.values.flatten()]
X_test = X_test.iloc[:, kbest_columns.values.flatten()]


gnb = GaussianNB()
gnb.fit(X_train, y_train)
test_pred = gnb.predict(X_test)
print(test_pred.sum())
input('stop')
test_proba = gnb.predict_proba(X_test)
test_proba = pd.DataFrame(test_proba)
result = (test_proba.iloc[:, 1] > 0.1)
result = result.astype(int)

plt.show()

# result = pd.DataFrame(test_pred)

dirname = os.path.dirname(__file__)
result.to_csv(os.path.join(dirname, 'data\\results.csv'))
