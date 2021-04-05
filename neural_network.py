import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np
from random import choices

dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

data_y = data['target']
X_data = data.drop(['target', 'ID_code'], axis=1)

def process_data(X_data, data_y):

    # mean spread ____________________________________________________
    # filter 1s from zeros
    X_0 = X_data[data_y == 0]
    X_1 = X_data[data_y == 1]

    # if mean 1 less than mean 0, flip sign so 1s always lean right
    mean_0 = X_0.mean(axis=0)
    mean_1 = X_1.mean(axis=0)

    # X_1_greater = X_data.loc[:, mean_1 > mean_0]
    X_1_lesser = X_data.loc[:, mean_1 < mean_0]
    X_data[X_1_lesser.columns] = np.negative(X_data[X_1_lesser.columns])
    rowsum = X_data.sum(axis=1)

    # variance ---------------------------------------------------------
    X_1 = X_data[data_y == 1]

    var_1 = X_1.var(axis=0)

    X_var = X_data[X_data.columns[var_1.values > 1.019]]

    varsum = X_var.var(axis=1)

    metadata = pd.concat([varsum, rowsum], axis=1)

    std = StandardScaler()
    metadata = pd.DataFrame(std.fit_transform(metadata))

    return X_data, metadata


def nn(X_data, data_y, metadata):

    # feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    # X_data = feature_cluster.fit_transform(X_data)  # reduce samples to
    # X_data = pd.DataFrame(X_data)
    clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_meta.csv'), header=0)

    rundata = []
    columns = list(range(0, 200))
    X_data.columns = columns
    for i in range(2, 3):
        print(i)
        X_set = X_data.loc[:, [6, 12, 22, 26, 53, 76, 81, 110, 139, 146, 166, 174]]
        for l in range(0, X_set.shape[1] // 2):
            X_set.iloc[:, 2 * l] = X_set.iloc[:, 2 * i] * X_set.iloc[:, 2 * l + 1]
        X_set = X_data.loc[:, [6, 22, 53, 81, 139, 166]]
        X_set = pd.concat([X_set, metadata], axis=1)

        trans = PolynomialFeatures(degree=2)
        X_set = trans.fit_transform(X_set)

        std = StandardScaler()
        X_set = pd.DataFrame(std.fit_transform(X_set))

        X_c = X_set[clusters.iloc[:, 1] == i]
        y_c = data_y[clusters.iloc[:, 1] == i]

        print(X_c.shape)
        print(y_c.shape)

        # x_train, x_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.25, random_state=42, stratify=y_c)
        #
        # mlp = MLPClassifier(hidden_layer_sizes=5, learning_rate='adaptive', activation='relu', solver='adam', verbose=False,
        #                     alpha=0.001, tol=0.0001)
        # mlp.fit(x_train, y_train)
        # print(mlp.score(x_test, y_test))

        pipe = Pipeline(steps=[
            ('mlp', MLPClassifier())
        ])

        param_grid = {
            'mlp__hidden_layer_sizes': [2, 5, 10],
            'mlp__learning_rate': ['adaptive'],
            'mlp__solver': ['lbfgs'],
            'mlp__activation': ['relu', 'identity', 'logistic', 'tanh'],
            'mlp__alpha': [0.0001, 0.001, 0.01, 0.1]
        }
        pipe.fit(X_c, y_c)  # apply scaling on training data

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_c, y_c.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

        rundata.append([i, y_c.shape, search.best_score_, search.best_params_])

    print(rundata)


    # Best parameter (CV score=0.918):
    # {'mlp__activation': 'identity', 'mlp__alpha': 0.0001, 'mlp__hidden_layer_sizes': 10, 'mlp__learning_rate': 'adaptive', 'mlp__solver': 'lbfgs'}
    # 0.91668
    #
    #     mlp = MLPClassifier(hidden_layer_sizes=10, learning_rate='adaptive', activation='identity', solver='lbfgs', verbose=False,
    #                         alpha=0.0001, tol=0.0001)
    #     mlp.fit(x_train, y_train)
    #     print(mlp.score(x_test, y_test))

    # Processing Pipeline

    # pipe = Pipeline(steps=[
    #     ('mlp', MLPClassifier())
    # ])
    #
    # param_grid = {
    #     'mlp__hidden_layer_sizes': [2, 5, 10],
    #     'mlp__learning_rate': ['adaptive'],
    #     'mlp__solver': ['lbfgs'],
    #     'mlp__activation': ['relu', 'identity', 'logistic', 'tanh'],
    #     'mlp__alpha': [0.0001, 0.001, 0.01, 0.1]
    # }
    # pipe.fit(x_train, y_train)  # apply scaling on training data
    #
    # # Performance Metrics
    # search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
    # search.fit(x_train, y_train.ravel())
    # print("Best parameter (CV score=%0.3f):" % search.best_score_)
    # print(search.best_params_)
    #
    # y_pred = search.predict(x_test)  # test predictions
    #
    # pipe = Pipeline(steps=[
    #     ('mlp', MLPClassifier())
    # ])
    #
    # param_grid = {
    #     'mlp__hidden_layer_sizes': [2, 10],
    #     'mlp__learning_rate': ['adaptive'],
    #     'mlp__solver': ['lbfgs', 'sgd', 'adam'],
    #     'mlp__activation': ['relu', 'identity', 'logistic', 'tanh'],
    #     'mlp__alpha': [0.0001, 0.01]
    # }
    # pipe.fit(X_c, y_c)  # apply scaling on training data
    #
    # # Performance Metrics
    # search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=True)
    # search.fit(X_c, y_c.ravel())
    # print("Best parameter (CV score=%0.3f):" % search.best_score_)
    # print(search.best_params_)
X_data, metadata = process_data(X_data, data_y)
nn(X_data, data_y, metadata)