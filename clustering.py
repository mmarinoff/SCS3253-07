from sklearn.cluster import OPTICS
import os
import pandas as pd
import numpy as np
from sklearn.cluster import FeatureAgglomeration, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

def import_data():

    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    # import data
    data_y = data['target']
    X_data = data.drop(['target', 'ID_code'], axis=1)

    return X_data, data_y


def process_data(X_data, data_y):

    # scale data
    std = StandardScaler()
    X_data = pd.DataFrame(std.fit_transform(X_data))

    # filter 1s from zeros
    X_0 = X_data[data_y == 0]
    X_1 = X_data[data_y == 1]

    # if mean 1 less than mean 0, flip sign so 1s always lean right
    mean_0 = X_0.mean(axis=0)
    mean_1 = X_1.mean(axis=0)

    # X_1_greater = X_data.loc[:, mean_1 > mean_0]
    X_1_lesser = X_data.loc[:, mean_1 < mean_0]

    X_data[X_1_lesser.columns] = np.negative(X_data[X_1_lesser.columns])

    X_data.iloc[:, -1] = X_data.sum(axis=1)

    # split high variance columns from low variance columns
    #X_0 = X_data[data_y == 0]
    X_1 = X_data[data_y == 1]

    variance = X_1.var(axis=0)
    v_mean = variance.mean()
    X_lesser = X_data.loc[:, variance < v_mean]
    X_greater = X_data.loc[:, variance > v_mean]

    return X_lesser, X_greater


def clustering(X_data):

    # dirname = os.path.dirname(__file__)
    # data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)
    #
    # data_y = data['target']
    # X_data = data.drop(['target', 'ID_code'], axis=1)
    #
    # std = StandardScaler()
    # X_data = pd.DataFrame(std.fit_transform(X_data))
    #
    # X_data.iloc[:, -1] = X_data.abs().sum(axis=1)
    #
    # for col in X_data.columns:
    #     X_data[col] = np.sign(X_data[col]) * X_data[col] ** 2
    #
    # X_data = pd.DataFrame(std.fit_transform(X_data))

    # X_data = X_data.abs()

    # feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    # X_reduced = feature_cluster.fit_transform(X_data)  # reduce samples to

    # X_data = pd.DataFrame(std.fit_transform(X_data))

    mbatch = MiniBatchKMeans(n_clusters=3, compute_labels=True)
    bkms = mbatch.fit_predict(X_data)
    bkms = pd.DataFrame(bkms)

    dirname = os.path.dirname(__file__)
    bkms.to_csv(os.path.join(dirname, 'data\\kmeans_clusters_lesser.csv'))

X_data, data_y = import_data()
lesser, greater = process_data(X_data, data_y)
clustering(lesser)


# dirname = os.path.dirname(__file__)
# clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_clusters2.csv'), header=0)
#
# print(data.iloc[:, 1])