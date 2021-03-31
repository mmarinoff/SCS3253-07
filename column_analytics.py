import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

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
    X_lesser = X_data.loc[:, variance > v_mean]
    X_greater = X_data.loc[:, variance > v_mean]

    return X_lesser, X_greater


def clustering(X_lesser, X_greater, data_y):

    #cluster data
    mbatch = MiniBatchKMeans(n_clusters=3, compute_labels=True, batch_size=1000)
    bkms = mbatch.fit_predict(X_lesser)
    bkms = pd.DataFrame(bkms)

    while True:
        random.randint(0, 200)
        a = random.randint(0, 199)
        b = random.randint(0, 199)

        for i in range(0, 2):

            X_c = X_lesser.loc[bkms.values == i]
            y_c = data_y.loc[bkms.values == i]

            X_1 = X_c[y_c == 1]
            X_0 = X_c[y_c == 0]

            plt.scatter(X_0.iloc[:, a], X_0.iloc[:, b], s=1)
            plt.scatter(X_1.iloc[:, a], X_1.iloc[:, b], s=1)

            plt.xlabel(a)
            plt.ylabel(b)
            plt.show()


# def plotting(X_data, data_y):
#
#         random.randint(0, 200)
#         a = random.randint(0, 199)
#         b = random.randint(0, 199)
#
#         X_1 = X_data[data_y == 1]
#         X_0 = X_data[data_y == 0]
#
#         plt.scatter(X_0.iloc[:, a], X_0.iloc[:, b], s=1)
#         plt.scatter(X_1.iloc[:, a], X_1.iloc[:, b], s=1)
#
#         plt.xlabel(a)
#         plt.ylabel(b)
#         plt.show()


data_x, data_y = import_data()
greater, lesser = process_data(data_x, data_y)
clustering(greater, lesser, data_y)
