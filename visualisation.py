import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

data_y = data['target']
X_data = data.drop(['target', 'ID_code'], axis=1)

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

while True:

    random.randint(0, 200)
    a = random.randint(0, 199)
    b = random.randint(0, 199)

    X_1 = X_data[data_y == 1]
    X_0 = X_data[data_y == 0]

    plt.scatter(X_0.iloc[:, -1], X_0.iloc[:, 1], s=1)
    plt.scatter(X_1.iloc[:, -1], X_1.iloc[:, 1], s=1)

    plt.xlabel(a)
    plt.ylabel(b)
    plt.show()


#
# for i in range(0, X_data.shape[1]-1):
#     X_data.iloc[:, i] = X_data.iloc[:, i]*X_data.iloc[:, i+1]
# X_data.iloc[:, -1] = X_data.iloc[:, -1]*X_data.iloc[:, 0]
#
# for col in X_data.columns:
#     X_data[col] = np.sign(X_data[col])*X_data[col]**2
#
# X_data = pd.DataFrame(std.fit_transform(X_data))
# X_data = X_data.abs()
#
# X_data = pd.DataFrame(std.fit_transform(X_data))
#
clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_clusters_lesser.csv'), header=0)

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

while True:

    random.randint(0, 200)
    a = random.randint(0, 199)
    b = random.randint(0, 199)

    for i in range(0, 3):
        print(i)
        X_c = X_data[clusters.iloc[:, 1] == i]

        X_1 = X_c[data_y == 1]
        X_0 = X_c[data_y == 0]

        print(X_0.shape[0])
        print(X_1.shape[0])
        print(X_1.shape[0]/X_0.shape[0])

        # if mean 1 less than mean 0, flip sign so 1s always lean right
        # mean_0 = X_0.mean(axis=0)
        # mean_1 = X_1.mean(axis=0)
        # v_0 = X_0.var(axis=0)
        # v_1 = X_1.var(axis=0)
        #
        # print(mean_0-mean_1)
        # print(v_0-v_1)

        plt.scatter(X_0.iloc[:, a], X_0.iloc[:, b], s=1)
        plt.scatter(X_1.iloc[:, a], X_1.iloc[:, b], s=1)

        plt.xlabel(a)
        plt.ylabel(b)
        plt.show()


# while True:
#
#     random.randint(0, 200)
#     a = random.randint(0, 199)
#     b = random.randint(0, 199)
#
#     X_1 = X_c[data_y == 1]
#     X_0 = X_c[data_y == 0]
#
#
#     plt.scatter(X_0.iloc[:, a], X_0.iloc[:, b], s=1)
#     plt.scatter(X_1.iloc[:, a], X_1.iloc[:, b], s=1)
#
#     plt.xlabel(a)
#     plt.ylabel(b)
#     plt.show()