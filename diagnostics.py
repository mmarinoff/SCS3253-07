import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import LocallyLinearEmbedding
import math as m
import time
import matplotlib.pyplot as plt

def mean_spread(data_x):
    delta = []
    for i in range(0, data_x.shape[1]):
        x_0 = data_x.iloc[:, i][data_y == 0]
        x_1 = data_x.iloc[:, i][data_y == 1]
        x_i = data_x.iloc[:, i]

        x_0_mean = x_0.mean()
        x_1_mean = x_1.mean()
        x_i_mean = x_i.mean()

        delta.append(x_1_mean - x_0_mean)

    plt.plot(delta)
    plt.show()


def generate_pca_elbow_curve(data_x):
    x = np.array(range(1, 100, 1)) / 100
    pca_elbow = []

    for n in x:
        pca = PCA(n_components=n)
        pca_training_x = pca.fit_transform(data_x)
        print(n)
        print(pca_training_x.shape)
        pca_elbow.append(pca_training_x.shape[1])

    # last value is all dimensions for n=1
    pca_elbow = pd.DataFrame(pca_elbow)
    pca_elbow.to_csv(os.path.join(dirname, 'data\\pca_elbow_curve.csv'))


def show_pca_elbow_curve():
    pca_elbow = pd.read_csv(os.path.join(dirname, 'data\\pca_elbow_curve.csv'), header=0)
    print(pca_elbow)
    plt.plot(pca_elbow['N_Dimensions'], pca_elbow['Explained Variance'])
    plt.title("PCA Elbow Curve")
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.show()


def correlation_tables(data_x):

    # pearson, spearman correlation coefficients
    store = data_x.corr(method='pearson')
    store.to_csv(os.path.join(dirname, 'data\\pearson.csv'))

    # mean spread as a function of standard deviation
    # mean_spread(X_data)


# import data
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

# split into x & y, drop ID column and target column from X
data_y = data['target']
X_data = data.drop(['target', 'ID_code'], axis=1)
std = StandardScaler()
X_data = pd.DataFrame(std.fit_transform(X_data))
X_data = X_data.abs()

#generate_pca_elbow_curve(X_data)
show_pca_elbow_curve()

# lle = LocallyLinearEmbedding(n_components=2)
# X_new = lle.fit_transform(X_data, data_y)
# pd.DataFrame(X_new).plot()
# plt.show()
# print(lle.reconstruction_error_)

# start = time.time()
# lle.fit_transform(X_data, data_y)
# end = time.time()
#
# t = end - start
#
# print(lle.reconstruction_error_)
# N = 19044.0
# D = 35.0
# k = 5.0
# d = 5
#
# o = t/(D*m.log(k)*N*m.log(N) + D*N*k**3 + d*N**2)
# print(o)
#
# N = 100000
# D = 200
# k = 5.0
# d = 30
#
# t = o*(D*m.log(k)*N*m.log(N) + D*N*k**3 + d*N**2)
# print(t)




