import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from utils import import_data, process_data, import_clusters
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def plot_random_feature(X_data, data_y):
    while True:
        random.randint(0, 200)
        a = random.randint(0, 199)
        b = random.randint(0, 199)

        X_1 = X_data[data_y == 1]
        X_0 = X_data[data_y == 0]

        plt.scatter(X_0.iloc[:, a], X_0.iloc[:, b], s=1)
        plt.scatter(X_1.iloc[:, a], X_1.iloc[:, b], s=1)

        plt.xlabel(a)
        plt.ylabel(b)
        plt.show()


def plot_metafeatures(meta_train, train_y, meta_test, test=False):
    i_1 = [1, 2, 0]

    for i in range(0, 3):

        X_1 = meta_train[train_y == 1]
        X_0 = meta_train[train_y == 0]

        plt.scatter(X_0.iloc[:, i], X_0.iloc[:, i_1[i]], s=1)
        plt.scatter(X_1.iloc[:, i], X_1.iloc[:, i_1[i]], s=1)

        plt.xlabel(meta_train.columns[i])
        plt.ylabel(meta_train.columns[i_1[i]])

        if test:
            plt.scatter(meta_test.iloc[:, i], meta_test.iloc[:, i_1[i]], s=1)

            plt.xlabel(meta_test.columns[i])
            plt.ylabel(meta_test.columns[i_1[i]])

        plt.show()


def plot_cluster_histogram(X_data, data_y, n_cluster=2):

    clusters = import_clusters()

    columns = list(range(0, 200))
    X_data.columns = columns

    X_0 = X_train[y_train == 0]
    X_1 = X_train[y_train == 1]

    for i in range(0, n_cluster):

        X_c = X_data[clusters.iloc[:, 1] == n_cluster]
        X_c = X_c.loc[:, [6, 12, 22, 26, 53, 76, 81, 110, 139, 146, 166, 174]]

        colsave = []
        for col in X_c.columns:
            h0, bin0 = np.histogram(X_0.iloc[:, col])
            h1, bin1 = np.histogram(X_1.iloc[:, col], bin0)

            delta_spread = ((h1 / h0) - [X_1.shape[0] / X_0.shape[0]] * 10) * h1
            print(sum(delta_spread[5:]) - sum(delta_spread[:5]))
            if abs(sum(delta_spread[5:]) - sum(delta_spread[:5])) > 400:
                colsave.append(col)
        print(colsave)
        input('stop')


def plot_clusters(X_train, data_y, X_test, n_clusters=3, test=False):

    clusters_train, clusters_test = import_clusters()

    columns = list(range(0, 200))
    X_train.columns = columns

    while True:
        random.randint(0, 200)
        a = random.randint(0, 199)
        b = random.randint(0, 199)

        for i in range(0, n_clusters):
            print(i)
            print('-----')
            X_c = X_train[clusters_train.iloc[:, 1] == i]

            X_1 = X_c[data_y == 1]
            X_0 = X_c[data_y == 0]

            print(X_0.shape[0])
            print(X_1.shape[0])
            print(X_1.shape[0]/X_0.shape[0])

            h0, bin0 = np.histogram(X_0.iloc[:, a])
            h1, bin1 = np.histogram(X_1.iloc[:, a], bin0)
            print(bin0)
            print(bin1)
            print(h1)
            print(h0)
            print(h1/h0)
            delta_spread = ((h1/h0)-[X_1.shape[0]/X_0.shape[0]]*10)*h1
            print(sum(delta_spread[5:])- sum(delta_spread[:5]))
            #print((X_1.shape[0]/X_0.shape[0] - np.var(h1/h0))*h1)
            plt.show()

            plt.scatter(X_0.iloc[:, a], X_0.iloc[:, b], s=1)
            plt.scatter(X_1.iloc[:, a], X_1.iloc[:, b], s=1)

            plt.xlabel(a)
            plt.ylabel(b)
            if test:
                plt.figure()
                X_tc = X_test[clusters_test.iloc[:, 1] == i]
                plt.scatter(X_tc.iloc[:, a], X_tc.iloc[:, b], s=1)

            plt.show()


def mean_spread(data_x, data_y):
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


def generate_pca_elbow_curve(X_data):
    columns = list(range(0, 200))
    X_data.columns = columns
    data_x = X_data.loc[:, [6, 12, 22, 26, 53, 76, 81, 110, 139, 146, 166, 174]]
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
    dirname = os.path.dirname(__file__)
    pca_elbow.to_csv(os.path.join(dirname, 'data\\pca_elbow_curve.csv'))


def show_pca_elbow_curve():
    dirname = os.path.dirname(__file__)
    pca_elbow = pd.read_csv(os.path.join(dirname, 'data\\pca_elbow_curve.csv'), header=0)
    print(pca_elbow)
    plt.plot(pca_elbow['N_Dimensions'], pca_elbow['Explained Variance'])
    plt.title("PCA Elbow Curve")
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.show()


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)


def correlation_tables(data_x):

    # pearson, spearman correlation coefficients
    store = data_x.corr(method='pearson')
    dirname = os.path.dirname(__file__)
    store.to_csv(os.path.join(dirname, 'data\\pearson.csv'))

    # mean spread as a function of standard deviation
    # mean_spread(X_data)


# import data from csv
X_train, y_train, X_test = import_data()

# standard scale, create metafeatures
meta_train, meta_test = process_data(X_train, y_train, X_test)


# plot_random_feature(X_train, y_train)

# plot_metafeatures(meta_train, y_train, meta_test, test=True)

plot_clusters(X_train, y_train, X_test, n_clusters=2, test=True)
