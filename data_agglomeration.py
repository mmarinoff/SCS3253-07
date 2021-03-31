import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, KFold, GridSearchCV
from sklearn.cluster import FeatureAgglomeration, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics \
    import confusion_matrix, roc_curve, roc_auc_score, mean_squared_error, precision_recall_fscore_support, \
    plot_roc_curve, accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


def reduce_data():
    # import data
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    # split into x & y, drop ID column and target column from X
    data_y = data['target']
    X_data = data.drop(['target', 'ID_code'], axis=1)
    std = StandardScaler()
    X_data = pd.DataFrame(std.fit_transform(X_data))

    feature_cluster = FeatureAgglomeration(n_clusters=2)  # reduce dimensions to 20
    X_reduced = feature_cluster.fit_transform(X_data)  # reduce samples to

    X_1 = X_data[data_y == 1]
    X_0 = X_data[data_y == 0]

    plt.scatter(X_0.iloc[:, 0], X_0.iloc[:, 1])
    plt.scatter(X_1.iloc[:, 0], X_1.iloc[:, 1])
    plt.show()

    input('stop')

    mbatch = MiniBatchKMeans(n_clusters=25636, compute_labels=False)
    bkms = mbatch.fit(X_0)
    X_reduced = bkms.cluster_centers_
    X_reduced = pd.DataFrame(X_reduced)

    X_reduced.to_csv(os.path.join(dirname, 'data\\kmeans_reduced_0s.csv'))

    mbatch = MiniBatchKMeans(n_clusters=2864, compute_labels=False)
    bkms = mbatch.fit(X_1)
    X_reduced = bkms.cluster_centers_
    X_reduced = pd.DataFrame(X_reduced)

    X_reduced.to_csv(os.path.join(dirname, 'data\\kmeans_reduced_1s.csv'))


def apply_kpca():

    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_clusters.csv'), header=0)

    data_y = data['target']
    X_data = data.drop(['target', 'ID_code'], axis=1)

    feature_cluster = FeatureAgglomeration(n_clusters=10)  # reduce dimensions to 20
    X_reduced = feature_cluster.fit_transform(X_data)  # reduce samples to

    std = StandardScaler()
    X_data = pd.DataFrame(std.fit_transform(X_reduced))

    for i in range(0, 8):
        print(i)
        X_c = X_data[clusters.iloc[:, 1] == i]
        y_c = data_y[clusters.iloc[:, 1] == i]

        print(X_c.shape)
        print(y_c.shape)

        kpca = KernelPCA(kernel='rbf')
        X_data = kpca.fit_transform(X_c)

        X_data = pd.DataFrame(X_data)
        X_1 = X_data[data_y == 1]
        X_0 = X_data[data_y == 0]
        plt.scatter(X_0.iloc[:, 0], X_0.iloc[:, 1])
        plt.scatter(X_1.iloc[:, 0], X_1.iloc[:, 1])
        plt.show()
        #X_data.to_csv(os.path.join(dirname, 'data\\kpca_reduced.csv'))

def feature_reduction():
    # import data
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    # split into x & y, drop ID column and target column from X
    data_y = data['target']
    data_x = data.drop(['target', 'ID_code'], axis=1)

    # Feature Reduction
    data_minmax = MinMaxScaler().fit_transform(data_x)
    kbest = SelectKBest(chi2, k='all').fit(data_minmax, data_y)

    kbest_scores = pd.DataFrame(kbest.scores_)
    kbest_columns = kbest_scores >= 15
    data_x = data_x.iloc[:, kbest_columns.values.flatten()]
    print(data_x.shape)

    kpca = KernelPCA(n_components=2)
    X_data = kpca.fit_transform(data_x)

    X_data = pd.DataFrame(X_data)
    X_1 = X_data[data_y == 1]
    X_0 = X_data[data_y == 0]
    plt.scatter(X_0.iloc[:, 0], X_0.iloc[:, 1])
    plt.scatter(X_1.iloc[:, 0], X_1.iloc[:, 1])
    plt.show()

def feature_merge():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    # split into x & y, drop ID column and target column from X
    data_y = data['target']
    data_x = data.drop(['target', 'ID_code'], axis=1)

    std = StandardScaler()
    data_x = pd.DataFrame(std.fit_transform(data_x))

    data_x.iloc[:, -1] = data_x.abs().sum(axis=1)

    X_1 = data_x[data_y == 1]
    X_0 = data_x[data_y == 0]

    print(len(X_0.shape[0]*[0]))
    print(len(X_0.iloc[:, -1]))

    plt.scatter(X_0.shape[0]*[0], X_0.iloc[:, -1])
    plt.scatter(X_1.shape[0]*[1], X_1.iloc[:, -1])
    plt.show()

feature_merge()