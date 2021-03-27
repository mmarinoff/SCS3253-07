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


def reduce_data():
    # import data
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    # split into x & y, drop ID column and target column from X
    data_y = data['target']
    X_data = data.drop(['target', 'ID_code'], axis=1)
    std = StandardScaler()
    X_data = pd.DataFrame(std.fit_transform(X_data))

    feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    X_reduced = feature_cluster.fit_transform(X_data)  # reduce samples to

    data_zeros = X_reduced[data_y == 0]  # split 1s from 0s
    data_ones = X_reduced[data_y == 1]

    mbatch = MiniBatchKMeans(n_clusters=25636, compute_labels=False)
    bkms = mbatch.fit(data_zeros)
    X_reduced = bkms.cluster_centers_
    X_reduced = pd.DataFrame(X_reduced)

    X_reduced.to_csv(os.path.join(dirname, 'data\\kmeans_reduced_0s.csv'))

    mbatch = MiniBatchKMeans(n_clusters=2864, compute_labels=False)
    bkms = mbatch.fit(data_ones)
    X_reduced = bkms.cluster_centers_
    X_reduced = pd.DataFrame(X_reduced)

    X_reduced.to_csv(os.path.join(dirname, 'data\\kmeans_reduced_1s.csv'))


def apply_kpca():

    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\kmeans_reduced.csv'), header=0)

    data_y = data['target']
    X_data = data.drop(['target', 'ID_Code'], axis=1)

    kpca = KernelPCA(n_components=10, kernel='poly')
    X_data = kpca.fit_transform(X_data)

    X_data = pd.DataFrame(X_data)
    X_data.to_csv(os.path.join(dirname, 'data\\kpca_reduced.csv'))





