from sklearn.cluster import OPTICS
import os
import pandas as pd
from sklearn.cluster import FeatureAgglomeration, MiniBatchKMeans


def clustering():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    data_y = data['target']
    X_data = data.drop(['target', 'ID_code'], axis=1)

    feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    X_reduced = feature_cluster.fit_transform(X_data)  # reduce samples to

    mbatch = MiniBatchKMeans(n_clusters=8, compute_labels=True)
    bkms = mbatch.fit_predict(X_data)
    bkms = pd.DataFrame(bkms)

    bkms.to_csv(os.path.join(dirname, 'data\\kmeans_clusters.csv'))

dirname = os.path.dirname(__file__)
clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_clusters.csv'), header=0)

print(data.iloc[:, 1])