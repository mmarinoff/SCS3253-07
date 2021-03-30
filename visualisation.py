import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

data_y = data['target']
X_data = data.drop(['target', 'ID_code'], axis=1)

std = StandardScaler()
X_data = pd.DataFrame(std.fit_transform(X_data))

for col in X_data.columns:
    X_data[col] = np.sign(X_data[col])*X_data[col]**2


std = StandardScaler()
X_data = pd.DataFrame(std.fit_transform(X_data))
# X_data = X_data.abs()

clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_clusters.csv'), header=0)


while True:

    random.randint(0, 200)
    a = random.randint(0, 199)
    b = random.randint(0, 199)

    for i in range(0, 8):
        print(i)
        X_c = X_data[clusters.iloc[:, 1] == i]

        X_1 = X_c[data_y == 1]
        X_0 = X_c[data_y == 0]

        print(X_0.shape[0])
        print(X_1.shape[0])
        print(X_0.shape[0]/X_1.shape[0])

        plt.scatter(X_0.iloc[:, a], X_0.iloc[:, b], s=1)
        plt.scatter(X_1.iloc[:, a], X_1.iloc[:, b], s=1)

        plt.xlabel(a)
        plt.ylabel(b)
        plt.show()
