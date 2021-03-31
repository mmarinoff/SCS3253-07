import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

data_y = data['target']
X_data = data.drop(['target', 'ID_code'], axis=1)

std = StandardScaler()
X_data = pd.DataFrame(std.fit_transform(X_data))

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

rowsum = pd.DataFrame(X_data.sum(axis=1))

X_1 = X_data[data_y == 1]

var_1 = X_1.var(axis=0)

X_data = X_data[X_data.columns[var_1.values > 1.019]]

varsum = X_data.var(axis=1)


varsum_1 = varsum[data_y == 1]
varsum_0 = varsum[data_y == 0]
rowsum_1 = rowsum[data_y == 1]
rowsum_0 = rowsum[data_y == 0]

plt.scatter(varsum_0, rowsum_0, s=1)
plt.scatter(varsum_1, rowsum_1, s=1)

plt.show()
