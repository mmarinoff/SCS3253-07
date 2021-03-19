import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline


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

# import data
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

# split into x & y, drop ID column and target column from X
data_y = data['target']
X_data = data.drop(['target', 'ID_code'], axis=1)

# standard scale data
std = StandardScaler()
X_data = pd.DataFrame(std.fit_transform(X_data))

# pearson, spearman correlation coefficients
store = X_data.corr(method='pearson')
store.to_csv(os.path.join(dirname, 'data\\pearson.csv'))

# mean spread as a function of standard deviation
# mean_spread(X_data)




