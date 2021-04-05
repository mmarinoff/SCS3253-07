import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# loads the training data
def loadTrainingData():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)
    return data

training_data = loadTrainingData()
std = StandardScaler()
training_data = pd.DataFrame(std.fit_transform(training_data))
filtered_data_0 = training_data[training_data['target'] == 0]
filtered_data_1 = training_data[training_data['target'] == 1]

for i in range(0, 1):
    #var_0_0 = filtered_data_0['var_' + str(i)]
    #var_0_1 = filtered_data_1['var_' + str(i)]
    #ax = sns.violinplot(x=filtered_data_0['var_' + str(i)])
    ax = sns.violinplot(x=filtered_data_1['var_' + str(i)])
    plt.show()
