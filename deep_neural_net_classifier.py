import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from tensorflow import keras  # tf.keras


# loads the training data
def loadTrainingData():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)
    return data

# loads the test data
def loadTestData():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\test.csv'), header=0)
    return data

# Merge features to reduce them
def feature_reduction(X_data, numberOfClusters):
    feature_cluster = FeatureAgglomeration(n_clusters=numberOfClusters)  # reduce dimensions to 20
    X_reduced = feature_cluster.fit_transform(X_data)  # reduce samples to
    return X_reduced

# Create the neural network
def createModel(numberOfColumns) :
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=numberOfColumns * 1.5, input_dim=numberOfColumns, activation='relu'))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    # softmax for binary classifier
    model.add(keras.layers.Dense(2, activation="softmax"))
    return model

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.show()

print("Loading Data");
training_data = loadTrainingData()

y_train_full = training_data['target']
X_train_full = training_data.drop(['target', 'ID_code'], axis=1)

# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2)

print("Reducing Features");
pca = PCA(n_components=0.95)
pca_training_X = pca.fit_transform(X_train)
pca_test_X = pca.fit_transform(X_test)

numberOfFeatures = 200

print("Creating Neural Net")
model = createModel(numberOfFeatures)
print(model.layers)
print(model.summary())

#sparse_categorical_crossentropy
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=["accuracy"])

print("Training")
history = model.fit(X_train, y_train, epochs=10)
#plot_learning_curves(history)

#print("Evaluating Test Data")
result = model.evaluate(X_test, y_test)
print(result)