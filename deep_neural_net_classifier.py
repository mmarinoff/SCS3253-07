import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from tensorflow import keras  # tf.keras


# loads the training data
def loadTrainingData():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)
    return data;

# loads the test data
def loadTestData():
    dirname = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(dirname, 'data\\test.csv'), header=0)
    return data;

# Merge features to reduce them
def feature_reduction(X_data, numberOfClusters):
    feature_cluster = FeatureAgglomeration(n_clusters=numberOfClusters)  # reduce dimensions to 20
    X_reduced = feature_cluster.fit_transform(X_data)  # reduce samples to
    return X_reduced

# Create the neural network
def createModel(numberOfColumns) :
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(input_dim=numberOfColumns, activation='relu'))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    # softmax for binary classifier
    model.add(keras.layers.Dense(2, activation="softmax"))
    return model;

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.show()

print("Loading Data");
training_data = loadTrainingData();
testing_data = loadTestData();

y_train = training_data['target'];
X_train = training_data.drop(['target', 'ID_code'], axis=1);

y_test = testing_data['target'];
X_test = testing_data.drop(['target', 'ID_code'], axis=1);

numberOfClusters = 200;
#print("Reducing Features");
#X_train_reduced = feature_reduction(X_train, numberOfClusters);
#X_test_reduced = feature_reduction(X_test, numberOfClusters);
#print(X_train_reduced);

print("Creating Neural Net");
model = createModel(numberOfClusters);
print(model.layers)
print(model.summary())

#sparse_categorical_crossentropy
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=["accuracy"])

print("Training")
history = model.fit(X_train, y_train, epochs=10)
#plot_learning_curves(history)

print("Evaluating Test Data")
result = model.evaluate(X_test, y_test)
print(result)