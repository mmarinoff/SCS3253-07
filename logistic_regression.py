import os
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

data_y = data['target']
data_x = data.drop(['target', 'ID_code'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=42, stratify=data_y)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(x_train, y_train)  # apply scaling on training data
print(pipe.score(x_test, y_test))  # apply scaling on testing data, without leaking training data.

y_pred = pipe.predict(x_test)

conf = confusion_matrix(y_test, y_pred)

print(conf)

