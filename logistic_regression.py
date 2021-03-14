import os
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics \
    import confusion_matrix, roc_curve, roc_auc_score, mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from metrics import plot_roc_curve, model_update_results


# import data
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

# split into x & y, drop ID column and target column from X
data_y = data['target']
data_x = data.drop(['target', 'ID_code'], axis=1)

# Test/Train Split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=42, stratify=data_y)

# Processing Pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(x_train, y_train)  # apply scaling on training data

# Performance Metrics
y_pred = pipe.predict(x_test)  # test predictions
y_proba = pipe.predict_log_proba(x_test)  # test probabilities
conf = confusion_matrix(y_test, y_pred)  # confusion matrix
score = pipe.score(x_test, y_test)
precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred)
fpr_log, tpr_log, thresholds = roc_curve(y_test, y_proba[:, 1])  # ROC Curve Values
auc = roc_auc_score(y_test, y_proba[:, 1])  # ROC AUC score
mse = mean_squared_error(y_test, y_pred)

# Plot ROC Curve
plot_roc_curve(fpr_log, tpr_log, "Logistic Regression")
plt.annotate('ROC AOC Score: ' + str(round(auc, 2)), xy=(0.6, 0.2))
# plt.show()
# print(pipe.score(x_test, y_test))  # accuracy
# print(conf)
# print(auc)
# print(mse)
# print(precision, recall, fbeta)

# model_update_results(mse, auc, score, precision[1], recall[1], fbeta[1], modelname='Baseline Logistic Regression')
model_update_results(mse, auc, score, precision[1], recall[1], fbeta[1], ID=1)


