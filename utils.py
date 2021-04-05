import pandas as pd
import numpy as np
import os
import shutil
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from tempfile import NamedTemporaryFile


def import_data():
    # Import Data
    dirname = os.path.dirname(__file__)
    train_data = pd.read_csv(os.path.join(dirname, 'data\\train.csv'), header=0)

    test_data = pd.read_csv(os.path.join(dirname, 'data\\test.csv'), header=0)

    y_train = train_data['target']
    X_train = train_data.drop(['target', 'ID_code'], axis=1)

    X_test = test_data.drop(['ID_code'], axis=1)

    std = StandardScaler()
    X_train = pd.DataFrame(std.fit_transform(X_train))
    X_test = pd.DataFrame(std.transform(X_test))

    return X_train, y_train, X_test


def import_clusters():
    dirname = os.path.dirname(__file__)
    train_clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_clusters_train.csv'), header=0)
    test_clusters = pd.read_csv(os.path.join(dirname, 'data\\kmeans_clusters_test.csv'), header=0)
    return train_clusters, test_clusters


def process_data(X_train, y_train, X_test):

    # split X dataset where y=0 from y=1
    X_0 = X_train[y_train == 0]
    X_1 = X_train[y_train == 1]

    # mean ------------------------------------------------------------------
    # find mean of all columns for X=1, X=0 subsets
    mean_0 = X_0.mean(axis=0)
    mean_1 = X_1.mean(axis=0)

    # remove any columns where mean spread is not statistically significant
    mean_delta = abs(abs(mean_1) - abs(mean_0))
    mean_delta_filter = mean_delta > 0.025
    train_meanspread = X_train.loc[:, mean_delta_filter]
    test_meanspread = X_test.loc[:, mean_delta_filter]
    mean_0 = mean_0[mean_delta_filter]
    mean_1 = mean_1[mean_delta_filter]

    # invert signs on any column where mean 0 > mean 1 so result of mean_0 - mean_1 is always positive
    X_1_lesser = train_meanspread.loc[:, mean_1 < mean_0]
    train_meanspread[X_1_lesser.columns] = np.negative(train_meanspread[X_1_lesser.columns])
    test_meanspread[X_1_lesser.columns] = np.negative(test_meanspread[X_1_lesser.columns])

    # row sum each datapoint across all significant
    rowsum = train_meanspread.sum(axis=1)
    rowsum_test = test_meanspread.sum(axis=1)

    # variance ---------------------------------------------------------
    var_1 = X_1.var(axis=0)

    # remove any columns where variance spread is not statistically significant
    X_var = X_train[X_train.columns[var_1.values > 1.03]]
    X_var_test = X_test[X_train.columns[var_1.values > 1.03]]

    varsum = X_var.var(axis=1)
    varsum_test = X_var_test.var(axis=1)
    # median ------------------------------------------------------------

    # find median of all columns for X=1, X=0 subsets
    median_0 = X_0.median(axis=0)
    median_1 = X_1.median(axis=0)

    # remove any columns where mean spread is not statistically significant
    median_delta = abs(abs(median_1) - abs(median_0))
    med_delta_filter = median_delta > 0.075
    train_medspread = X_train.loc[:, med_delta_filter]
    test_medspread = X_test.loc[:, med_delta_filter]
    med_0 = median_0[med_delta_filter]
    med_1 = median_1[med_delta_filter]

    # invert signs on any column where med 0 > med 1 so result of med_0 - med_1 is always positive
    X_1_lesser = train_medspread.loc[:, med_1 < med_0]
    train_medspread[X_1_lesser.columns] = np.negative(train_medspread[X_1_lesser.columns])
    test_medspread[X_1_lesser.columns] = np.negative(test_medspread[X_1_lesser.columns])

    medsum = train_medspread.sum(axis=1)
    medsum_test = test_medspread.sum(axis=1)

    metadata_train = pd.concat([varsum, rowsum, medsum], axis=1)
    metadata_test = pd.concat([varsum_test, rowsum_test, medsum_test], axis=1)
    metadata_test.columns = ['Variation', 'Mean', 'Median']

    # std = StandardScaler()
    # metadata_train = pd.DataFrame(std.fit_transform(metadata_train), columns=['Variation', 'Mean', 'Median'])
    # metadata_test = pd.DataFrame(std.transform(metadata_test), columns=['Variation', 'Mean', 'Median'])

    return metadata_train, metadata_test


def clustering(meta_train, meta_test):

    mbatch = MiniBatchKMeans(n_clusters=2, compute_labels=True)
    mb_train_c = mbatch.fit_predict(meta_train)
    mb_train_c = pd.DataFrame(mb_train_c)
    mb_test_c = mbatch.predict(meta_test)
    mb_test_c = pd.DataFrame(mb_test_c)

    dirname = os.path.dirname(__file__)
    mb_train_c.to_csv(os.path.join(dirname, 'data\\kmeans_clusters_train.csv'))
    mb_test_c.to_csv(os.path.join(dirname, 'data\\kmeans_clusters_test.csv'))


def model_update_results(scores, ID=None, modelname=None):
    """
    opens results.csv as tempory file and updates metrics for relevant model
    """
    # import data
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'results\\results.csv')
    tempfile = NamedTemporaryFile(mode='w', delete=False, newline='')

    mse = scores[0]
    score = scores[1]
    precision = scores[2]
    recall = scores[3]
    f1_score = scores[4]

    with open(filename, 'r') as csvfile, tempfile:
        fieldnames = ['Model_ID', 'Model Name', 'MSE', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)  # existing file
        writer = csv.DictWriter(tempfile, fieldnames=fieldnames)  # tempfile to overwrite exisiting file
        if ID:  # update existing entry
            entry_found = False
            for row in reader:
                if row['Model_ID'] == str(ID):
                    row['MSE'], row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score'] \
                        = round(mse, 3), round(score, 3), round(precision, 3), round(recall, 3), \
                          round(f1_score, 3)
                    entry_found = True
                    print('Existing Entry Updated')
                writer.writerow(row)
            if not entry_found:
                raise EOFError('Existing Entry Not Found')
        else:  # create new entry
            for row in reader:
                writer.writerow(row)
            idnum = int(row['Model_ID'])+1
            new_entry = {'MSE': mse, 'Accuracy': score, 'Precision': precision, 'Recall': recall,
                         'F1 Score': f1_score, 'Model_ID': idnum, 'Model Name': modelname}

            print('new entry created')
            writer.writerow(new_entry)

    shutil.move(tempfile.name, filename)


# # import data from csv
# X_train, y_train, X_test = import_data()
# # standard scale, create metafeatures
# meta_train, meta_test = process_data(X_train, y_train, X_test)
#
# clustering(meta_train, meta_test)

