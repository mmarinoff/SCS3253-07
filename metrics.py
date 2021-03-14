"""
Place any general functions for plotting, manipulating data, here and import as necessary into other modules
"""
import os
from tempfile import NamedTemporaryFile
import shutil
import csv

from matplotlib import pyplot as plt


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)


def model_update_results(mse, auc, score, precision, recall, f1_score, ID=None, modelname=None):
    """
    opens results.csv as tempory file and updates metrics for relevant model
    """
    # import data
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'results\\results.csv')
    tempfile = NamedTemporaryFile(mode='w', delete=False, newline='')

    with open(filename, 'r') as csvfile, tempfile:
        fieldnames = ['Model_ID', 'Model Name', 'MSE', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)  # existing file
        writer = csv.DictWriter(tempfile, fieldnames=fieldnames)  # tempfile to overwrite exisiting file
        if ID:  # update existing entry
            entry_found = False
            for row in reader:
                if row['Model_ID'] == str(ID):
                    row['MSE'], row['AUC'], row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score'] \
                        = round(mse, 3), round(auc, 3), round(score, 3), round(precision, 3), round(recall, 3), \
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
            new_entry = {'MSE': mse, 'AUC': auc, 'Accuracy': score, 'Precision': precision, 'Recall': recall,
                         'F1 Score': f1_score, 'Model_ID': idnum, 'Model Name': modelname}

            print('new entry created')
            writer.writerow(new_entry)

    shutil.move(tempfile.name, filename)

