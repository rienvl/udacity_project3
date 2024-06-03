import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

# dataset_csv_path = os.path.join(config['output_folder_path'])  # confusing name: data should come from test_data_path
output_path = os.path.join(config['output_folder_path'])  # confusing name: data should come from test_data_path
test_data_path = os.path.join(config['test_data_path'])


# Function for reporting
def score_model():
    # read in dataset test_data_df
    full_test_data_path = os.path.join(test_data_path, 'testdata.csv')
    test_data_df = pd.read_csv(full_test_data_path)
    y_true = test_data_df['exited'].values.tolist()

    logging.info("OK - reporting.py: loaded test_data containing {} rows".format(test_data_df.shape[0]))

    # derive model predictions
    y_pred = model_predictions(test_data_df)
    logging.info("OK - reporting.py: derived model predictions")
    # print('y_true: {:}'.format(y_true))
    # print('y_pred: {:}'.format(y_pred))

    # calculate a confusion matrix using the test data and the deployed model
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
    disp.plot()
    # plt.show()

    # write the confusion matrix to the workspace
    full_output_path = os.path.join(output_path, 'confusionmatrix.png')
    plt.savefig(full_output_path)
    logging.info("OK - reporting.py: saved plot confusion matrix to {}".format(full_output_path))


if __name__ == '__main__':
    score_model()
