from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import logging
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


# Function for model scoring
def score_model(model):
    """
    this function takes a trained model as input, loads test data,
    calculates an F1 score for the model relative to the test data,
    and writes the result to the latestscore.txt file
    Note on test_data:
    test_data: dataframe with test dataset with at least the following columns:
               ['lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited']

    :param model: trained model
    :return: f1_score
    """

    # load test data
    full_test_data_path = os.path.join(test_data_path, 'testdata.csv')
    test_data = pd.read_csv(full_test_data_path)
    logging.info("OK - scoring.py: loaded test_data containing {} rows".format(test_data.shape[0]))

    # model prediction
    X = test_data.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = test_data['exited'].values.reshape(-1, 1).ravel()  # target
    predicted = model.predict(X)

    # calculate an F1 score for the model relative to the test data
    f1_score = metrics.f1_score(predicted, y)
    logging.info("OK - scoring.py: calculated F1 score = {:.2f}".format(f1_score))

    # write the result to the latestscore.txt file
    full_output_path = os.path.join(output_model_path, 'latestscore.txt')
    with open(full_output_path, "w") as text_file:
        text_file.write("%s" % f1_score)
    logging.info("OK - scoring.py: F1 score data saved to {}".format(full_output_path))

    return f1_score


if __name__ == '__main__':
    # load deployed model from model_path specified in config
    full_model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(full_model_path, 'rb') as file:
        model = pickle.load(file)
    logging.info("OK - main: loaded model".format(full_model_path))

    # run score_model()
    f1_score = score_model(model)
