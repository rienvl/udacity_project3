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
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


def train_model():
    """
    Function for training the model
    Trained model is stored in 'trainedmodel.pkl'
    """

    # read in finaldata.csv using the pandas module
    full_input_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    training_data = pd.read_csv(full_input_path)
    logging.info("OK - training.py: loaded train_data containing {} rows".format(training_data.shape[0]))
    # print(training_data.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.shape)
    X = training_data.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = training_data['exited'].values.reshape(-1, 1).ravel()  # target

    # use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    # multi_class='warn', n_jobs=None, penalty='l2',
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # fit the logistic regression to your data
    model = logit.fit(X, y)
    logging.info("OK - training.py: model training finished")

    # write the trained model to your workspace in a file called trainedmodel.pkl
    full_output_path = os.path.join(model_path, 'trainedmodel.pkl')
    filehandler = open(full_output_path, 'wb')
    pickle.dump(model, filehandler)
    logging.info("OK - training.py: stored model in {}".format(full_output_path))


if __name__ == '__main__':
    train_model()
