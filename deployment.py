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

# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


# function for deployment
def store_model_into_pickle():
    """
    # copies the latest pickle file,  the latestscore.txt value, and the ingestfiles.txt file
    to the deployment directory
    """
    # target: the deployment directory
    # copy the latest pickle file
    full_source_path = os.path.join(model_path, 'trainedmodel.pkl')
    full_destination_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    os.system("cp {} {}".format(full_source_path, full_destination_path))
    logging.info("OK - deployment.py: copied {} to {}".format(full_source_path, full_destination_path))

    # copy the latestscore.txt value
    full_source_path = os.path.join(dataset_csv_path, 'latestscore.txt')
    full_destination_path = os.path.join(prod_deployment_path, 'latestscore.txt')
    os.system("cp {} {}".format(full_source_path, full_destination_path))
    logging.info("OK - deployment.py: copied {} to {}".format(full_source_path, full_destination_path))

    # copy the ingestedfiles.txt file
    full_source_path = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    full_destination_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    os.system("cp {} {}".format(full_source_path, full_destination_path))
    logging.info("OK - deployment.py: copied {} to {}".format(full_source_path, full_destination_path))


if __name__ == '__main__':
    store_model_into_pickle()
