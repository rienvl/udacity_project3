import pandas as pd
import numpy as np
import logging
import pickle
import timeit
import os
import json


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

# Function to get model predictions
def model_predictions(test_data_df):
    # read the deployed model
    full_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    with open(full_model_path, 'rb') as file:
        model = pickle.load(file)
    logging.info("OK - model_predictions.py: loaded model".format(full_model_path))

    # calculate predictions
    X = test_data_df.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    predictions_list = model.predict(X)
    logging.info("OK - model_predictions.py: derived model predictions (length={})".format(len(predictions_list)))

    return predictions_list  # return value should be a list containing all predictions


# Function to get summary statistics
def dataframe_summary():
    # load the dataset from output_path
    full_test_data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    data_df = pd.read_csv(full_test_data_path)
    logging.info("OK - diagnostics.py - loaded dataframe with {} rows".format(data_df.shape[0]))

    # calculate summary statistics for all columns with numeric data
    statistics_list = []
    counter = 0
    for column in data_df.columns:
        if pd.api.types.is_numeric_dtype(data_df[column]):
            # logging.info("OK - diagnostics.py - column {} is numeric".format(column))
            # compute mean, median, and std
            statistics_list.append(np.mean(data_df[column]))
            statistics_list.append(np.median(data_df[column]))
            statistics_list.append(np.std(data_df[column]))
            counter += 1

    logging.info("OK - diagnostics.py - dataframe_summary for {} numeric columns".format(counter))
    logging.info("OK - diagnostics.py - dataframe_summary returned:")
    logging.info("OK - {}".format(statistics_list))

    return statistics_list  # return value should be a list containing all summary statistics


# Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds


# Function to check dependencies
def outdated_packages_list():
    # get a list of
    pass


if __name__ == '__main__':
    # read in dataset test_data_df
    full_test_data_path = os.path.join(test_data_path, 'testdata.csv')
    test_data_df = pd.read_csv(full_test_data_path)
    logging.info("OK - main.py: loaded test_data containing {} rows".format(test_data_df.shape[0]))

    # get model predictions from data in output_path
    predictions_list = model_predictions(test_data_df)

    # calculate summary statistics on the data
    statistics_list = dataframe_summary()

    execution_time()
    outdated_packages_list()
