from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import create_prediction_model
import diagnosis
import diagnostics
import scoring
import predict_exited_from_saved_model
import json
import os


# Set up variables for use in our script
app = Flask(__name__)  # instantiate app
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
full_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')

prediction_model = None


def read_pandas(filename):
    df = pd.read_csv(filename)
    return df


# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    filename = request.args.get('filename')
    data_df = read_pandas(filename)
    # call the prediction function you created in Step 3
    predictions_list = diagnostics.model_predictions(data_df)

    return predictions_list, 200  # add return value for prediction outputs


# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():
    # load deployed model from model_path specified in config
    with open(full_model_path, 'rb') as file:
        model = pickle.load(file)

    # check the score of the deployed model on the test data specified in \testdata\testdata.csv
    f1_score = scoring.score_model(model)
    return f1_score, 200  # add return value (a single F1 score number)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    # check means, medians, and modes for each column
    statistics_list = diagnostics.dataframe_summary()

    return statistics_list, 200  # return a list of all calculated summary statistics


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():
    """
    check timing, percentage of NA values, and dependencies
    :return: timings, percentage_of_na, outdated
    """
    # check timing
    timings = diagnostics.execution_time()

    # check percent NA values
    percentage_of_na = diagnostics.missing_data()

    # check dependencies
    outdated = diagnostics.outdated_packages_list()

    return timings, percentage_of_na, outdated, 200  # add return value for all diagnostics


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
