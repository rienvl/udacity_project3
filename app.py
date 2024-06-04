from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model
# import predict_exited_from_saved_model
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
def prediction():
    filename = request.args.get('filename')
    data_df = read_pandas(filename)
    # call the prediction function you created in Step 3
    predictions_list = model_predictions(data_df)

    return {'predictions_list': predictions_list, 'status': 200}  # add return value for prediction outputs


# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    # load deployed model from model_path specified in config
    print('loading model from {}'.format(full_model_path))
    with open(full_model_path, 'rb') as file:
        model = pickle.load(file)

    print('OK - model loaded')
    # check the score of the deployed model on the test data specified in \testdata\testdata.csv
    f1_score = score_model(model)
    print('OK - f1_score = {}'.format(f1_score))

    return {'f1_score': f1_score, 'status': 200}  # add return value (a single F1 score number)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    # check means, medians, and modes for each column
    statistics_list = dataframe_summary()

    return {'statistics_list': statistics_list, 'status': 200}  # return a list of all calculated summary statistics


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    check timing, percentage of NA values, and dependencies
    :return: timings, percentage_of_na, outdated
    """
    # check timing
    timings = execution_time()

    # check percent NA values
    percentage_of_na = missing_data()

    # check dependencies
    outdated = outdated_packages_list()

    return {'timings': timings, 'percentage_of_na': percentage_of_na, 'outdated': outdated,  'status': 200}


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
