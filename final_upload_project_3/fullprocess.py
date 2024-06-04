import os
import json
import pickle
import pandas as pd
import logging
import training
import scoring
import apicalls
import deployment
import diagnostics
import reporting
import ingestion
from sklearn import metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

input_csv_path = os.path.join(config['input_folder_path'])
ingested_data_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

# Check and read new data
logging.info("OK - Checking for new data:")
# first, read ingestedfiles.txt
full_data_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
with open(full_data_path, 'r') as f:
    ingested_files_names = f.read()

logging.info("     loaded ingestedfiles.txt")
logging.info("     {}".format(ingested_files_names))

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
# identify csv files in input_csv_path folder
input_dir = os.path.join(os.getcwd(), input_csv_path)
new_csv_data_found = False  # initialize
filenames = os.listdir(input_dir)
for filename in filenames:
    if '.csv' in filename:
        # check if current file is new data
        if filename not in ingested_files_names:
            new_csv_data_found = True

# Deciding whether to proceed, part 1
if new_csv_data_found:
    logging.info("OK - new data found --> starting ingestion of new data")
    ingestion.merge_multiple_dataframe()
else:
    # no new data: end the process here
    logging.info("OK - no new data found --> exit")
    exit()

# Checking for model drift
# check whether the score from the deployed model is different
# from the score from the model that uses the newest ingested data
logging.info("OK - Checking for model drift")
full_file_name = os.path.join(prod_deployment_path, 'latestscore.txt')
with open(full_file_name) as f:
    score_old = float(f.read())

# load new data
full_data_path = os.path.join(ingested_data_path, 'finaldata.csv')
new_data_df = pd.read_csv(full_data_path)
logging.info("        loaded new_data containing {} rows".format(new_data_df.shape[0]))

# model prediction
predictions_list = diagnostics.model_predictions(new_data_df)
y = new_data_df['exited'].values.reshape(-1, 1).ravel()  # target
#predicted = model.predict(X)

# calculate an F1 score for the model relative to the new data
score_new = metrics.f1_score(predictions_list, y)

# score new data using current model
logging.info("        f1 score old: {:.3f}".format(score_old))
logging.info("        f1 score new: {:.3f}".format(score_new))

# Deciding whether to proceed, part 2
model_drift_occurred = score_new < score_old
if model_drift_occurred:
    logging.info("OK - model drift occurred --> starting re-deployment")
else:
    # no model_drift_occurred: end the process here
    logging.info("OK - no model drift occurred --> exit")
    exit()

# Re-train the model using the new ingestion data
logging.info("OK - retraining model using new ingested data")
training.train_model()
# load new model
full_model_path = os.path.join(model_path, 'trainedmodel.pkl')
with open(full_model_path, 'rb') as file:
    model = pickle.load(file)

# score new model using test data
scoring
# Re-deployment
deployment.store_model_into_pickle()
logging.info("OK - re-deployed new model")
# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
reporting.score_model()  # using test data
logging.info("OK - re-computed new confusion matrix")
apicalls
logging.info("OK - performed apicalls()")
