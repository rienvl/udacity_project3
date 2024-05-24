import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    # instantiate final_dataframe as None
    final_dataframe = None
    # preferred way would be df_list = pd.DataFrame(columns=['col1','col2','col3'])

    # for every csv file it finds, it appends the file's data to the
    input_dir = os.getcwd() + input_folder_path
    count_csv = 0  # reset
    filenames = os.listdir(input_dir)
    for filename in filenames:
        if '.csv' in filename:
            count_csv += 1
            full_path = os.path.join(input_dir, filename)
            current_df = pd.read_csv(full_path)
            if final_dataframe is None:
                final_dataframe = current_df
            else:
                final_dataframe.append(current_df).reset_index(drop=True)

    logging.info("OK - ingestion.py: loaded {count_csv}} csv files from directory {input_dir}")

    # remove duplicates from dataframe
    output_df = final_dataframe.drop_duplicates()
    logging.info("OK ingestion.py: dataset after de-dupe contains {output_df.shape[0]} unique rows")

    # write to an output file
    output_dir = os.getcwd() + output_folder_path
    full_path_out = os.join(output_dir, 'finaldata.csv')
    output_df.to_csv(full_path_out)
    logging.info("OK ingestion.py: dataset save in {full_path_out}")


if __name__ == '__main__':
    merge_multiple_dataframe()
