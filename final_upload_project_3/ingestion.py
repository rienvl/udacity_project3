import pandas as pd
import numpy as np
import logging
import ast
import os
import json
# from datetime import datetime


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    # instantiate final_dataframe as None
    final_dataframe = None
    all_csv_files_list = []

    # for every csv file it finds, it appends the file's data to the
    input_dir = os.path.join(os.getcwd(), input_folder_path)
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
                final_dataframe = pd.concat([final_dataframe, current_df], ignore_index=True).reset_index(drop=True)

            # add full csv file name to all_csv_files_list
            all_csv_files_list.append(full_path)

    logging.info("OK - ingestion.py: loaded {} csv files from directory {}".format(count_csv, input_dir))

    # remove duplicates from dataframe
    output_df = final_dataframe.drop_duplicates()
    logging.info("OK - ingestion.py: dataset after de-dupe contains {} unique rows".format(output_df.shape[0]))

    # write merged dataframe to an output file
    output_dir = os.path.join(os.getcwd(), output_folder_path)
    full_path_out = os.path.join(output_dir, 'finaldata.csv')
    output_df.to_csv(full_path_out, index=False)
    logging.info("OK - ingestion.py: dataset save in {}".format(full_path_out))

    # store a record of all combined .csv files in a file called ingestedfiles.txt
    # get a current timestamp
    # dateTimeObj = datetime.now()
    # the_time_now = str(dateTimeObj.year) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.day)
    full_path_out = os.path.join(output_dir, 'ingestedfiles.txt')
    with open(full_path_out, 'w') as f:
        for full_file_name in all_csv_files_list:
            # compile all the relevant information
            # all_records = [sourcelocation, filename, len(data.index), the_time_now]
            f.write(f"{full_file_name}\n")

    logging.info("OK - ingestion.py: record of all .csv files stored in {}".format(full_path_out))

    # test output file
    # Open a list of previous scores, using the ast module:
    with open(full_path_out) as f:
        files_list = f.readlines()

    print(files_list)
    print(len(files_list))
    print(files_list[0])


if __name__ == '__main__':
    merge_multiple_dataframe()
