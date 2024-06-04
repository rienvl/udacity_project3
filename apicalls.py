import requests
import os
import json


# Specify a URL that resolves to your workspace
#URL = 'http://127.0.0.1:8000'
#URL = 'http://0.0.0.0:8000'

with open('config.json','r') as f:
    config = json.load(f)

output_path = os.path.join(config['output_model_path'])


# Call each API endpoint and store the responses
response1 = requests.get('http://127.0.0.1:8000/prediction?filename=/testdata/testdata.csv').content  # returns list of predictions
response2 = requests.get('http://localhost:8000/scoring').content  # returns f1_score
response3 = requests.get('http://127.0.0.1:8000/summarystats').content  # returns percentage_of_na for each num. column
response4 = requests.get('http://127.0.0.1:8000/diagnostics').content  # returns timings, percentage_of_na, outdated
print(response1)
print(response2)
print(response3)
print(response4)

# combine all API responses
#responses =  # combine responses here

# write the responses to your workspace
#full_output_path = os.path.join(output_path, 'apireturns.txt')
#with open(full_output_path, "w") as text_file:
#    text_file.write("%s" % responses)
