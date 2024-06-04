import requests
import os
import json


# Specify a URL that resolves to your workspace
URL = 'http://127.0.0.1'
URL = 'http://0.0.0.0'

with open('config.json','r') as f:
    config = json.load(f)

output_path = os.path.join(config['output_model_path'])

# Call each API endpoint and store the responses
response1 = requests.post(URL + ':8000/prediction?filename=~/git/udacity_project3/testdata/testdata.csv').content  # returns list predictions
response2 = requests.get(URL + ':8000/scoring').content  # returns f1_score
response3 = requests.get(URL + ':8000/summarystats').content  # returns percentage_of_na for each num. column
response4 = requests.get(URL + ':8000/diagnostics').content  # returns timings, percentage_of_na, outdated

response1 = eval(response1.decode('ascii'))
response2 = eval(response2.decode('ascii'))
response3 = eval(response3.decode('ascii'))
response4 = eval(response4.decode('ascii'))

# pop status from all responses
response1.pop('status')
response2.pop('status')
response3.pop('status')
response4.pop('status')

print('response_1: {}'.format(response1))
print('response_2: {}'.format(response2))
print('response_3: {}'.format(response3))
print('response_4: {}'.format(response4))

# combine all API responses
responses = response1 | response2 | response3 | response4
print('combined responses = {}'.format(responses))

# write the responses to your workspace
full_output_path = os.path.join(output_path, 'apireturns.txt')
with open(full_output_path, 'w') as file:
    file.write(json.dumps(responses))
print('OK - apicalls completed')
