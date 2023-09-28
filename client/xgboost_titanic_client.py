# Usage: python xgboost_titanic_serve.py <profile> <name> <csv_file_path>
# Ex: python xgboost_titanic_serve.py dkubex xgboost-model test_file/test.csv
import configparser
import requests
import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

config = configparser.ConfigParser()


# Check if a parameter is provided
if len(sys.argv) > 2:
    param1 = sys.argv[1]
    param2 = sys.argv[2]
    param3 = sys.argv[3]
else:
    print("Please provide uuid & csv file path.")
    exit(1)

# get http url & token
ini_file = "/userdata/.d3x.ini"
config.read(ini_file)
url = config.get(param1,"url")
token = config.get(param1,"auth-token")


# get deployment details
headers = {'Authorization': token}
r = requests.get(f"{url}/llm/api/deployments/{param2}", headers=headers, verify=False)
deployment = r.json()['deployment']

# get serving details
SERVING_TOKEN = deployment['serving_token']
SERVING_ENDPOINT = f"{url}{deployment['endpoint']}"
FILE_PATH =  param3 #"test_file/test.csv"

# Read csv File
data = open(FILE_PATH, 'rb').read()

# serving request
headers={'Authorization': SERVING_TOKEN}
resp = requests.post(SERVING_ENDPOINT, data=data, headers=headers, verify=False)
print (resp.json())
