# Utility functions for downloading data from the FPL API

import os
import requests
import json
import csv

# global variables
FPL_DATA_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'

def download_data(url, file_path):
    """
    Downloads data from the FPL API and saves .csv file to the specified file path
    """
    response = requests.get(url)
    with open(file_path, 'w') as f:
        f.write(response.text)

def convert_json_to_csv(json_file, csv_file):
    """
    Converts a .json file to a .csv file
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
        players_data_keys = data['elements'][0].keys()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, players_data_keys)
            writer.writeheader()
            writer.writerows(data['elements'])


if __name__ == '__main__':
    # download data
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    csv_file = os.path.join(data_path, 'fpl_data.csv')
    json_file = os.path.join(data_path, 'fpl_data.json')
    convert_json_to_csv(json_file, csv_file)
    print('Data downloaded and converted successfully!')
