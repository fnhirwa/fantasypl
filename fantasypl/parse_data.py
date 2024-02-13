# Utility functions for downloading data from the FPL API
import os
import requests
import json
import csv

# global variables
FPL_GENERIC_INFO = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_INFO = "https://fantasy.premierleague.com/api/fixtures/"
FPL_PLAYERS_INFO_BASE = "https://fantasy.premierleague.com/api/element-summary/"
PLAYERS_IN_A_GAMEWEEK = "https://fantasy.premierleague.com/api/event/{}/live/"
FPL_MANAGER_INFO = "https://fantasy.premierleague.com/api/entry/{}/"
FPL_LEAGUE_INFO = "https://fantasy.premierleague.com/api/leagues-classic/{}/standings/"
FPL_MANAGER_GIVEN_GAMEWEEK = (
    "https://fantasy.premierleague.com/api/entry/{}/event/{}/picks/"
)


def download_data(url, file_path):
    """
    Downloads data from the FPL API and saves .csv file to the specified file path
    """
    response = requests.get(url)
    with open(file_path, "w") as f:
        f.write(response.text)


def convert_json_to_csv(json_file, csv_file):
    """
    Converts a .json file to a .csv file
    """
    with open(json_file, "r") as f:
        data = json.load(f)
        players_data_keys = data["elements"][0].keys()
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, players_data_keys)
            writer.writeheader()
            writer.writerows(data["elements"])


def download_fpl_generic_data():
    global FPL_GENERIC_INFO
    """
    Downloads the generic data from the FPL API and saves it to a .csv file
    The downloaded data contains

    - A brief overview of all 38 gameweeks
    - Game settings
    - Phases of the season
    - Info on all 20 PL teams
    - Total number of FPL players
    - Basic info on all FPL players
    - FPL positions
    - PL Players info
    """
    data_path = os.path.join(os.path.dirname(__file__), "data")
    csv_file_generic = os.path.join(data_path, "fpl_generic_data.csv")
    json_file_generic = os.path.join(data_path, "fpl_generic_data.json")
    download_data(FPL_GENERIC_INFO, json_file_generic)
    convert_json_to_csv(json_file_generic, csv_file_generic)
    print("Data downloaded and converted successfully!")


# Everytime we want to update the data we will need to download it from API
download_fpl_generic_data()
print("Data downloaded successfully!")

data_path = os.path.join(os.path.dirname(__file__), "data")
json_file = os.path.join(data_path, "fpl_generic_data.json")
# Replace 'your_json_file.json' with the path to your JSON file

with open(json_file, "r") as file:
    DATA = json.load(file)

TOTAL_PLAYERS = DATA["total_players"]
# keys of the generic data
GENERIC_DATA_KEYLIST = [
    "events",
    "game_settings",
    "phases",
    "teams",
    "total_players",
    "elements",
    "element_stats",
    "element_types",
]


# organize the events data into a csv file for easy viewing
def _write_the_generic_dict_settings_to_csv(key_element, preferred_file_name=None):
    global DATA
    key_data = DATA[key_element]
    key_data_keys = key_data.keys()
    if preferred_file_name:
        file_path = os.path.join(data_path, f"generic_{preferred_file_name}.csv")
    else:
        file_path = os.path.join(data_path, f"generic_{key_element}.csv")
    with open(file_path, "w", newline="") as generic_file:
        writer = csv.DictWriter(generic_file, key_data_keys)
        writer.writeheader()
        writer.writerow(key_data)


def write_the_generic_list_element_to_csv(key_element, preferred_file_name=None):
    global DATA
    key_data = DATA[key_element]
    if isinstance(key_data, list):
        key_data_keys = key_data[0].keys()
        if preferred_file_name:
            file_path = os.path.join(data_path, f"generic_{preferred_file_name}.csv")
        else:
            file_path = os.path.join(data_path, f"generic_{key_element}.csv")
        with open(file_path, "w", newline="") as generic_file:
            writer = csv.DictWriter(generic_file, key_data_keys)
            writer.writeheader()
            writer.writerows(key_data)
    elif isinstance(key_data, dict):
        _write_the_generic_dict_settings_to_csv(
            key_element, preferred_file_name=preferred_file_name
        )


for key in GENERIC_DATA_KEYLIST:
    write_the_generic_list_element_to_csv(key)


def generic_data_writting():
    print("Preparing the generic data...")
    for key in GENERIC_DATA_KEYLIST:
        if key != "total_players":
            if key == "events":
                preffered_file_name = "events"
            elif key == "game_settings":
                preffered_file_name = "game_settings"
            elif key == "phases":
                preffered_file_name = "phases"
            elif key == "teams":
                preffered_file_name = "teams"
            elif key == "elements":
                preffered_file_name = "players"
            elif key == "element_stats":
                preffered_file_name = "players_stats"
            elif key == "element_types":
                preffered_file_name = "players_types"
            write_the_generic_list_element_to_csv(
                key, preferred_file_name=preffered_file_name
            )

    print("Data written successfully!")


if __name__ == "__main__":
    generic_data_writting()
