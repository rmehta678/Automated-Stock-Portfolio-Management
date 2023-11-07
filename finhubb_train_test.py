import finnhub
import json
import datetime
import time

name = []

def extract_names_from_json(json_file_path):
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    names = []
    for entry in json_data:
        names.append(entry["Symbol"])

    return names


def main():
    json_file_path = "s&p500.json"
    names = extract_names_from_json(json_file_path)

    print(len(names))

if __name__ == '__main__':
    main()