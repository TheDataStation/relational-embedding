import numpy as np 
import pandas as pd
import argparse
import json

# This file preprocess the datasets and remove target column from the dataset
def task_loader(args):
    with open(args.data_config, 'r') as json_file:
        configs = json.load(json_file)
        if args.task not in configs: 
            print("No such task")
            return 
        
        config = configs[args.task]
        location = config['location']
        target_file = config['target_file']
        location_processed = config['location_processed']
        target_column = config['target_column']
        task_type = config['task_type']

        df = pd.read_csv(location + target_file)
        df = df.drop(columns=[target_column])
        df.to_csv(location_processed, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--task', type=str, help='name of the task to be preprocessed')
    parser.add_argument('--data_config', type=str, 
        default='./data/data_config.txt', help='where to load task information')

    args = parser.parse_args()
    task_loader(args)