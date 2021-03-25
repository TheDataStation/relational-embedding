import numpy as np 
import pandas as pd
import argparse
import json
from collections import defaultdict 
from os import listdir
from pathlib import Path
from os.path import isfile, join

def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store" and f != "base_processed.csv"]
    return fs

# This file preprocess the datasets and remove target column from the dataset
def task_loader(args):
    with open('./data/data_config.txt', 'r') as json_file:
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

    # Remove target column
    df = pd.read_csv(location_processed)
    if target_column in df.columns:
        df.to_csv(location + target_file, index=False)
        df = df.drop(columns=[target_column])
    df.to_csv(location_processed, index=False)

    # Generate textfication strategies
    fs = all_files_in_path(config["location"])
    strategies = dict()

    for path in fs:
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)
        if target_column in df.columns: 
            df = df.drop(columns = [target_column])
            df.to_csv(path, index=False)

        table_name = path.split("/")[-1]
        strategies[table_name] = defaultdict(dict)

        for col in df.columns:
            integer_strategy, grain_strategy = "augment", "cell"
            num_distinct_numericals = df[col].nunique()

            if df[col].dtype in [np.float, np.float16, np.float32, np.float64]:
                if abs(df[col].skew()) >= 2: 
                    integer_strategy = "eqw_quantize"
                else:
                   integer_strategy = "eqh_quantize"
            
            if df[col].dtype in [np.int64, np.int32, np.int64, np.int]:
                if df[col].max() - df[col].min() >= 5 * df[col].shape[0]:
                    if abs(df[col].skew()) >= 2: 
                        integer_strategy = "eqw_quantize"
                    else:
                        integer_strategy = "eqh_quantize"

            if df[col].dtype == np.object:
                num_tokens_med = (df[col].str.count(' ') + 1).median()
                if num_tokens_med >= 10: 
                    grain_strategy = "token"
            
            strategies[table_name][col]["int"] = integer_strategy
            strategies[table_name][col]["grain"] = grain_strategy
    
    with open("./data/strategies/" + args.task + ".txt", "w") as json_file:
        json.dump(strategies, json_file, indent=4)
    
    dbdir = Path.cwd() / "graph" / args.task
    dbdir.mkdir(parents=True, exist_ok=True)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--task', type=str, help='name of the task to be preprocessed')

    args = parser.parse_args()
    task_loader(args)