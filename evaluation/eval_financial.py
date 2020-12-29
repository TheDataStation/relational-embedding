import sys
sys.path.append('..')

import eval_utils as EU
import os
import numpy as np 
import pandas as pd
from os.path import isfile, join
from tqdm import tqdm
from os import listdir

from gensim.models import Word2Vec
import word2vec
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import json

def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store"]
    return fs

def join_tables_kraken(df):
    #TODO 
    return df

def simple_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators = 200)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    pscore = accuracy_score(y_test, y_pred)
    import pdb; pdb.set_trace();
    print("RF Test score:", pscore)

def joined_and_feature_elim_random_forest(X_train, X_test, y_train, y_test):
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(X_train, y_train)
    selected_feat= X_train.columns[(sel.get_support())]
    print("Num of selected features {} out of {}".format(len(selected_feat), len(X_train.columns)))

    X_train_elim = X_train[selected_feat]
    X_test_elim = X_test[selected_feat]

    simple_random_forest(X_train_elim, X_test_elim, y_train, y_test)

if __name__ == "__main__":
    print("Loading & splitting financial data")
    
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)
    config = data_config["financial"]
    location = config["location"]
    target_file = config["target_file"]
    location_processed = config["location_processed"]
    target_column = config["target_column"]

    # Load data 
    trimmed_table = pd.read_csv(os.path.join("../", location_processed), sep=',', encoding='latin')
    full_table = pd.read_csv(os.path.join("../", location + target_file), sep=',', encoding='latin')
    trimmed_table["date"] = trimmed_table["date"].apply(
        lambda x: 
        int("".join(x.split("-")))
    )

    # Create X and Y
    Y = full_table[target_column]
    X_train, X_test, y_train, y_test = train_test_split(trimmed_table, Y, test_size = 0.2, random_state=1234)

    # X_joined = df_joined.drop(["result"], axis = 1)
    # Y_joined = df_joined["result"].values.ravel()
    # X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(X_joined, Y_joined, test_size = test_size, random_state=10)
    
    # Baseline 1: simple random forest
    print("Baseline 1: Simple RF")
    for n in [3,5,10,15,20,50]:
        EU.remove_hubness_and_run(trimmed_table, Y, n_neighbors=n)
    
    simple_random_forest(X_train, X_test, y_train, y_test)

    # Baseline 2: Join all tables & random forest
    # print("Baseline 2: Joined & RF")
    # simple_random_forest(X_train_j, X_test_j, y_train_j, y_test_j) 

    # # Baseline 3: Join all tables & elim features & random forest
    # print("Baseline 3: Joined & Feature Selection & RF")
    # joined_and_feature_elim_random_forest(X_train_j, X_test_j, y_train_j, y_test_j)
