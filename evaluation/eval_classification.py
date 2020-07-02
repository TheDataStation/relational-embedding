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

K = 20 
test_size = 0.2
num_bins = 50
kraken_path = "../data/kraken/"
school_path = "../data/school/"

def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store"]
    return fs

def join_tables_kraken(df):
    df_joined = df 
    fs = all_files_in_path(kraken_path)
    for f in fs:
        if f == "../data/kraken/kraken.csv": 
            continue
        df_new = pd.read_csv(f)
        try:
            df_joined = pd.merge(df_joined, df_new, left_on = "event_id", right_on = "event_id", how = "left")
        except:
            print("i can't find matching id column in file %s, skip!" % f)
    return df_joined 

def join_tables_school(df):
    df_joined = df
    fs = all_files_in_path(school_path)
    for f in fs:
        if f == "../data/school/base.csv" or f == "../data/school/schema.csv": 
            continue
        df_new = pd.read_csv(f)
        try:
            if "DBN" in df_new.columns:
                df_joined = pd.merge(df_joined, df_new, left_on = "DBN", right_on = "DBN", how = "left")
            if "dbn" in df_new.columns:
                df_joined = pd.merge(df_joined, df_new, left_on = "DBN", right_on = "dbn", how = "left")
            if f == "../data/school/2012_-_2013_School_Locations.csv":
                df_joined = pd.merge(df_joined, df_new, left_on = "DBN", right_on = "LOCATION_NAME", how = "left")
            print("Successfully joined with %s:", f)
            print(df_joined.shape)
        except Exception as e:
            print("i can't find matching id column in file %s, skip!" % f)
    return df_joined 

def simple_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    pscore = accuracy_score(y_test, y_pred)
    print("RF Test score:", pscore)

def joined_and_feature_elim_random_forest(X_train, X_test, y_train, y_test):
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(X_train, y_train)
    selected_feat= X_train.columns[(sel.get_support())]
    print("Num of selected features {} out of {}".format(len(selected_feat), len(X_train.columns)))

    X_train_elim = X_train[selected_feat]
    X_test_elim = X_test[selected_feat]

    simple_random_forest(X_train_elim, X_test_elim, y_train, y_test)

def scatter_plot():
    plt.figure(figsize = (16, 8))
    plt.scatter(df["f4k"], df["f109k"], c=(df["result"] == "nofail"))
    plt.xlabel("f4k")
    plt.ylabel("f109k")
    plt.savefig("plot.png")

if __name__ == "__main__":
    print("Loading & splitting kraken data")
    df = pd.read_csv(os.path.join(kraken_path, "kraken.csv"))
    df = EU.quantize(df, excluding = ["event_id", "result"])
    df["result"] = (df["result"] == "nofail")
    df_joined = join_tables_kraken(df)

    X = df.drop(['result'], axis = 1)
    Y = df['result'].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = test_size, random_state=10)
    
    X_joined = df_joined.drop(["result"], axis = 1)
    Y_joined = df_joined["result"].values.ravel()
    X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(X_joined, Y_joined, test_size = test_size, random_state=10)
    
    # Baseline 1: simple random forest 
    print("Baseline 1: Simple RF")
    simple_random_forest(X_train, X_test, y_train, y_test)

    # Baseline 2: Join all tables & random forest
    print("Baseline 2: Joined & RF")
    simple_random_forest(X_train_j, X_test_j, y_train_j, y_test_j) 

    # Baseline 3: Join all tables & elim features & random forest
    print("Baseline 3: Joined & Feature Selection & RF")
    joined_and_feature_elim_random_forest(X_train_j, X_test_j, y_train_j, y_test_j)

    # print("Loading & splitting school data")
    # df = pd.read_csv(os.path.join(school_path, "base.csv"))
    # print(df.shape)
    # df = EU.quantize(df, excluding = ["class"])
    # df["class"] = (df["class"] == True)
    # print(df.shape)
    # df_joined = join_tables_school(df)

    # print(df_joined.shape)
    # X = df.drop(['class'], axis = 1)
    # Y = df['class'].values.ravel()
    # X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = test_size, random_state=10)
    
    # X_joined = df_joined.drop(["class"], axis = 1)
    # Y_joined = df_joined["class"].values.ravel()
    # X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(X_joined, Y_joined, test_size = test_size, random_state=10)
    
    # # Baseline 1: simple random forest 
    # print("Baseline 1: Simple RF")
    # simple_random_forest(X_train, X_test, y_train, y_test)

    # # # Baseline 2: Join all tables & random forest
    # print("Baseline 2: Joined & RF")
    # simple_random_forest(X_train_j, X_test_j, y_train_j, y_test_j) 

    # # # Baseline 3: Join all tables & elim features & random forest
    # print("Baseline 3: Joined & Feature Selection & RF")
    # joined_and_feature_elim_random_forest(X_train_j, X_test_j, y_train_j, y_test_j)
