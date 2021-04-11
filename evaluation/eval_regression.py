import sys
sys.path.append('..')

import eval_utils as EU
import os
import numpy as np 
import pandas as pd 
# import visualizer as VS
# from gensim.models import Word2Vec
import word2vec
from os.path import isfile, join
from tqdm import tqdm
from os import listdir
import json
from eval_utils import all_files_in_path
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble.forest import RandomForestClassifier

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]

data_path = "../data/taxi_samll/data"
word2vec_model_path = "../word2vec/taxi_small.bin"

def join_tables(df):
    df_joined = df 
    fs = all_files_in_path(data_path)
    for f in fs:
        if f == "../data/taxi_small/base_data.csv" or "meta" in f or "json" in f:
            continue
        df_new = pd.read_csv(f)
        try:
            df_joined = pd.merge(df_joined, df_new, left_on = "datetime", right_on = "CMPLNT_FR_DT", how = "left")
            print("Successfully joined with %s:", f)
        except:
            print("i can't find matching id column %s", f)
    return df_joined 

def simple_regression(X_train, X_test, y_train, y_test):
    X_train = X_train.drop(["datetime"], axis = 1)
    X_test = X_test.drop(["datetime"], axis = 1)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)

    y_tmp = lr.predict(X_train)
    print("LR Train score: {}, Test score: {}".format(train_score, test_score))

def joined_and_lasso(X_train, X_test, y_train, y_test):
    X_train = X_train.drop(["datetime"], axis = 1)
    X_test = X_test.drop(["datetime"], axis = 1)
    
    alpha = 1e-2
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train) 
    train_score = lasso.score(X_train, y_train)
    test_score = lasso.score(X_test, y_test)
    coeff_used = np.sum(lasso.coef_!=0)
    print("LASSO alpha {} Train score: {}, Test score: {}".format(alpha, train_score, test_score))
    print("LASSO Num of coeff used: {}".format(coeff_used))

def scatter_plot(feature, target):
    plt.figure(figsize = (16, 8))
    plt.scatter(data[feature], data[target], c='black')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()

if __name__ == "__main__":
    print("Loading & splitting taxi data")
    df = pd.read_csv(os.path.join(taxi_data_path, "base_data.csv"))
    df = EU.quantize(df, excluding = ['n. collisions'])
    df_joined = join_tables(df).fillna(0)

    X = df.drop(['n. collisions'], axis = 1)
    Y = df['n. collisions'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = test_size, random_state=10)
    
    X_joined = df_joined.drop(['n. collisions'], axis = 1)
    Y_joined = df_joined['n. collisions'].values.reshape(-1, 1)
    X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(X_joined, Y_joined, test_size = test_size, random_state=10)
    
    # Baseline 1: simple regression 
    print("Baseline 1: Simple Regression")
    simple_regression(X_train, X_test, y_train, y_test)

    # Baseline 2: Join all tables & regression
    print("Baseline 2: Joined & Regression")
    simple_regression(X_train_j, X_test_j, y_train_j, y_test_j) 

    # Baseline 3: Join all tables & LASSO
    print("Baseline 3: Joined & Feature Selection LASSO")
    joined_and_lasso(X_train_j, X_test_j, y_train_j, y_test_j)