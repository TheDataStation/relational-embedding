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

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble.forest import RandomForestClassifier

K = 20 
test_size = 0.2
num_bins = 20
taxi_data_path = "../data/taxi_small/"
word2vec_model_path = "../word2vec/word2vec_neg_20_i50.bin"

def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store"]
    return fs

def join_tables(df):
    df_joined = df 
    fs = all_files_in_path(taxi_data_path)
    for f in fs:
        df_new = pd.read_csv(f)
        try:
            df_joined = pd.merge(df_joined, df_new, left_on = "d3mIndex", right_on = "id")
        except:
            print("i can't find matching id column %s", f)
    return df_new 

def simple_regression(X_train, X_test, y_train, y_test):
    X_train = X_train.drop(["datetime"], axis = 1)
    X_test = X_test.drop(["datetime"], axis = 1)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    train_score=lr.score(X_train, y_train)
    test_score=lr.score(X_test, y_test)
    print("LR Train score: %d, Test score: %d", train_score, test_score)

def joined_and_lasso(X_train, X_test, y_train, y_test):
    X_train = X_train.drop(["datetime"], axis = 1)
    X_test = X_test.drop(["datetime"], axis = 1)
    
    alphas = [1e-10, 1e-8, 1e-4, 1e-2, 1, 5, 10, 20]
    for alpha in alphas:
        lasso = Lasso(alpha = alpha)
        lasso.fit(X_train, y_train) 
        train_score = lasso.score(X_train, y_train)
        test_score = lasso.score(X_test, y_test)
        coeff_used = np.sum(lasso.coef_!=0)
        print("LASSO alpha %d Train score: %d, Test score: %d", alpha, train_score, test_score)
        print("LASSO Num of coeff used: %d", coeff_used)

def quantize(df, hist = "width"):
    cols = df.columns
    bin_percentile = 100 / num_bins
    for col in cols:
        if df[col].dtype not in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
            continue 
        
        if hist == "width":
            print(bin_percentile)
            bins = [np.percentile(df[col], i * bin_percentile) for i in range(num_bins)]
        else: 
            bins = [i * (df[col].max() - df[col].min()) / num_bins for i in range(num_bins)]
        
        df[col] = np.digitize(df[col], bins)
    return df

def scatter_plot(feature, target):
    plt.figure(figsize = (16, 8))
    plt.scatter(data[feature], data[target], c='black')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()

if __name__ == "__main__":
    print("Loading & splitting taxi data")
    df = pd.read_csv(os.path.join(taxi_data_path, "base_data.csv"))
    df = quantize(df)
    df_joined = join_tables(df)

    X = df.drop(['n. collisions'], axis = 1)
    Y = df['n. collisions'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = test_size, random_state=10)
    
    X_joined = df.drop(['n. collisions'], axis = 1)
    Y_joined = df['n. collisions'].values.reshape(-1, 1)
    X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(X,Y, test_size = test_size, random_state=10)
    
    # Baseline 1: simple regression 
    simple_regression(X_train, X_test, y_train, y_test)

    # Baseline 2: Join all tables & regression
    simple_regression(X_train_j, X_test_j, y_train_j, y_test_j) 

    # Baseline 3: Join all tables & LASSO
    joined_and_lasso(X_train_j, X_test_j, y_train_j, y_test_j)