import sys
sys.path.append('..')

from sklearn.feature_selection import SelectKBest, f_regression
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

from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from textification import textify_relation as tr

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]
test_size = embeddding_config["test_size"]

data_path = "../data/Biodegradability/"
#word2vec_model_path = "../word2vec/taxi_small.bin"

filenames = ['atom.csv', 'bond.csv', 'gmember.csv']
atom_df = pd.read_csv(data_path + 'atom.csv')
bond_df = pd.read_csv(data_path + 'bond.csv')
gmemeber_df = pd.read_csv(data_path + 'gmember.csv')
group_df = pd.read_csv(data_path + 'group.csv')
molecule_df = pd.read_csv(data_path + 'base_processed.csv')

df_joined = molecule_df.merge(atom_df, how="left", on = "molecule_id")
df_joined = df_joined.merge(bond_df, how="left", on="atom_id")
df_joined2 = gmemeber_df.merge(group_df, how="left", on="group_id")
df_joined = df_joined.merge(df_joined2, how="left", on="atom_id")

categorical = ['molecule_id', 'atom_id', 'atom_id2', 'type_x','group_id', 'type']
for c in categorical:   
    df_joined[c] = pd.Categorical(df_joined[c]).codes
#df_joined = df_joined.dropna()



def simple_regression(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
    train_score = mean_absolute_error(y_pred, y_train)
    test_score = mean_absolute_error(y_test_pred, y_test)

    y_tmp = lr.predict(X_train)
    print("LR Train score: {}, Test score: {}".format(train_score, test_score))

def joined_and_feature_elim_random_forest(X_train, X_test, y_train, y_test, kk):
    fs = SelectKBest(score_func=f_regression, k=kk)
    fs.fit(X_train, y_train)
    selected_feat= X_train.columns[(fs.get_support())]
    print("Num of selected features {} out of {}".format(len(selected_feat), len(X_train.columns)))

    X_train_elim = X_train[selected_feat]
    X_test_elim = X_test[selected_feat]
    simple_regression(X_train_elim, X_test_elim, y_train, y_test)

def joined_and_lasso(X_train, X_test, y_train, y_test):
    alpha = 0.2
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train) 
    y_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    train_score = mean_absolute_error(y_pred, y_train)
    test_score = mean_absolute_error(y_test_pred, y_test)
    coeff_used = np.sum(lasso.coef_!=0)
    print("LASSO alpha {} Train score: {}, Test score: {}".format(alpha, train_score, test_score))
    print("LASSO Num of coeff used: {}".format(coeff_used))

def randomForestRegression(X_train, X_test, y_train, y_test, history=None):
    
    rfr = Pipeline([
        #("normalizer", Normalizer()),
        ("rfr", RandomForestRegressor(n_estimators=100, random_state=7))
        ])
    parameters = {
        'rfr__max_depth': [2,5,10,20],
        'rfr__min_samples_split': [2,5],
        'rfr__max_leaf_nodes': [5,10,20],
        'rfr__min_samples_leaf': [2,5],
        #'rfr__max_samples': [0.2,0.5,1],
    }
    greg = GridSearchCV(estimator = rfr, param_grid=parameters, cv=5, verbose=3)
    greg.fit(X_train, y_train)
    best_param = greg.best_params_
    best_score = greg.best_score_
    best_estim = greg.best_estimator_

    print(best_param)
    print(best_score)
    print(best_estim)
    print(X_train.shape)

def scatter_plot(feature, target):
    plt.figure(figsize = (16, 8))
    plt.scatter(data[feature], data[target], c='black')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()

if __name__ == "__main__":
    print("Loading & splitting bio data")
    df = pd.read_csv(os.path.join(data_path, "base_processed.csv"))
    #df = EU.quantize(df, excluding = ['logp'])
    df['molecule_id'] = pd.Categorical(df['molecule_id']).codes
    df_joined = df_joined.fillna(0)

    X = df
    Y = df['logp'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = test_size)
    
    X_joined = df_joined
    Y_joined = df_joined['logp'].values.reshape(-1, 1)
    X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(X_joined, Y_joined, test_size = test_size)
    
    X_train = X_train.drop(["logp"], axis = 1)
    X_test = X_test.drop(["logp"], axis = 1)
    X_train_j = X_train_j.drop(["logp"], axis = 1)
    X_test_j = X_test_j.drop(["logp"], axis = 1)
    
    
    
    # # Baseline 1: simple regression 
    # print("Baseline 1: Simple Regression")
    # simple_regression(X_train, X_test, y_train, y_test)
    # joined_and_lasso(X_train, X_test, y_train, y_test)

    # # Baseline 2: Join all tables & regression
    # print("Baseline 2: Joined & Regression")
    # simple_regression(X_train_j, X_test_j, y_train_j, y_test_j) 
    # # for i in range(1, X_train_j.shape[1]):
    #     # joined_and_feature_elim_random_forest(X_train_j, X_test_j, y_train_j, y_test_j, kk=i) 

    # # Baseline 3: Join all tables & LASSO
    # print("Baseline 3: Joined & LASSO")
    # joined_and_lasso(X_train_j, X_test_j, y_train_j, y_test_j)

    # # # Baseline 4: Join all tables & RandomForest
    # # print("Baseline 3: Joined & RandomForest")
    # # randomForestRegression(X_train_j, X_test_j, y_train_j, y_test_j)
    # EU.regression_task_nn(X_train_j, X_test_j, y_train_j, y_test_j)
    EU.regression_task_nn(X_train, X_test, y_train, y_test)
    EU.regression_task_nn(X_train_j, X_test_j, y_train_j, y_test_j)