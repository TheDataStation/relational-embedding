import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import eval_utils as EU
import visualizer as VS
from gensim.models import Word2Vec, KeyedVectors

import os
from keras.models import Sequential
from keras import layers

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

K = 20 
node2vec_model_path_kraken = "../node2vec/emb/kraken.emb"
kraken_path = "../data/kraken/"
node2vec_model_path_school = "../node2vec/emb/school.emb"
test_size = 0.2
num_bins = 10

def classification_task(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators = 1000)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    pscore = accuracy_score(y_test, y_pred)
    print("RF Test score:", pscore)

def classification_task_nn(X_train, X_test, y_train, y_test): 
    input_size = len(X_train[0])
    model = Sequential([
                layers.Flatten(input_shape = (input_size,)),
                layers.Dense(64, activation = tf.nn.sigmoid),                        
                layers.Dense(20, activation = tf.nn.sigmoid)
            ])
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics =['accuracy']
    )
    X_train = np.asarray(X_train) 
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    model.fit(np.asarray(X_train), np.asarray(y_train), epochs = 25)
    results = model.evaluate(np.asarray(X_test), np.asarray(y_test))
    print(results)

def regression_task(): 
    return None
 
def kraken_task():
    # Load model 
    model = KeyedVectors.load_word2vec_format(node2vec_model_path_kraken)

    # Obtain textified & quantized data
    df = pd.read_csv(os.path.join(kraken_path, "kraken.csv"))
    df_textified = EU.textify_df(df, "alex")
    print(df_textified[0])
    x_vec, y_vec = EU.vectorize_df(df_textified, model, 4, model_type = "node2vec")
    df2 = pd.read_csv(os.path.join(kraken_path, "base_processed.csv"), sep=',', encoding='latin')
    df2["result"] = (df2["result"] == "nofail")
    Y = df2['result'].values.ravel()

    EU.remove_hubness_and_run(x_vec, Y)

    # Train a Random Forest classifier
    X_train, X_test, y_train, y_test = train_test_split(x_vec, Y, test_size = test_size, random_state=10)

    classification_task(X_train, X_test, y_train, y_test)
    # classification_task_nn(X_train, X_test, y_train, y_test)


def school_task():
    # Load model 
    print("Obtain model")
    model = word2vec.load_word2vec_format(node2vec_model_path_school)

    # Obtain textified & quantized data
    df = pd.read_csv("../data/school/base.csv")
    df_textified = EU.textify_df(df, "row_and_col")
    x_vec, y_vec = EU.vectorize_df(df_textified, model, 7, model_type = "node2vec")
    Y = df['class'].values.ravel()

    # Train a Random Forest classifier
    print("Training")
    X_train, X_test, y_train, y_test = train_test_split(x_vec, Y, test_size = test_size, random_state=10)
    classification_task(X_train, X_test, y_train, y_test)
 

if __name__ == "__main__":
    print("Evaluating results with node2vec model:")
    print("Kraken Dataset:")
    kraken_task()

    # print("School dataset")
    # school_task()
