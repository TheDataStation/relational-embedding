from gensim.models import Word2Vec, KeyedVectors
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import eval_utils as EU
import visualizer as VS
import word2vec
import os
import argparse
import json 

# from keras.models import Sequential
# from keras import layers

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

node2vec_embedding_storage = '../node2vec/emb/'

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]
test_size = embeddding_config["test_size"]

def classification_task(X_train, X_test, y_train, y_test, n_estimators=100):
    rf = RandomForestClassifier(n_estimators = n_estimators)
    logr = LogisticRegression(penalty='l2', solver='liblinear')

    rf.fit(X_train, y_train)
    logr.fit(X_train, y_train)
    def show_stats(model):
        pscore_train = accuracy_score(y_train, model.predict(X_train))
        pscore_test = accuracy_score(y_test, model.predict(X_test))
        print("Train accuracy {}, Test accuracy {}".format(pscore_train, pscore_test))
    show_stats(rf)
    show_stats(logr)



def classification_task_nn(task, X_train, X_test, y_train, y_test): 
    input_size = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (input_size,)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=['accuracy', tf.keras.metrics.CategoricalCrossentropy()]
    )

    model.fit(X_train, y_train, epochs = 25)
    results = model.evaluate(X_test, y_test)
    print(results)

def regression_task(): 
    return None

def evaluate_task(args):
    # Load task config information 
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)
    config = data_config[args.task]
    location = config["location"]
    target_file = config["target_file"]
    location_processed = config["location_processed"]
    target_column = config["target_column"]

    with open("../data/strategies/" + args.task + ".txt", "r") as jsonfile:
        strategies = json.load(jsonfile)

    # Load data 
    trimmed_table = pd.read_csv(os.path.join("../", location_processed), sep=',', encoding='latin')
    full_table = pd.read_csv(os.path.join("../", location + target_file), sep=',', encoding='latin')
    Y = full_table[target_column]

    if args.task in ["kraken", "financial", "genes"]:
        Y = pd.Categorical(Y).codes
    if args.task == "sample":
        Y = Y.apply(lambda x: int(x / 200))

    # Set embeddings that are to be evaluated 
    all_embeddings_path = [] 
    if args.embedding is not None:
        all_embeddings_path = [args.embedding]
    else: 
        all_embeddings_path = EU.all_files_in_path(node2vec_embedding_storage, args.task)

    # Run through the embedding list and do evaluation 
    for path in all_embeddings_path:
        model = KeyedVectors.load_word2vec_format(path)
        textification_strategy, integer_strategy = EU.parse_strategy(path)

        # Obtain textified & quantized data
        df_textified = EU.textify_df(trimmed_table, textification_strategy, strategies, location_processed)
        import pdb; pdb.set_trace()
        x_vec = pd.DataFrame(EU.vectorize_df(df_textified, model, model_type = "node2vec"))
        import pdb; pdb.set_trace()
        x_vec = x_vec.dropna(axis=1)

        # Train a Random Forest classifier
        # remove_hubness_and_run(x_vec, Y)
        X_train, X_test, y_train, y_test = train_test_split(x_vec, Y, test_size = 0.1, random_state=10)
        print("Evaluating model", path)
        # EU.remove_hubness_and_run(x_vec, Y)

        for n_estimators in [10, 50, 100]:
            classification_task(X_train, X_test, y_train, y_test, n_estimators=n_estimators)
        # classification_task_nn(args.task, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    print("Evaluating results with word2vec model:")

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to be evaluated on')
    parser.add_argument('--embedding', 
        type=str, 
        help='Pass in single a w2v embedding to evaluate instead of doing evaluation on everything'
    )

    args = parser.parse_args()

    print("Evaluating on task {}".format(args.task))
    evaluate_task(args)