import json
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import word2vec
import visualizer as VS
import eval_utils as EU
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import sys


# from keras.models import Sequential
# from keras import layers

embedding_storage = {
    "node2vec": '../node2vec/emb/',
    "ProNE": '../ProNE/emb/'
}

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]
test_size = embeddding_config["test_size"]


def classification_task(X_train, X_test, y_train, y_test, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators)
    logr = LogisticRegression(penalty='l2', solver='liblinear')

    rf.fit(X_train, y_train)
    logr.fit(X_train, y_train)

    def show_stats(model):
        pscore_train = accuracy_score(y_train, model.predict(X_train))
        pscore_test = accuracy_score(y_test, model.predict(X_test))
        print("Train accuracy {}, Test accuracy {}".format(
            pscore_train, pscore_test))

    show_stats(rf)
    show_stats(logr)


def classification_task_nn(task, X_train, X_test, y_train, y_test):
    input_size = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size,)),
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

    model.fit(X_train, y_train, epochs=25)
    results = model.evaluate(X_test, y_test)
    print(results)


def evaluate_task(args):
    # Load task config information
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)
    with open("../data/strategies/" + args.task + ".txt", "r") as jsonfile:
        strategies = json.load(jsonfile)
    config = data_config[args.task]
    location = config["location"]
    target_file = config["target_file"]
    location_processed = config["location_processed"]
    target_column = config["target_column"]

    # Load data
    trimmed_table = pd.read_csv(os.path.join(
        "../", location_processed), sep=',', encoding='latin')
    full_table = pd.read_csv(os.path.join(
        "../", location + target_file), sep=',', encoding='latin')
    Y = full_table[target_column]

    if args.task in ["kraken", "financial", "genes"]:
        Y = pd.Categorical(Y).codes
    if args.task == "sample":
        Y = Y.apply(lambda x: int(x / 200))

    # Set embeddings that are to be evaluated
    method = args.method
    all_embeddings_path = EU.all_files_in_path(
        embedding_storage[method], args.task)

    # Run through the embedding list and do evaluation
    for path in all_embeddings_path:
        model = KeyedVectors.load_word2vec_format(path)
        table_name = path.split("/")[-1][:-4]
        model_dict_path = "../graph/{}/{}.dict".format(args.task, table_name)

        # Obtain textified & quantized data
        df_textified = EU.textify_df(
            trimmed_table, strategies, location_processed)
        x_vec = pd.DataFrame(EU.vectorize_df(
            df_textified, model, model_dict=model_dict_path, model_type=method))
        x_vec = x_vec.dropna(axis=1)

        # Train a Random Forest classifier
        # remove_hubness_and_run(x_vec, Y)
        X_train, X_test, y_train, y_test = train_test_split(
            x_vec, Y, test_size=0.1, random_state=10)
        print("Evaluating model", path)
        # EU.remove_hubness_and_run(x_vec, Y)

        for n_estimators in [10, 50, 100]:
            classification_task(X_train, X_test, y_train,
                                y_test, n_estimators=n_estimators)
        # classification_task_nn(args.task, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    print("Evaluating results with word2vec model:")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='task to be evaluated on'
    )
    parser.add_argument(
        '--method',
        type=str,
        help='method of training'
    )

    args = parser.parse_args()

    print("Evaluating on task {}".format(args.task))
    evaluate_task(args)
