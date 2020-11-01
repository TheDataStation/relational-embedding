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

from keras.models import Sequential
from keras import layers

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

K = 20 

word2vec_embedding_storage = '../word2vec/emb/'

word2vec_model_path_kraken = "../word2vec/emb/kraken_textified_row_and_col_quantize.emb"
kraken_path = "../data/kraken/"
word2vec_model_path_school = "../word2vec/school.bin"
test_size = 0.1
num_bins = 20

def classification_task(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators = 1000)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    import pdb;pdb.set_trace();
    pscore = accuracy_score(y_test, y_pred)
    print("RF Test score:", pscore)

def classification_task_nn(X_train, X_test, y_train, y_test): 
    input_size = len(X_train[0])
    model = Sequential([
                layers.Flatten(input_shape = (input_size,)),
                layers.Dense(64, activation = tf.nn.sigmoid),
                layers.Dense(64, activation = tf.nn.sigmoid),                        
                layers.Dense(10, activation = tf.nn.softmax)
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

def evaluate_task(args):
    # Load task config information 
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)
    config = data_config[args.task]
    location = config["location"]
    target_file = config["target_file"]
    location_processed = config["location_processed"]
    target_column = config["target_column"]

    # Load data 
    trimmed_table = pd.read_csv(os.path.join("../", location_processed), sep=',', encoding='latin')
    full_table = pd.read_csv(os.path.join("../", location + target_file), sep=',', encoding='latin')
    Y = full_table[target_column]

    # Set embeddings that are to be evaluated 
    all_embeddings_path = [] 
    if args.embedding is not None:
        all_embeddings_path = [args.embedding]
    else: 
        all_embeddings_path = EU.all_files_in_path(word2vec_embedding_storage, args.task)

    # Run through the embedding list and do evaluation 
    for path in all_embeddings_path:
        model = KeyedVectors.load_word2vec_format(path)
        textification_strategy, integer_strategy = EU.parse_strategy(path)

        # Obtain textified & quantized data
        df_textified = EU.textify_df(trimmed_table, textification_strategy, integer_strategy)
        x_vec = pd.DataFrame(EU.vectorize_df(df_textified, model, model_type = "word2vec"))
        x_vec = x_vec.dropna(axis=1)

        # Train a Random Forest classifier
        # remove_hubness_and_run(x_vec, Y)
        X_train, X_test, y_train, y_test = train_test_split(x_vec, Y, test_size = test_size, random_state=10)
        print("Evaluating model", path)
        # EU.remove_hubness_and_run(x_vec, Y)

        classification_task(X_train, X_test, y_train, y_test)
        classification_task_nn(X_train, X_test, y_train, y_test)

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