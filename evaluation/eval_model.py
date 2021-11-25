import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.model_selection import cross_val_score, train_test_split
import json
import argparse
import eval_utils as EU
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import word2vec
import visualizer as VS
from gensim.models import Word2Vec, KeyedVectors
import sys
from sklearn.linear_model import Lasso, LinearRegression


def check_if_eval_path(path, task, suffix):
    # task = "genes", suffix = "": only runs genes.emb 
    # task = "genes", suffix = "suf", only runs genes_suf.emb
    # task = "genes", suffix = "all", run all genes_*.emb
    if suffix == "all": return True 
    if suffix != "" and task + "_" + suffix not in path:
        return False 
    if suffix == "" and task + ".emb" not in path:
        return False
    return True 

def get_emb_name_from_path(path):
    emb_name = path.split("/")[-1]
    emb_name = emb_name[:emb_name.find(".emb")]
    
    if "_sparse" in emb_name or "_spectral" in emb_name or "_restart" in emb_name:
        emb_name = "_".join(emb_name.split("_")[:-1])
    return emb_name 

embedding_storage = {"node2vec": '../node2vec/emb/', "ProNE": '../ProNE/emb/'}

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]
test_size = embeddding_config["test_size"]


def evaluate_task(args):
    # Load task config information
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)
    with open("../data/strategies/" + args.task + ".txt", "r") as jsonfile:
        strategies = json.load(jsonfile)
    config = data_config[args.task]
    location = config["location"]
    target_file = config["target_file"]
    task_type = config["task_type"]
    location_processed = config["location_processed"]
    target_column = config["target_column"]

    # Load data
    trimmed_table = pd.read_csv(os.path.join("../", location_processed),
                                sep=',',
                                encoding='latin')
    full_table = pd.read_csv(os.path.join("../", location + target_file),
                             sep=',',
                             encoding='latin')

    Y = full_table[target_column[0]]
    if task_type == "classification":
        Y = pd.Categorical(Y).codes

    # Set embeddings that are to be evaluated
    method = args.method
    all_embeddings_path = EU.all_files_in_path(
        embedding_storage[method],
        args.task
    )

    # Run through the embedding list and do evaluation
    for path in all_embeddings_path:
        if check_if_eval_path(path, args.task, args.suffix) == False:
            continue 
        emb_name = get_emb_name_from_path(path)
        model = KeyedVectors.load_word2vec_format(path)
        model_dict_path = "../graph/{}/{}.dict".format(args.task, emb_name)

        print("Evaluating: dict, ", model_dict_path)
        print("emb path,", path)
        # Obtain textified & quantized data
        training_loss = []
        testing_loss = []
        df_textified = EU.textify_df(trimmed_table, strategies,
                                     location_processed)
        x_vec = EU.vectorize_df(df_textified,
                                model,
                                file=location_processed.split("/")[-1],
                                model_dict=model_dict_path,
                                model_type=method)
        tests = train_test_split(x_vec,
                                 Y,
                                 test_size=test_size,
                                 random_state=10)
        if task_type == "classification":
            # print("nn:")
            # train_loss, test_loss = EU.classification_task_nn(
            #     *tests, history_name=emb_name)
            # print("logistic reg:")
            # train_loss, test_loss = EU.classification_task_logr(*tests)
            print("Random Forest:")
            train_loss, test_loss = EU.classification_task_rf(*tests)
        else:
            print("linear net:")
            train_loss, test_loss = EU.linearRegression(*tests)
            # print("lasso:")
            # train_loss, test_loss = EU.lassoRegression(*tests)
            # print("random forest:")
            # train_loss, test_loss = EU.randomForestRegression(*tests)
            print("elastic net:")
            train_loss, test_loss = EU.elasticNetRegression(*tests)
            print("neural net:")
            train_loss, test_loss = EU.regression_task_nn(*tests)
            print()


if __name__ == "__main__":
    print("Evaluating results with model:")

    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        required=True,
                        help='task to be evaluated on')

    parser.add_argument('--suffix', type=str, default="",
                        help='suffix of training experiment')
    parser.add_argument('--method', type=str,
                        default="node2vec", help='method of training')

    args = parser.parse_args()

    print("Evaluating on task {}".format(args.task))
    evaluate_task(args)
