import sys
sys.path.append('..')
import csv
import pandas as pd
import numpy as np 
from relational_embedder.data_prep import data_prep_utils as dpu
from textification import textify_relation as tr

from os.path import isfile, join
from os import listdir
import json

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]


def all_files_in_path(path, task):
    fs = [join(path, f) for f in listdir(path)
          if isfile(join(path, f)) and f != ".DS_Store"
          and f.find(task) != -1 and f[-4:] == ".emb"]
    return fs


def get_PCA_for_embedding(model, ndim=2):
    from sklearn.decomposition import PCA
    words = list(model.vocab.keys())
    word_vectors = np.array([model[w] for w in words])

    twodim = pd.DataFrame(
        PCA().fit_transform(word_vectors)[:, :ndim],
        columns=["d" + str(i+1) for i in range(ndim)]
    )
    twodim.index = words
    return twodim.T


def vectorize_df(df, model, model_vocab=None, model_dict=None, model_type="word2vec"):
    length = len(df)
    x_vectorized = [[] for i in range(length)]

    if model_type == "node2vec" or model_type == "ProNE":
        from token_dict import TokenDict
        cc = TokenDict()
        cc.load(model_dict)
        for i in range(length):
            row = df[i]
            for j in range(len(row)):
                if cc.check(row[j]) in model_vocab:
                    x_vectorized[i] += list(model[cc.check(row[j])])
    return x_vectorized

def textify_df(df, strategies, path):
    table_name = path.split("/")[-1]
    df = tr.quantize(df, strategies[table_name])
    columns = df.columns
    input = [[] for i in range(df.shape[0])]

    for cell_value, row, col in tr._read_rows_from_dataframe(df, columns, strategies[table_name]):
        grain_strategy = strategies[table_name][col]["grain"]
        decoded_row = dpu.encode_cell(row, grain=grain_strategy)
        decoded_value = dpu.encode_cell(cell_value, grain=grain_strategy)
        for value in decoded_value:
            input[row].append(value)

    filename = "".join(table_name.split(".")[:-1])
    for row in range(df.shape[0]):
        row_name = "{}_row:{}".format(filename, str(row))
        input[row].append(row_name)
    return input


def measure_quality(ground_truth, predicted_truth):
    precision = []
    for i in range(len(ground_truth)):
        flag = ground_truth[i] in predicted_truth[i]
        precision.append(flag)
    return precision


def remove_hubness_and_run(X, y, n_neighbors=15):
    from skhubness import Hubness
    from sklearn.model_selection import cross_val_score
    from skhubness.neighbors import KNeighborsClassifier

    # Measure Hubness before and after removal (method mutual proximity)
    hub = Hubness(k=10, metric='cosine')
    hub.fit(X)
    k_skew = hub.score()
    print(f'Skewness = {k_skew:.3f}')

    hub_mp = Hubness(k=10, metric='cosine',
                     hubness='mutual_proximity')
    hub_mp.fit(X)
    k_skew_mp = hub_mp.score()
    print(f'Skewness after MP: {k_skew_mp:.3f} '
          f'(reduction of {k_skew - k_skew_mp:.3f})')
    print(f'Robin hood: {hub_mp.robinhood_index:.3f} '
          f'(reduction of {hub.robinhood_index - hub_mp.robinhood_index:.3f})')

    # Measure Classfication Accuracy before and after removal
    # vanilla kNN
    knn_standard = KNeighborsClassifier(
        n_neighbors=n_neighbors, metric='cosine')
    acc_standard = cross_val_score(knn_standard, X, y, cv=10)

    # kNN with hubness reduction (mutual proximity)
    knn_mp = KNeighborsClassifier(n_neighbors=n_neighbors,
                                  metric='cosine',
                                  hubness='mutual_proximity')
    acc_mp = cross_val_score(knn_mp, X, y, cv=10)

    print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')
    print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')
    return (acc_standard.mean(), acc_mp.mean())


def parse_strategy(s):
    integer_strategy = "quantize"
    textification_strategy = "row_and_col"

    if s.find("alex") != -1:
        textification_strategy = "alex"
    elif s.find("row_and_col") != -1:
        textification_strategy = "row_and_col"
    elif s.find("row") != -1:
        textification_strategy = "row"
    elif s.find("col") != -1:
        textification_strategy = "col"

    if s.find("quantize") != -1:
        integer_strategy = "quantize"
    elif s.find("stringify") != -1:
        integer_strategy = "stringify"

    return textification_strategy, integer_strategy
