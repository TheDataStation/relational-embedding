import csv
import sys
import pandas as pd
import numpy as np 
from relational_embedder.data_prep import data_prep_utils as dpu
from textification import textify_relation as tr

from os.path import isfile, join
from os import listdir

num_bins = 10

def all_files_in_path(path, task):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store" and f.find(task) != -1]
    return fs

def quantize(df, excluding = [], hist = "width", num_bins = num_bins):
    cols = df.columns
    bin_percentile = 100 / num_bins
    for col in cols:
        if col in excluding:
            continue
        if df[col].dtype not in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
            continue 
        
        if hist == "width":
            bins = [np.percentile(df[col], i * bin_percentile) for i in range(num_bins)]
        else: 
            bins = [i * (df[col].max() - df[col].min()) / num_bins for i in range(num_bins)]
        
        df[col] = np.digitize(df[col], bins)
    return df

def vectorize_df(df, model, model_type = "word2vec"):
    length = len(df)
    x_vectorized = [[] for i in range(length)]
    if model_type == "word2vec":
        for i in range(length):
            row = df[i]
            for j in range(len(row)):
                if row[j] in model.vocab:
                    x_vectorized[i] += list(model[row[j]])
    
    if model_type == "node2vec":
        for i in range(length):
            x_vectorized[i] += list(model["row:" + str(i)])
    return x_vectorized

# Assume that textification strategy is skip (integer), row & col, sequence text
def textify_df(df, ts = "row_and_col", integer_strategy = "quantize"):
    df = quantize(df, excluding = ["event_id", "result"], num_bins=num_bins)
    columns = df.columns
    input = [[] for i in range(df.shape[0])]
    # Rows
    if ts == "row_and_col" or ts == "row":
        for cell_value in tr._read_rows_from_dataframe(df, columns, integer_strategy=integer_strategy):
            values = dpu.encode_cell(cell_value, grain="cell")
            for cv in values:
                # f.write(" " + cv)
                input[cell_value[1]].append(cv)

    # Columns
    if ts == "row_and_col" or ts == "col":
        cnt = 0 
        for cell_value in tr._read_columns_from_dataframe(df, columns, integer_strategy=integer_strategy):
            values = dpu.encode_cell(cell_value, grain="cell")
            for cv in values:
                # f.write(" " + cv)
                input[cnt].append(cv) 
                cnt = (cnt + 1) % df.shape[0]
    
    if ts == "alex": 
        cnt = 0
        offset = 0
        for cell_value, c, index in tr.alex__read_rows_from_dataframe(df, columns, integer_strategy=integer_strategy):
            values = dpu.encode_cell((cell_value, c), grain="cell")
            for cv in values:
                input[cnt].append(cv)
                offset += 1 
                if offset >= df.shape[1]: 
                    offset = 0
                    cnt += 1
    return input

def obtain_ground_truth_name(sample_size = 1000):
    df = pd.read_csv(name_basics_file, encoding = 'latin1', sep = '\t')
    sample = df.head(n = sample_size)

    ground_truth = textify_df(sample[['primaryName']])
    sample_query = textify_df(sample.drop(['primaryName'], axis = 1))
    return sample_query, ground_truth

# def obtain_ground_truth_profession(sample_size = 1000):
#     df = pd.read_csv(name_basics_file, encoding = 'latin1', sep = '\t')
#     sample = df.sample(n = sample_size)

#     ground_truth = textify_df(sample['primaryProfession'])
#     sample_query = textify_df(sample.drop(['primaryProfession'], axis = 1)
#     return sample_query.values.tolist(), ground_truth.values.tolist()

def measure_quality(ground_truth, predicted_truth):
    precision = [] 
    for i in range(len(ground_truth)):
        flag = ground_truth[i] in predicted_truth[i]
        precision.append(flag)
    return precision

def remove_hubness_and_run(X, y):
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
    knn_standard = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    acc_standard = cross_val_score(knn_standard, X, y, cv=5)

    # kNN with hubness reduction (mutual proximity)
    knn_mp = KNeighborsClassifier(n_neighbors=10,
                              metric='cosine',
                              hubness='mutual_proximity')
    acc_mp = cross_val_score(knn_mp, X, y, cv=10)

    print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')
    print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')

def parse_strategy(s):
    integer_strategy = "quantize"
    textification_strategy = "row"

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
