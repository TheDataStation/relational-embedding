import csv
import sys
import pandas as pd
import numpy as np 
from relational_embedder.data_prep import data_prep_utils as dpu
from textification import textify_relation as tr

name_basics_file = "../data/imdb/name.basics.tsv"

# Assume that textification strategy is skip (integer), row & col, sequence text
def textify_df(df):
    columns = df.columns
    query_after_textification = [] 
    for index, el in df.iterrows():
        query_row = []
        for c in columns:
            cell_value = el[c]
            if (cell_value == "\\N"):
                continue
            if not dpu.valid_cell(cell_value):
                continue
            if df[c].dtype in [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]:
                if integer_strategy == 'skip':
                    continue 
            query_row.append(dpu.encode_cell((cell_value, c), grain='cell'))
        query_after_textification.append(query_row)
    return query_after_textification

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

if __name__ == "__main__":
    print("Evaluation Utils:")

    # test 1 : obtain grouth truth for missing names 
    print("Obtain ground truth for missing names")
    print(obtain_ground_truth_name(100))