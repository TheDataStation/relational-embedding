import csv
import sys
import pandas as pd
import numpy as np 
from relational_embedder.data_prep import data_prep_utils as dpu
from textification import textify_relation as tr

name_basics_file = "../data/imdb/name.basics.tsv"

def quantize(df, excluding = [], hist = "width", num_bins = 20):
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

def vectorize_df(df, model):
    length = len(df)
    x_vectorized = [[] for i in range(length)]
    y_vectorized = [[] for i in range(length)]
    for i in range(length):
        row = df[i]
        for j in range(len(row)):
            if j == 3 or j == 7: 
                y_vectorized[i] += list(model[row[j]])
            else:
                x_vectorized[i] += list(model[row[j]])
    return x_vectorized, y_vectorized

# Assume that textification strategy is skip (integer), row & col, sequence text
def textify_df(df):
    df = quantize(df, excluding = ["event_id", "result"])
    columns = df.columns
    input = [[] for i in range(df.shape[0])]
    # Rows
    for cell_value in tr._read_rows_from_dataframe(df, columns, integer_strategy="stringify"):
        values = dpu.encode_cell(cell_value, grain="cell")
        for cv in values:
            # f.write(" " + cv)
            input[cell_value[1]].append(cv)
    # Columns
    cnt = 0 
    for cell_value in tr._read_columns_from_dataframe(df, columns, integer_strategy="stringify"):
        values = dpu.encode_cell(cell_value, grain="cell")
        for cv in values:
            # f.write(" " + cv)
            input[cnt].append(cv) 
            cnt = (cnt + 1) % df.shape[0]
    return input
                
    # columns = df.columns
    # query_after_textification = [] 
    # for index, el in df.iterrows():
    #     query_row = []
    #     for c in columns:
    #         cell_value = el[c]
    #         if (cell_value == "\\N"):
    #             continue
    #         query_row.append(dpu.encode_cell((cell_value, c), grain='cell'))
    #     query_after_textification.append(query_row)
    # return query_after_textification

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