import argparse
import pandas as pd
import csv
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
import networkx as nx
import sys 

sys.path.append('..')
from relational_embedder.data_prep import data_prep_utils as dpu

graph = nx.Graph()
INT_TYPE = [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]

def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store"]
    return fs

# Assume that textification strategy is skip (integer), row & col, sequence text
def textify_df_row(df, integer_strategy = 'skip'):
    columns = df.columns
    rows = [] 
    for index, el in df.iterrows():
        curr_row = []
        for c in columns:
            cell_value = el[c]
            if (cell_value == "\\N"):
                continue
            if not dpu.valid_cell(cell_value):
                continue
            if df[c].dtype in INT_TYPE:
                if integer_strategy == 'skip':
                    continue 
            curr_row.append(dpu.encode_cell((cell_value, index), grain='cell'))
        rows.append(curr_row)
    return rows

def textify_df_col(df, integer_strategy = 'skip'):
    columns = df.columns
    cols = [] 
    for c in columns:
        curr_col = []
        for cell_value in df[c]:
            if (cell_value == "\\N"):
                continue
            if not dpu.valid_cell(cell_value):
                continue
            if df[c].dtype in INT_TYPE:
                if integer_strategy == 'skip':
                    continue 
            curr_col.append(dpu.encode_cell((cell_value, c), grain='cell'))
        cols.append(curr_col)
    return cols

def generate_graph(args):
    path = args.dataset
    output = args.output 
    
    fs = all_files_in_path(path)
    total = len(fs)
    current = 0 
    for path in tqdm(fs):
        df = pd.read_csv(path, encoding = 'latin1', sep='\t')
        if not dpu.valid_relation(df): 
            continue 
        columns = df.columns 
        textified_rows = textify_df_row(df)
        for row in textified_rows:
            for col in range(len(row) - 1):
                for entry_x in row[col]:
                    for entry_y in row[col+1]:
                        graph.add_edge(entry_x, entry_y, weight = 3)
        
        textified_cols = textify_df_col(df)
        for col in textified_cols:
            for row in range(len(col) - 1):
                for entry_x in row[col]:
                    for entry_y in row[col+1]:
                        graph.add_edge(entry_x, entry_y, weight = 1)
                    
    nx.write_edgelist(graph, args.output)

if __name__ == "__main__":
    print("Generating graph for input")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
        type=str, 
        default='../data/1/', # This is my small dataset
        help='path to collection of relations'
    )

    parser.add_argument('--output', 
        type=str, 
        default='./graph_textified.edgelist'
    )

    args = parser.parse_args()

    # Generate and Save graph 
    generate_graph(args)
    print("Done! saved under ./graph")