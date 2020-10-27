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
import textification.textify_relation as tr 

graph = nx.Graph()
INT_TYPE = [np.int64, np.int32, np.int64, np.float, np.int, np.float16, np.float32, np.float64]

def all_files_in_path(path):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store" and f != "base_processed.csv"]
    return fs

def generate_graph(args):
    path = args.dataset
    output = args.output 
    
    fs = all_files_in_path(path)
    total = len(fs)
    current = 0 
    for path in tqdm(fs):
        df = pd.read_csv(path, encoding = 'latin1', sep=',')
        df = tr.quantize(df, excluding = ["eventid", "result"])

        columns = df.columns 

        for cell_value, row in tr._read_rows_from_dataframe(df, columns, integer_strategy="stringify"):
            decoded_row = dpu.encode_cell(row, grain="cell")
            decoded_value = dpu.encode_cell(cell_value, grain="cell")
            for value in decoded_value:
                for row in decoded_row:
                    if (value, "row:" + row) in graph.edges():
                        graph[value]["row:" + row]['weight'] += 1 
                    else:
                        graph.add_edge(value, "row:" + row, weight = 1)
    
        for cell_value, col in tr._read_columns_from_dataframe(df, columns, integer_strategy="stringify"):
            decoded_col = dpu.encode_cell(col, grain="cell")
            decoded_value = dpu.encode_cell(cell_value, grain="cell")
            for value in decoded_value:
                for col in decoded_col:
                    if (value, "col:" + col) in graph.edges():
                        graph[value]["col:" + col]['weight'] += 1 
                    else:
                        graph.add_edge(value, "col:" + col, weight = 1)         

    nx.write_edgelist(graph, args.output)

if __name__ == "__main__":
    print("Generating graph for input")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
        type=str, 
        default='./sample/', # This is my small dataset
        help='path to collection of relations'
    )

    parser.add_argument('--output', 
        type=str, 
        default='./graph/textified.edgelist'
    )

    args = parser.parse_args()
    # Generate and Save graph 
    generate_graph(args)
    print("Done! saved under ./graph")