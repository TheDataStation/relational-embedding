import argparse
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import networkx as nx
import sys 
import json

sys.path.append('..')
from relational_embedder.data_prep import data_prep_utils as dpu
import textification.textify_relation as tr 
from collections import defaultdict 

def all_files_in_path(path):
    path = os.path.join("../", path)
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store" and f != "base_processed.csv"]
    return fs

def generate_graph(args):
    task = args.task
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)

    output = args.output 
    
    fs = all_files_in_path(data_config[task]["location"])
    total = len(fs)
    edges = defaultdict()

    current = 0 
    for path in tqdm(fs):
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)
        df = tr.quantize(df, excluding = ["eventid", "result"])
        filename = path.split("/")[-1]

        columns = df.columns 

        for cell_value, row in tr._read_rows_from_dataframe(df, columns, integer_strategy="stringify"):
            decoded_row = dpu.encode_cell(row, grain="cell")
            decoded_value = dpu.encode_cell(cell_value, grain="cell")
            for value in decoded_value:
                for row in decoded_row:
                    row = filename + "row:" + row
                    if (value, row) in edges:
                        edges[(value, row)] += 1
                    else:
                        edges[(value, row)] = 1
    
        for cell_value, col in tr._read_columns_from_dataframe(df, columns, integer_strategy="stringify"):
            decoded_col = dpu.encode_cell(col, grain="cell")
            decoded_value = dpu.encode_cell(cell_value, grain="cell")
            for value in decoded_value:
                for col in decoded_col:
                    col = filename + "col:" + col
                    if (value, col) in edges:
                        edges[(value, col)] += 0.3
                    else:
                        edges[(value, col)] = 0.3    

    print("Adding edges now:")
    graph = nx.Graph()
    for edge, val in edges.items():
        graph.add_edge(edge[0], edge[1], weight=val)
    nx.write_edgelist(graph, args.output)

if __name__ == "__main__":
    print("Generating graph for input")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        type=str, 
        default='sample', # This is my small dataset
        help='task to generate relation from'
    )

    parser.add_argument('--output', 
        type=str, 
        default='./graph/textified.edgelist'
    )

    args = parser.parse_args()
    # Generate and Save graph 
    generate_graph(args)
    print("Done! saved under ./graph")