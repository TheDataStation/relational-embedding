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

class Counter:
    def __init__(self):
        self.vocab = dict()
        self.cnt = 0
    
    def get(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.cnt 
            self.cnt += 1 
        return str(self.vocab[token])
    
    def save(self, output_path):
        import pickle
        f = open(output_path,"wb")
        pickle.dump(self.vocab, f)
        f.close()

    def load(self, input_path):
        import pickle
        f = open(input_path, "rb")
        self.vocab = pickle.load(f)
        self.cnt = len(self.vocab) - 1

def generate_graph(args):
    task = args.task
    output = "./graph/{}.edgelist".format(task)
    with open("../data/data_config.txt", "r") as jsonfile:
        data_config = json.load(jsonfile)
    fs = all_files_in_path(data_config[task]["location"])
    total = len(fs)
    edges = set()

    current = 0 
    for path in tqdm(fs):
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)
        df = tr.quantize(df, excluding = ["eventid", "result"])
        table_name = path.split("/")[-1]
        columns = df.columns 

        for cell_value, row in tr._read_rows_from_dataframe(df, columns, integer_strategy="stringify"):
            decoded_row = dpu.encode_cell(row, grain="cell")
            decoded_value = dpu.encode_cell(cell_value, grain="cell")
            for value in decoded_value:
                for row in decoded_row:
                    filename = table_name[:-4]
                    row_name = "row:" + row
                    edges.add((value, row_name))
    
        for cell_value, col in tr._read_columns_from_dataframe(df, columns, integer_strategy="stringify"):
            decoded_col = dpu.encode_cell(col, grain="cell")
            decoded_value = dpu.encode_cell(cell_value, grain="cell")
            for value in decoded_value:
                for col in decoded_col:
                    col_name = "col:" + col
                    edges.add((value, col_name))

    graph = nx.Graph()
    cc = Counter()

    print("Edge number:", len(edges)) 
    with open(output + ".txt", "w") as f:
        for node_x, node_y in edges:
            graph.add_edge(node_x, node_y)
            f.write(cc.get(node_x) + " " + cc.get(node_y) + "\n")

    nx.write_edgelist(graph, output)
    cc.save(output + ".dictionary.pickle")

if __name__ == "__main__":
    print("Generating graph for input")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        type=str, 
        default='sample', # This is my small dataset
        help='task to generate relation from'
    )

    args = parser.parse_args()
    # Generate and Save graph 
    generate_graph(args)
    print("Done! saved under ./graph")
