import argparse
import numpy as np
import pandas as pd 
from gensim.models import Word2Vec

def load_walks(path, walk_length):
    with open(path, "r") as f:
        content = f.readlines()
    content = [row.split()[:walk_length] for row in content]
    return content


def run(path, walk_length, dim, window_size, output):
    walks = load_walks(path, walk_length)
    model = Word2Vec(walks, size=dim, window=window_size,
                     min_count=0, sg=1, workers=8, iter=15)
    model.wv.save_word2vec_format(output)
    print("Model trained and saved under {}".format(output))


def main(task, suffix):
    file_name = task if suffix == "" else "{}_{}".format(task, suffix)
    path = "./walks/{}.txt".format(file_name)
    for walk_length in [40, 60, 80]:
        for dim in [5, 20, 50, 100, 200]: 
            for window_size in [5, 10]:
                s = "{},{},{}".format(walk_length, dim, window_size)
                print(s)
                output = "./emb/{}.emb{}".format(file_name, s)
                run(path, walk_length, dim, window_size, output)

main("ftp", "")
main("ncaa", "")