# Given an embedding, visualize the embedding with PCA 

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import eval_utils as EU 
import argparse

def convert_code_to_three_categories(x):
    # This is purely to show that the method works on the sample dataset
    if "'" in x: x = x.split("'")[1]
    letters = "".join([i for i in x if i.isalpha()])
    div = "".join([i for i in x if i.isdigit()])
    if div != "": 
        div = str(int(div) % 3)
    return "".join(letters + div)

def display_pca_scatterplot(path_to_model, task, words=None):
    model = KeyedVectors.load_word2vec_format(path_to_model)
    model_name = path_to_model.split("/")[-1]
    if words is None: 
        words = list(model.vocab.keys())
    word_vectors = np.array([model[w] for w in words])

    twodim = pd.DataFrame(
        PCA().fit_transform(word_vectors)[:,:2], 
        columns = ["x", "y"]
    )
    if "word2vec" in path_to_model: 
        twodim["type"] = pd.Series(words).apply(lambda x: convert_code_to_three_categories(x))

    if "node2vec" in path_to_model:
        twodim["type"] = pd.Series(words).apply(
            lambda x: "row&col" if "row" in x or "col" in x else convert_code_to_three_categories(x)
        )

    plt.figure(figsize=(15,15))
    sns.scatterplot(data = twodim, x="x", y="y", hue="type")
    plt.savefig("./embedding_plots/" + task + "/" + model_name.split(".")[0] + ".png")


node2vec_embedding_storage = '../node2vec/emb/'
word2vec_embedding_storage = '../word2vec/emb/'
if __name__ == "__main__":
    # used embeddings from node2vec for testing purposes
    print("Visualizing results of specific tasks via PCA:")

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        type=str, 
        required=True, 
        help='task to be evaluated on'
    )

    args = parser.parse_args()
    task = args.task
    all_n2v_emb_path = EU.all_files_in_path(node2vec_embedding_storage, task)
    all_w2v_emb_path = EU.all_files_in_path(word2vec_embedding_storage, task)
    for path_to_model in all_w2v_emb_path + all_n2v_emb_path:
        print(path_to_model)
        display_pca_scatterplot(path_to_model, task)