# Given an embedding, visualize the embedding with PCA 
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

def convert_code_to_three_categories(x):
    # This is purely to show that the method works on the sample dataset
    letters = "".join([i for i in x if not i.isdigit()])
    div = "".join([i for i in x if i.isdigit()])
    if div != "": 
        div = str(int(div) % 3)
    return "".join(letters + div)

def display_pca_scatterplot(path_to_model, words=None):
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
        twodim["type"] = pd.Series(words).apply(lambda x: x.split("'")[-2])

    if "node2vec" in path_to_model:
        twodim["type"] = pd.Series(words).apply(
            lambda x: "row&col" if "row" in x or "col" in x else convert_code_to_three_categories(x)
        )

    plt.figure(figsize=(15,15))
    sns.scatterplot(data = twodim, x="x", y="y", hue="type")
    plt.savefig("./visualizer_plot/" + model_name.split(".")[0] + ".png")


if __name__ == "__main__":
    # used embeddings from node2vec for testing purposes
    path_to_model = '../node2vec/emb/sample.emb'
    display_pca_scatterplot(path_to_model)