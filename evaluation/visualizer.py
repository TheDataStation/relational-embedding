# Given an embedding, visualize the embedding with PCA 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.switch_backend('agg')
import word2vec
from sklearn.decomposition import PCA

def display_pca_scatterplot(model, words=None, sample=0, output = "display.png", vocab_list = model.vocab):
    colors = ['r'] * len(words) + ['b'] * sample
    if sample > 0:
        words_original = words
        words = np.concatenate((np.array(words), np.random.choice(list(vocab_list), sample)), axis=None)
        
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]

    print("Evaluation visualizer: Ready to plot")
    plt.figure(figsize=(10, 10))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c=colors)
    for word, (x,y) in zip(words, twodim):
        if (word in words_original):
            plt.text(x+0.05, y+0.05, word)
    plt.savefig(output)


if __name__ == "__main__":
    # used embeddings from word2vec for testing purposes
    path_to_model = "../word2vec/word2vec_trial_2.bin"
    model = word2vec.load(path_to_model, encoding = "ISO-8859-1")

    # Display with PCA on two dimensions
    cells_array = ["('tt0072308'_'titleid')", "('nm0000001'_'nconst')", "('tt0053137'_'titleid')", "('tt0043044'_'titleid')", "('tt0050419'_'titleid')"]
    display_pca_scatterplot(model, cells_array, 300, output = "dummy_name.png", vocab_list = model.vocab)