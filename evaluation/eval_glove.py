# Eval Script for GloVe
import sys 
import numpy as np
import visualizer as V

path_to_vocab = "../GloVe/vocab.txt"
path_to_vector_file = "../GloVe/vectors.txt"

def load():
    with open(path_to_vocab, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(path_to_vector_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)

# K: Top K words that would be shown
def distance(W, vocab, ivocab, input_term, K = 20):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.Inf

    a = np.argsort(-dist)[:K]

    topk_vocab = [] 
    topk_dist = [] 
    for x in a:
        topk_vocab.append(ivocab[x])
        topk_dist.append(dist[x])
    return topk_vocab, topk_dist

if __name__ == "__main__":
    print("Evaluating results with GloVe model:")

    # Obtain queries and ground truth 
    sample_query, ground_truth = EU.obtain_ground_truth_name(20)

    # Load model 
    W, vocab, ivocab = load()

    # Get top K words and metrics from word2vec model
    results = []
    results_dist = []
    for i in len(sample_query):
        topk_words, topk_metrics = distance(W, vocab, ivocab, topn=K)
        dist = [] 
        for word in query[i]:
            dist.append(word2vec.wv.similarity(word, ground_truth[i]))
        
        results.append(topk_words)
        results_dist.append(dist)

    # Plot the vectors
    V.display_pca_scatterplot(model, sample_query[0], 300, output = "GloVe_eval.png")

    # 