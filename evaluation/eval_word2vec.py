from gensim import word2vec
import eval_utils as EU

K = 20 
word2vec_model_path = '../word2vec/word2vec.bin'

def word2vec_get_similar(query):
    return None, None

if __name__ == "__main__":
    print("Evaluating results with word2vec model:")

    # Obtain queries and ground truth 
    sample_query, ground_truth = EU.obtain_ground_truth_name(20)

    # Load model 
    model = word2vec.load(word2vec_model_path, encoding = "ISO-8859-1")

    # Get top K words and metrics from word2vec model 
    results = []
    results_dist = []
    for i in len(sample_query):
        topk_words, topk_metrics = word2vec.predict_output_word(query[i], topn=K)
        dist = [] 
        for word in query[i]:
            dist.append(word2vec.wv.similarity(word, ground_truth[i]))
        
        results.append(topk_words)
        results_dist.append(dist)

    # measure quality 
    precision = EU.measure_quality(ground_truth, results)
    print(precision)

    # Save evaluation to file

    