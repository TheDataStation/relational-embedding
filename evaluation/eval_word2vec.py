from gensim.models import Word2Vec
import sys
sys.path.append('..')

import pandas as pd
import eval_utils as EU
import visualizer as VS
import word2vec
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

K = 20 
word2vec_model_path_kraken = "../word2vec/kraken.bin"
kraken_path = "../data/kraken/"
word2vec_model_path_school = "../word2vec/school.bin"
test_size = 0.2
num_bins = 20

def classification_task(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators = 1000)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    pscore = accuracy_score(y_test, y_pred)
    print("RF Test score:", pscore)

def regression_task(): 
    return None 

def kraken_task():
    # Load model 
    model = word2vec.load(word2vec_model_path_kraken)

    # Obtain textified & quantized data
    df = pd.read_csv(os.path.join(kraken_path, "kraken.csv"))
    df_textified = EU.textify_df(df)
    x_vec, y_vec = EU.vectorize_df(df_textified, model, 4)
    Y = df['result'].values.ravel()

    # Train a Random Forest classifier
    X_train, X_test, y_train, y_test = train_test_split(x_vec, Y, test_size = test_size, random_state=10)
    classification_task(X_train, X_test, y_train, y_test)


def school_task():
    # Load model 
    print("Obtain model")
    model = word2vec.load(word2vec_model_path_school)

    # Obtain textified & quantized data
    df = pd.read_csv("../data/school/base.csv")
    df_textified = EU.textify_df(df)
    x_vec, y_vec = EU.vectorize_df(df_textified, model, 7)
    Y = df['class'].values.ravel()

    # Train a Random Forest classifier
    print("Training")
    X_train, X_test, y_train, y_test = train_test_split(x_vec, Y, test_size = test_size, random_state=10)
    classification_task(X_train, X_test, y_train, y_test)
 

if __name__ == "__main__":
    print("Evaluating results with word2vec model:")
    # print("Kraken Dataset:")
    # kraken_task()


    print("School dataset")
    school_task()


    # # Get top K words and metrics from word2vec model 
    # results = []
    # results_dist = []
    # for i in len(sample_query):
    #     topk_words, topk_metrics = word2vec.predict_output_word(query[i], topn=K)
    #     dist = [] 
    #     for word in query[i]:
    #         dist.append(word2vec.wv.similarity(word, ground_truth[i]))
        
    #     results.append(topk_words)
    #     results_dist.append(dist)

    # # measure quality 
    # precision = EU.measure_quality(ground_truth, results)

    # # Plot Similarity 
    # VS.display_pca_scatterplot(model, sample_query[0], 300, output = "word2vec_eval.png", vocab_list = model.vocab)

    