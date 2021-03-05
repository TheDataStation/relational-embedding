import sys
sys.path.append('..')
import csv
import pandas as pd
import numpy as np
from relational_embedder.data_prep import data_prep_utils as dpu
from textification import textify_relation as tr

from os.path import isfile, join
from os import listdir
import json
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

with open("../embedding_config.json", "r") as jsonfile:
    embeddding_config = json.load(jsonfile)
num_bins = embeddding_config["num_bins"]


def all_files_in_path(path, task):
    fs = [
        join(path, f) for f in listdir(path) if isfile(join(path, f))
        and f != ".DS_Store" and f.find(task) != -1 and f[-4:] == ".emb"
    ]
    return fs


def get_PCA_for_embedding(model, ndim=2):
    from sklearn.decomposition import PCA
    words = list(model.vocab.keys())
    word_vectors = np.array([model[w] for w in words])

    twodim = pd.DataFrame(PCA().fit_transform(word_vectors)[:, :ndim],
                          columns=["d" + str(i + 1) for i in range(ndim)])
    twodim.index = words
    return twodim.T


def vectorize_df(df,
                 model,
                 model_vocab=None,
                 model_dict=None,
                 model_type="word2vec"):
    length = len(df)
    x_vectorized = [[] for i in range(length)]
    if model_type == "node2vec" or model_type == "ProNE":
        from token_dict import TokenDict
        cc = TokenDict()
        cc.load(model_dict)
        for i in range(length):
            row = df[i]
            for j in range(len(row)):
                if cc.check(row[j]) in model_vocab:
                    x_vectorized[i] += list(model[cc.check(row[j])])
    return pd.DataFrame(x_vectorized)


def textify_df(df, strategies, path):
    table_name = path.split("/")[-1]
    df = tr.quantize(df, strategies[table_name])
    columns = df.columns
    input = [[] for i in range(df.shape[0])]

    for cell_value, row, col in tr._read_rows_from_dataframe(
            df, columns, strategies[table_name]):
        grain_strategy = strategies[table_name][col]["grain"]
        decoded_row = dpu.encode_cell(row, grain=grain_strategy)
        decoded_value = dpu.encode_cell(cell_value, grain=grain_strategy)
        for value in decoded_value:
            input[row].append(value)

    # filename = "".join(table_name.split(".")[:-1])
    # for row in range(df.shape[0]):
    #     row_name = "{}_row:{}".format(filename, str(row))
    #     input[row].append(row_name)
    return input


# def measure_quality(ground_truth, predicted_truth):
#     precision = []
#     for i in range(len(ground_truth)):
#         flag = ground_truth[i] in predicted_truth[i]
#         precision.append(flag)
#     return precision

# def remove_hubness_and_run(X, y, n_neighbors=15):
#     from skhubness import Hubness
#     from sklearn.model_selection import cross_val_score
#     from skhubness.neighbors import KNeighborsClassifier

#     # Measure Hubness before and after removal (method mutual proximity)
#     hub = Hubness(k=10, metric='cosine')
#     hub.fit(X)
#     k_skew = hub.score()
#     print(f'Skewness = {k_skew:.3f}')

#     hub_mp = Hubness(k=10, metric='cosine',
#                      hubness='mutual_proximity')
#     hub_mp.fit(X)
#     k_skew_mp = hub_mp.score()
#     print(f'Skewness after MP: {k_skew_mp:.3f} '
#           f'(reduction of {k_skew - k_skew_mp:.3f})')
#     print(f'Robin hood: {hub_mp.robinhood_index:.3f} '
#           f'(reduction of {hub.robinhood_index - hub_mp.robinhood_index:.3f})')

#     # Measure Classfication Accuracy before and after removal
#     # vanilla kNN
#     knn_standard = KNeighborsClassifier(
#         n_neighbors=n_neighbors, metric='cosine')
#     acc_standard = cross_val_score(knn_standard, X, y, cv=10)

#     # kNN with hubness reduction (mutual proximity)
#     knn_mp = KNeighborsClassifier(n_neighbors=n_neighbors,
#                                   metric='cosine',
#                                   hubness='mutual_proximity')
#     acc_mp = cross_val_score(knn_mp, X, y, cv=10)

#     print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')
#     print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')
#     return (acc_standard.mean(), acc_mp.mean())


def parse_strategy(s):
    integer_strategy = "quantize"
    textification_strategy = "row_and_col"

    if s.find("alex") != -1:
        textification_strategy = "alex"
    elif s.find("row_and_col") != -1:
        textification_strategy = "row_and_col"
    elif s.find("row") != -1:
        textification_strategy = "row"
    elif s.find("col") != -1:
        textification_strategy = "col"

    if s.find("quantize") != -1:
        integer_strategy = "quantize"
    elif s.find("stringify") != -1:
        integer_strategy = "stringify"

    return textification_strategy, integer_strategy


###################################################################
#   More evaluation modules
#
###################################################################


def show_stats(model, X_train, X_test, y_train, y_test, argmax=False):
    X_pred_train = model.predict(X_train)
    X_pred_test = model.predict(X_test)
    if argmax == True:
        X_pred_train = np.argmax(X_pred_train, axis=1)
        X_pred_test = np.argmax(X_pred_test, axis=1)
    pscore_train = accuracy_score(y_train, X_pred_train)
    pscore_test = accuracy_score(y_test, X_pred_test)
    print("Train accuracy {}, Test accuracy {}".format(pscore_train,
                                                       pscore_test))
    return pscore_train, pscore_test


def classification_task_rf(X_train, X_test, y_train, y_test, n_estimators=100):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return show_stats(model, X_train, X_test, y_train, y_test)


def classification_task_logr(X_train,
                             X_test,
                             y_train,
                             y_test,
                             n_estimators=100):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    model = LogisticRegression(penalty='l2', solver='liblinear')
    model.fit(X_train, y_train)
    return show_stats(model, X_train, X_test, y_train, y_test)


def plot_tf_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_tf_history_rg(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def classification_task_nn(X_train, X_test, y_train, y_test, history=None):
    import tensorflow as tf
    input_size = X_train.shape[1]
    ncategories = np.max(y_train) + 1
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size, )),
        # tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(64, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(ncategories, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        epochs=500,
                        verbose=0,
                        validation_data=(X_test, y_test))
    plot_tf_history(history)
    model.evaluate(X_test, y_test, verbose=2)
    return show_stats(model, X_train, X_test, y_train, y_test, argmax=True)


def regression_task_nn(X_train, X_test, y_train, y_test, history=None):
    import tensorflow as tf
    # from tensorflow.keras.layers.experimental import preprocessing
    # normalizer = preprocessing.Normalization()

    # normalized = normalizer.adapt(np.array(X_train))
    model = tf.keras.Sequential([
    #   normalized,
      tf.keras.layers.Dense(64, activation='relu', input_dim = X_train.shape[1]),
      tf.keras.layers.Dense(64, activation='relu'), 
      tf.keras.layers.Dense(1)
  ])

    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_absolute_error'])

    history = model.fit(X_train,
                        y_train,
                        epochs=500,
                        verbose=0,
                        validation_data=(X_test, y_test))
    plot_tf_history_rg(history)
    train_loss = history.history['loss'][-1]
    test_loss = model.evaluate(X_test, y_test, verbose=2)
    return train_loss, test_loss

def randomForestRegression(X_train, X_test, y_train, y_test, history=None):
    
    rfr = Pipeline([
        #("normalizer", Normalizer()),
        ("rfr", RandomForestRegressor(n_estimators=100, random_state=7))
        ])
    parameters = {
        'rfr__max_depth': [2,5,10,20],
        'rfr__min_samples_split': [2,5],
        'rfr__max_leaf_nodes': [5,10,20],
        'rfr__min_samples_leaf': [2,5],
        #'rfr__max_samples': [0.2,0.5,1],
    }
    greg = GridSearchCV(estimator = rfr, param_grid=parameters, cv=5, verbose=3)
    greg.fit(X_train, y_train)
    best_param = greg.best_params_
    best_score = greg.best_score_
    best_estim = greg.best_estimator_

    print(best_param)
    print(best_score)
    print(best_estim)
    print(X_train.shape)

def lassoRegression(X_train, X_test, y_train, y_test, history=None):
    from sklearn.linear_model import Lasso, LinearRegression
    lasso = Pipeline([
    ("lasso", Lasso(normalize = True, random_state=7))
    ])
    parameters = {
        'lasso__alpha': [0.0001, 0.001,0.01, 0.1, 0.5, 0.7],
    }
    greg = GridSearchCV(estimator = lasso, param_grid=parameters, cv=5, verbose=3)
    greg.fit(X_train, y_train)
    best_param = greg.best_params_
    best_score = greg.best_score_
    best_estim = greg.best_estimator_

    print(best_param)
    print(best_score)
    print(best_estim)
    print(X_train.shape)

def elasticNetRegression(X_train, X_test, y_train, y_test, history=None):
    from sklearn.linear_model import ElasticNet
    en = Pipeline([
    ("en", ElasticNet(normalize = True, random_state=7))
    ])
    parameters = {
        'en__alpha': [0.0001, 0.001,0.01, 0.1, 0.5, 1],
        'en__l1_ratio':[0.2,0.5,0.8]
    }
    greg = GridSearchCV(estimator = en, param_grid=parameters, cv=5, verbose=3)
    greg.fit(X_train, y_train)
    best_param = greg.best_params_
    best_score = greg.best_score_
    best_estim = greg.best_estimator_

    print(best_param)
    print(best_score)
    print(best_estim)
    print(X_train.shape)

