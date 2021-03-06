import argparse
from enum import Enum

import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import word2vec as w2v
from relational_embedder.data_prep import data_prep_utils as dpu
from relational_embedder.model_analyzer import we_analyzer as wean


class CompositionStrategy(Enum):
    AVG = 0,
    WEIGHTED_AVG_EQUALITY = 1,
    AVG_UNIQUE = 2


def column_avg_composition(df, row_we_model, col_we_model, word_hubness_row, word_hubness_col):
    column_we_based_row = dict()
    column_we_based_col = dict()
    row_hubness_th = word_hubness_row["__QUALITY_HUBNESS_THRESHOLD"]
    col_hubness_th = word_hubness_col["__QUALITY_HUBNESS_THRESHOLD"]
    columns = df.columns
    missing_words = 0
    for c in columns:
        col_wes_based_row = []
        col_wes_based_col = []
        value = df[c]
        for el in value:
            # Check validity of cell
            if not dpu.valid_cell(el):
                continue
            el = dpu.encode_cell(el)
            if word_hubness_row[el] < row_hubness_th:
                try:
                    vector_row = row_we_model.get_vector(el)
                    col_wes_based_row.append(vector_row)
                except KeyError:
                    missing_words += 1
                    continue
            if word_hubness_col[el] < col_hubness_th:
                try:
                    vector_col = col_we_model.get_vector(el)
                    col_wes_based_col.append(vector_col)
                except KeyError:
                    missing_words += 1
                    continue
        col_wes_based_row = np.asarray(col_wes_based_row)
        col_we_based_row = np.mean(col_wes_based_row, axis=0)
        col_wes_based_col = np.asarray(col_wes_based_col)
        col_we_based_col = np.mean(col_wes_based_col, axis=0)
        # Store column only if not nan
        if not np.isnan(col_we_based_row).any():
            column_we_based_row[c] = col_we_based_row
        if not np.isnan(col_we_based_col).any():
            column_we_based_col[c] = col_we_based_col
    return column_we_based_row, column_we_based_col, missing_words


def column_avg_composition_row_only(df, row_we_model, word_hubness_row):
    """
    ONLY ROW - for convenience
    :param df:
    :param row_we_model:
    :return:
    """
    column_we_based_row = dict()
    row_hubness_th = word_hubness_row["__QUALITY_HUBNESS_THRESHOLD"]
    columns = df.columns
    missing_words = 0
    for c in columns:
        col_wes_based_row = []
        value = df[c]
        for el in value:
            # Check validity of cell
            if not dpu.valid_cell(el):
                continue
            el = dpu.encode_cell(el)
            if word_hubness_row[el] < row_hubness_th:
                try:
                    vector_row = row_we_model.get_vector(el)
                    col_wes_based_row.append(vector_row)
                except KeyError:
                    missing_words += 1
                    continue
        col_wes_based_row = np.asarray(col_wes_based_row)
        col_we_based_row = np.mean(col_wes_based_row, axis=0)
        # Store column only if not nan
        if not np.isnan(col_we_based_row).any():
            column_we_based_row[c] = col_we_based_row
    return column_we_based_row, missing_words


def column_avg_unique_composition(df, we_model):
    column_we = dict()
    columns = df.columns
    missing_words = 0
    for c in columns:
        col_wes = []
        value = df[c].unique()
        for el in value:
            # Check validity of cell
            if not dpu.valid_cell(el):
                continue
            el = dpu.encode_cell(el)
            try:
                vector = we_model.get_vector(el)
            except KeyError:
                missing_words += 1
                continue
            col_wes.append(vector)
        col_wes = np.asarray(col_wes)
        col_we = np.mean(col_wes, axis=0)
        column_we[c] = col_we
    return column_we, missing_words


def column_weighted_avg_equality_composition(df, we_model):
    # TODO
    return


def relation_column_avg_composition(column_we):
    relation_we = np.mean(np.asarray(
        [v for v in column_we.values() if not np.isnan(v).any()]
    ), axis=0)
    return relation_we


def relation_column_weighted_avg_equality_composition(column_we):
    # TODO
    return


def row_avg_composition(df, we_model, word_hubness_row):
    missing_words = 0
    emb_length = len(we_model.vectors[0])
    row_hubness_th = word_hubness_row["__QUALITY_HUBNESS_THRESHOLD"]
    row_we_dict = dict()
    columns = df.columns
    for i, row in df.iterrows():
        row_wes = []
        for c in columns:
            # Check validity of cell
            if not dpu.valid_cell(row[c]):
                continue
            el = dpu.encode_cell(row[c])
            if word_hubness_row[el] < row_hubness_th:  # filter out hubs
                try:
                    we = we_model.get_vector(el)
                    row_wes.append(we)
                except KeyError:
                    missing_words += 1
                    continue
        row_wes = np.asarray(row_wes)
        row_we = np.mean(row_wes, axis=0)
        if not np.isnan(row_we).any():
            row_we_dict[i] = row_we
        else:  # we need to fill in the index regardless
            row_we_dict[i] = np.asarray([np.NAN] * emb_length)
    return row_we_dict, missing_words


def row_weighted_avg_equality_composition(df, we_model):
    # TODO
    return


def compose_dataset_avg(path_to_relations, row_we_model, col_we_model, word_hubness_row, word_hubness_col):
    row_relational_embedding = dict()
    col_relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in tqdm(all_relations):
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we_based_row, col_we_based_col, missing_words = column_avg_composition(df, row_we_model, col_we_model, word_hubness_row, word_hubness_col)
        rel_we_based_row = relation_column_avg_composition(col_we_based_row)
        rel_we_based_col = relation_column_avg_composition(col_we_based_col)
        row_we, missing_words = row_avg_composition(df, row_we_model, word_hubness_row)
        row_relational_embedding[relation] = dict()
        row_relational_embedding[relation]["vector"] = rel_we_based_row
        row_relational_embedding[relation]["columns"] = col_we_based_row
        row_relational_embedding[relation]["rows"] = row_we

        col_relational_embedding[relation] = dict()
        col_relational_embedding[relation]["vector"] = rel_we_based_col
        col_relational_embedding[relation]["columns"] = col_we_based_col
    return row_relational_embedding, col_relational_embedding, word_hubness_row, word_hubness_col


def compose_dataset_avg_row_only(path_to_relations, row_we_model, word_hubness_row):
    """
    ONLY ROW - for convenience
    :param path_to_relations:
    :param row_we_model:
    :return:
    """
    row_relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in tqdm(all_relations):
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we_based_row, missing_words = column_avg_composition_row_only(df, row_we_model, word_hubness_row)
        rel_we_based_row = relation_column_avg_composition(col_we_based_row)
        row_we, missing_words = row_avg_composition(df, row_we_model, word_hubness_row)
        row_relational_embedding[relation] = dict()
        row_relational_embedding[relation]["vector"] = rel_we_based_row
        row_relational_embedding[relation]["rows"] = row_we
    return row_relational_embedding, word_hubness_row


def compose_dataset_avg_unique(path_to_relations, we_model):
    relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in all_relations:
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we, missing_words = column_avg_unique_composition(df, we_model)
        rel_we = relation_column_avg_composition(col_we)
        row_we, missing_words = row_avg_composition(df, we_model)
        relational_embedding[relation] = dict()
        relational_embedding[relation]["vector"] = rel_we
        relational_embedding[relation]["columns"] = col_we
        relational_embedding[relation]["rows"] = row_we
    return relational_embedding


def compose_dataset_weighted_avg_equality(path_to_relations, we_model):
    relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in all_relations:
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we, missing_words = column_weighted_avg_equality_composition(df, we_model)
        rel_we = relation_column_weighted_avg_equality_composition(col_we)
        row_we, missing_words = row_weighted_avg_equality_composition(df, we_model)
        relational_embedding[relation] = dict()
        relational_embedding[relation]["vector"] = rel_we
        relational_embedding[relation]["columns"] = col_we
        relational_embedding[relation]["rows"] = row_we
    return relational_embedding


def compose_dataset(path_to_relations, row_we_model, col_we_model, strategy=CompositionStrategy.AVG,
                    word_hubness_row=None, word_hubness_col=None):
    """
    Given a repository of relations compose column, row and relation embeddings and store it hierarchically
    :param path_to_relations:
    :param we_model:
    :return:
    """
    # TODO: because of the performance impact of computing hubness here (and especially with the method below)
    # TODO: this is now computed independently and fed directly into this composition. Keeping code for ref.
    # # compute hubness of embeddings
    # if word_hubness_row is None:
    #     print("Computing row hubness...")
    #     word_hubness_row = wean.compute_hubness_parallel_sample(row_we_model)
    #     print("Computing row hubness...OK")
    # if word_hubness_col is None:
    #     print("Computing col hubness...")
    #     word_hubness_col = wean.compute_hubness_parallel_sample(col_we_model)
    #     print("Computing row hubness...OK")

    if strategy == CompositionStrategy.AVG:
        print("Composing using AVG")
        return compose_dataset_avg(path_to_relations, row_we_model, col_we_model, word_hubness_row, word_hubness_col)
    elif strategy == CompositionStrategy.WEIGHTED_AVG_EQUALITY:
        print("Composing using WEIGHTED_AVG_EQUALITY")
        return compose_dataset_weighted_avg_equality(path_to_relations, row_we_model, col_we_model)
    elif strategy == CompositionStrategy.AVG_UNIQUE:
        print("Composing using AVG_UNIQUE")
        return compose_dataset_avg_unique(path_to_relations, row_we_model, col_we_model)


def compose_dataset_row_only(path_to_relations, row_we_model, strategy=CompositionStrategy.AVG):
    """
    ONLY ROW - for convenience
    :param path_to_relations:
    :param row_we_model:
    :param strategy:
    :return:
    """
    # compute hubness of embeddings
    word_hubness_row = wean.compute_hubness(row_we_model)
    if strategy == CompositionStrategy.AVG:
        print("Composing using AVG")
        return compose_dataset_avg_row_only(path_to_relations, row_we_model, word_hubness_row)
    elif strategy == CompositionStrategy.WEIGHTED_AVG_EQUALITY:
        print("Composing using WEIGHTED_AVG_EQUALITY")
        print("To implement...")
        exit()
    elif strategy == CompositionStrategy.AVG_UNIQUE:
        print("Composing using AVG_UNIQUE")
        print("To implement...")
        exit()


if __name__ == "__main__":
    print("Composition")

    import pickle

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_model', help='path to we model')
    parser.add_argument('--method', default='avg', help='composition method')
    parser.add_argument('--dataset', help='path to csv files')
    parser.add_argument('--output', default='textified.pkl', help='place to output relational embedding')

    args = parser.parse_args()

    we_model = w2v.load(args.we_model)
    relational_embedding = compose_dataset(args.dataset, we_model)
    with open(args.output, 'wb') as f:
        pickle.dump(relational_embedding, f)
    print("Relational Embedding serialized to: " + str(args.output))
