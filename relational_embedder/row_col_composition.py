import argparse
import os
import pickle
from tqdm import tqdm
from enum import Enum
import pandas as pd
import numpy as np

import word2vec as w2v
from textification import textify_relation as tx
from relational_embedder.data_prep import data_prep_utils as dpu


class CompositionStrategy(Enum):
    AVG = 0,  # vector composition is performed using an average
    WEIGHTED_AVG_EQUALITY = 1,  # not implemented
    AVG_UNIQUE = 2


def column_avg_composition(df, row_we_model, col_we_model, integer_strategy, grain):
    column_we_based_row = dict()
    column_we_based_col = dict()
    missing_words = 0

    map_column_values = tx.read_column_values(None, integer_strategy, grain, dataframe=df)

    for c, values in map_column_values.items():
        col_wes_based_row = []
        col_wes_based_col = []
        for el in values:
            # Check validity of cell
            if not dpu.valid_cell(el):
                continue
            try:
                vector_row = row_we_model.get_vector(el)
                col_wes_based_row.append(vector_row)
            except KeyError:
                missing_words += 1
                continue
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


def relation_column_avg_composition(column_we):
    relation_we = np.mean(np.asarray(
        [v for v in column_we.values() if not np.isnan(v).any()]
    ), axis=0)
    return relation_we


def row_avg_composition(df, we_model, integer_strategy, grain):
    missing_words = 0
    row_we_dict = dict()
    emb_length = len(we_model.vectors[0])
    map_rows_values = tx.read_rows_values(None, integer_strategy, grain, dataframe=df)

    for index, values in map_rows_values.items():
        row_wes = []
        for v in values:
            try:
                we = we_model.get_vector(v)
                row_wes.append(we)
            except KeyError:
                missing_words += 1
                continue
        row_wes = np.asarray(row_wes)
        row_we = np.mean(row_wes, axis=0)
        if not np.isnan(row_we).any():
            row_we_dict[index] = row_we
        else:  # we need to fill in the index regardless FIXME: why would there be a NaN vector though?
            row_we_dict[index] = np.asarray([np.NAN] * emb_length)
    return row_we_dict, missing_words


def compose_dataset_avg(path_to_relations, row_we_model, col_we_model, args):
    row_relational_embedding = dict()
    col_relational_embedding = dict()
    all_relations = [relation for relation in os.listdir(path_to_relations)]
    for relation in tqdm(all_relations):
        path = path_to_relations + "/" + relation
        df = pd.read_csv(path, encoding='latin1')
        if not dpu.valid_relation(df):
            continue
        col_we_based_row, col_we_based_col, missing_words = column_avg_composition(df,
                                                                                   row_we_model,
                                                                                   col_we_model,
                                                                                   args.integer_strategy,
                                                                                   args.grain)
        rel_we_based_row = relation_column_avg_composition(col_we_based_row)
        rel_we_based_col = relation_column_avg_composition(col_we_based_col)
        row_we, missing_words = row_avg_composition(df, row_we_model, args.integer_strategy, args.grain)

        row_relational_embedding[relation] = dict()
        row_relational_embedding[relation]["vector"] = rel_we_based_row
        row_relational_embedding[relation]["columns"] = col_we_based_row
        row_relational_embedding[relation]["rows"] = row_we

        col_relational_embedding[relation] = dict()
        col_relational_embedding[relation]["vector"] = rel_we_based_col
        col_relational_embedding[relation]["columns"] = col_we_based_col
    return row_relational_embedding, col_relational_embedding


def compose_dataset(path_to_relations, row_we_model, col_we_model, args, strategy=CompositionStrategy.AVG):
    """
    Given a repository of relations compose column, row and relation embeddings and store it hierarchically
    :param path_to_relations:
    :param we_model:
    :return:
    """
    if strategy == CompositionStrategy.AVG:
        print("Composing using AVG")
        return compose_dataset_avg(path_to_relations, row_we_model, col_we_model, args)
    elif strategy == CompositionStrategy.WEIGHTED_AVG_EQUALITY:
        print("Composing using WEIGHTED_AVG_EQUALITY")
        # TODO: removed as part of simplification
    elif strategy == CompositionStrategy.AVG_UNIQUE:
        print("Composing using AVG_UNIQUE")
        # TODO: removed as simplification


if __name__ == "__main__":
    print("Row-n-Col Composition")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--row_we_model', help='path to row we model')
    parser.add_argument('--col_we_model', help='path to col we model')
    parser.add_argument('--method', default='avg', help='composition method')
    parser.add_argument('--dataset', help='path to csv files')
    parser.add_argument('--integer_strategy', default='skip', type=str, choices=tx.INTEGER_STRATEGY,
                        help='strategy to determine how to deal with integers')
    parser.add_argument('--grain', default='cell', type=str, choices=tx.TEXTIFICATION_GRAIN,
                        help='choose whether to process individual tokens (token), or entire cells (cell)')
    parser.add_argument('--output', default='re.pkl', help='place to output relational embedding')
    # parser.add_argument('--row_hubness_path', default=None, help='path to row_hubness computed')
    # parser.add_argument('--col_hubness_path', default=None, help='path to col_hubness computed')

    args = parser.parse_args()

    row_we_model = w2v.load(args.row_we_model)
    col_we_model = w2v.load(args.col_we_model)

    method = None
    if args.method == "avg":
        method = CompositionStrategy.AVG
    elif args.method == "avg_unique":
        method = CompositionStrategy.AVG_UNIQUE

    row_relational_embedding, col_relational_embedding, = compose_dataset(args.dataset, row_we_model, col_we_model,
                                                                          args, strategy=method)

    # Store output
    with open(args.output + "/row_re.pkl", 'wb') as f:
        pickle.dump(row_relational_embedding, f)
    with open(args.output + "/col_re.pkl", 'wb') as f:
        pickle.dump(col_relational_embedding, f)
    print("Relational Embedding serialized to: " + str(args.output))
