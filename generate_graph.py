import argparse
import json
import pandas as pd
from tqdm import tqdm
import networkx as nx
import utils

from relational_embedder.data_prep import data_prep_utils as dpu
import textification.textify_relation as tr
from token_dict import TokenDict


def generate_graph(args):
    task = args.task
    suffix = "" if args.suffix == "" else "_" + args.suffix
    output_graph_path = "./financial_big{}.txt"
    # output_graph_path = "./graph/{}/{}{}.edgelist".format(task, task, suffix)
    # output_dictionary_path = "./graph/{}/{}{}.dict".format(task, task, suffix)
    data_strategy_path = "./data/strategies/{}.txt".format(task)
    data_config_path = "./data/data_config.txt"
    with open(data_config_path, "r") as jsonfile:
        data_config = json.load(jsonfile)[task]
    with open(data_strategy_path, "r") as jsonfile:
        strategies = json.load(jsonfile)

    fs = utils.all_data_files_in_path(data_config["location"])
    edges = set()

    current = 0
    for path in tqdm(fs):
        table_name = path.split("/")[-1]
        table_strategy = strategies[table_name]
        filename = "".join(table_name.split(".")[:-1])
        df = pd.read_csv(path, encoding='latin1', sep=',', low_memory=False)
        columns = df.columns
        df = tr.quantize(df, table_strategy)

        # Add row edges
        for cell_value, row, col in tr._read_rows_from_dataframe(df, columns, table_strategy):
            grain_strategy = table_strategy[col]["grain"]
            decoded_row = dpu.encode_cell(row, grain=grain_strategy)
            decoded_value = dpu.encode_cell(cell_value, grain=grain_strategy)
            for value in decoded_value:
                for row in decoded_row:
                    row_name = "{}_row:{}".format(filename, row)
                    edges.add((value, "/row", row_name))

        # Add col edges
        for cell_value, col in tr._read_columns_from_dataframe(df, columns, table_strategy):
            grain_strategy = table_strategy[col]["grain"]
            decoded_col = dpu.encode_cell(col, grain=grain_strategy)
            decoded_value = dpu.encode_cell(cell_value, grain=grain_strategy)
            for value in decoded_value:
                for col in decoded_col:
                    col_name = "col:" + col
                    edges.add((value, "/col", col_name))

    # Save output graph and dictionary
    graph = nx.Graph()
    cc = TokenDict()
    cnt = 0
    import pdb
    pdb.set_trace()
    with open(output_graph_path.format("train"), "w") as f1:
        with open(output_graph_path.format("test"), "w") as f2:
            with open(output_graph_path.format("validation"), "w") as f3:
                for node_x, type, node_y in iter(edges):
                    cnt = (cnt + 1) % 100
                    decoded_x, decoded_y = node_x, node_y  # cc.get(node_x), cc.get(node_y)
                    if cnt < 90:
                        f1.write("{}\t{}\t{}\n".format(decoded_x, type, decoded_y))
                    if cnt >= 90 and cnt < 95:
                        f2.write("{}\t{}\t{}\n".format(decoded_x, type, decoded_y))
                    if cnt >= 95:
                        f3.write("{}\t{}\t{}\n".format(decoded_x, type, decoded_y))
    # cc.save(output_dictionary_path)
    print("Saved under", output_graph_path)


if __name__ == "__main__":
    print("Generating graph for input")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='sample',
                        help='task to generate relation from'
                        )

    parser.add_argument('--suffix',
                        type=str,
                        default='',
                        help='a suffix to identify'
                        )

    args = parser.parse_args()
    generate_graph(args)
