import argparse
import run_textify as rt
from textification import textify_relation as tr
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os 
import json

def all_files_in_path(path, task):
    fs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store" and f != "base_processed.csv" and f.find(task) != -1]
    return fs

class SentenceIterator: 
    def __init__(self, filepath): 
        self.filepath = filepath 

    def __iter__(self): 
        for line in open(self.filepath): 
            yield line.split()

if __name__ == "__main__":
    def textify():
        parser.add_argument('--integer_strategy', default='skip')
        parser.add_argument('--relation_strategy', default='row_and_col')
        parser.add_argument('--output', type=str, default='textified.txt')
        parser.add_argument('--dataset', type=str, default=task_path)
        parser.add_argument('--grain', default='cell', type=str, choices=tr.TEXTIFICATION_GRAIN,
                            help='choose whether to process individual tokens (token), or entire cells (cell)')
        parser.add_argument('--output_format', default='sequence_text', choices=tr.OUTPUT_FORMAT,
                            help='sequence_text or windowed_text')
        parser.add_argument('--debug', default=False, help='whether to run program in debug mode or not')
        args = parser.parse_args()

        for relation_strategy in tr.RELATION_STRATEGY:
            for integer_strategy in ["quantize", "stringify"]:
                args.relation_strategy = relation_strategy
                args.integer_strategy = integer_strategy    
                args.output = "./textified_data/{}_textified_{}_{}.txt".format(args.task, relation_strategy, integer_strategy)    
                tr.main(args)

    def w2v(): 
        from gensim.models import Word2Vec
        paths = all_files_in_path("./textified_data/", args.task)
        for path in tqdm(paths):
            sentences = SentenceIterator(path) 
            for length in [5, 50, 100]:
                model = Word2Vec(sentences, size = length, workers = 24, window = 10, min_count = 0, sg = 0, iter = 80)
                output_name = "./word2vec/emb/" + str(length) + path.split("/")[-1].replace('txt', 'emb')
                model.wv.save_word2vec_format(output_name)

    parser = argparse.ArgumentParser()
    # Input a task name 
    parser.add_argument('--task', type=str, required=True, help='name of the task to be run')
    parser.add_argument('--run_textify', type=bool, default=False)
    parser.add_argument('--run_w2v', type=bool, default=False)
    args = parser.parse_args()

    # Just place holders
    with open("./data/data_config.txt", "r") as json_file: 
        config = json.load(json_file)
        task_path = config[args.task]["location"]

    if args.run_textify: textify()
    if args.run_w2v: w2v()
    