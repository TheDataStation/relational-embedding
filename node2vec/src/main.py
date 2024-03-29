'''
Reference implementation of node2vec.
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import pandas as pd 
from collections import defaultdict
import node2vec
from gensim.models import Word2Vec
from random import random 


def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--task',
		type=str,
		default="sample",
		required=True,
		help="task to generate embedding from"
    )

	parser.add_argument('--suffix', nargs='?', default='',
	                    help='A suffix of the experiment')

	parser.add_argument('--input', nargs='?', default='',
	                    help='Graph path')

	parser.add_argument('--output', nargs='?', default='',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=50,
	                    help='Number of dimensions. Default is 50.')

	parser.add_argument('--walk-length', type=int, default=40,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=5,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=16,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	args = parser.parse_args()

	file_name = args.task if args.suffix == "" else "{}_{}".format(args.task, args.suffix)
	args.input = "../graph/{}/{}.edgelist".format(args.task, file_name)
	args.output = "./emb/{}.emb".format(file_name)
	return args

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	print("starting")
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph(), delimiter=' ', comments = "?")
	else:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph(), delimiter=' ', comments = "?")
		# G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph(), delimiter=' ', comments = "?")
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1
	G = G.to_undirected()
	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	print("Model trained and saved under {}".format(args.output))
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	print("Reading Done!")
	G = node2vec.Graph(nx_G, args.p, args.q, args.weighted)
	print("Creation Done!")
	G.preprocess_transition_probs()
	print("Preprocess Done!")
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	print("Walking Done!")
	current, peak = tracemalloc.get_traced_memory()
	print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
 
	file_name = args.task if args.suffix == "" else "{}_{}".format(args.task, args.suffix)
	walks_save_path = "walks/{}.txt".format(file_name)
	with open(walks_save_path, 'w') as f:
		for walk in walks: 
			f.writelines("%s " % place for place in walk)
			f.writelines("\n")
	learn_embeddings(walks)

	# cnts = pd.DataFrame(walks).stack().value_counts()
	# restart_lst = list(cnts[cnts < cnts.quantile(0.25)].index)
	# additional_walks = max(int(args.num_walks * 0.1), 4)
	# print("additional walks", additional_walks)
	# restart_walks = G.simulate_walks(additional_walks * 4, args.walk_length, nodes=restart_lst)
	# args.output = args.output[:-4] + "_restart.emb"

	# walks_restart_save_path = "walks/{}_restart.txt".format(file_name)
	# new_walks = restart_walks + walks[:-additional_walks * cnts.shape[0]]
	# with open(walks_restart_save_path, 'w') as f:
	# 	for walk in new_walks: 
	# 		f.writelines("%s " % place for place in walk)
	# 		f.writelines("\n")
	# learn_embeddings(new_walks)

if __name__ == "__main__":
    import tracemalloc
    tracemalloc.start()
    args = parse_args()
    main(args)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()