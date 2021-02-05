import numpy as np
import networkx as nx
import random
import time
from tqdm import tqdm

class Graph():
	def __init__(self, nx_G, p, q):
		self.G = nx_G
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		walk = [start_node]
		for i in range(walk_length):
			walk.append(random.choice(self.adjList[walk[i]]))
		return list(map(lambda x: str(x), walk))

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		self.adjList = [list(G.neighbors(x)) for x in G.nodes()]
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in tqdm(nodes):
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
		return walks

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''

		return