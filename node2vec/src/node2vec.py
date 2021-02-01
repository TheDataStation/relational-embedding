import numpy as np
import networkx as nx
import random
import time

class Graph():
	def __init__(self, nx_G, p, q):
		self.G = nx_G
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G

		walk = [start_node]
		for i in range(walk_length):
			cur_nbrs = [n for n in self.adjM[walk[i]]]
			walk.append(random.choice(cur_nbrs))
		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		self.adjM = [G.neighbors(x) for x in G.nodes()]
		import pdb; pdb.set_trace()
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
		return walks

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''

		return