import numpy as np
import networkx as nx
import random
import time
from numpy.random import choice 
from tqdm import tqdm
from collections import defaultdict

class Graph():
	def __init__(self, nx_G, p, q, is_weighted, limit = 1000):
		self.G = nx_G
		self.p = p
		self.q = q
		self.weighted = is_weighted
		self.adjList = [list(nx_G.neighbors(x)) for x in nx_G.nodes()]
		self.adjList_prob = [[nx_G[y][x]['weight'] for y in nx_G.neighbors(x)] for x in nx_G.nodes()]
		self.adjList_prob = [[float(i) / sum(prob_vector) for i in prob_vector] for prob_vector in self.adjList_prob]

		self.limit_dict = defaultdict(int)
		self.limit = limit 

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		walk = [start_node]
		curr = walk[0]
		for i in range(walk_length):
			if self.adjList[curr] == []: break
			if self.weighted: 
				nxt = choice(self.adjList[curr], p = self.adjList_prob[curr])
			else: 
				nxt = choice(self.adjList[curr])
			self.limit_dict[nxt] += 1 
			if self.limit_dict[nxt] >= self.limit: 
				pass
				# idx = self.adjList[curr].index(nxt)
				# self.adjList[curr].pop(idx)
				# self.adjList_prob[curr].pop(idx)
				# norm_sum = sum(self.adjList_prob[curr])
				# self.adjList_prob[curr] = [float(i) / norm_sum for i in self.adjList_prob[curr]]
			else:
				walk.append(nxt)
			curr = nxt 
		return list(map(lambda x: str(x), walk))

	def simulate_walks(self, num_walks, walk_length, nodes = None):
		'''
		Repeatedly simulate random walks from each node.
		'''
		walks = []
		if nodes is None:
			nodes = list(self.G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in tqdm(nodes):
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=int(node)))
		return walks

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''

		return