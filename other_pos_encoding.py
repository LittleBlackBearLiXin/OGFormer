import torch
import random
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
import networkx as nx
from scipy.sparse.linalg import eigsh


class GraphEmbedding:
    def __init__(self, tensor_x, tensor_adjacency, embedding_method='DeepWalk', num_eigenvectors=16, walk_length=80,
                 num_walks=10, dimensions=128):

        self.tensor_x = tensor_x
        self.tensor_adjacency = tensor_adjacency
        self.embedding_method = embedding_method
        self.num_eigenvectors = num_eigenvectors
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions

    def random_walk(self, graph, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = list(graph.neighbors(current_node))
            if len(neighbors) == 0:
                break
            next_node = random.choice(neighbors)
            walk.append(next_node)
        return walk

    def deepwalk(self, graph):
        walks = []
        for node in graph.nodes():
            for _ in range(self.num_walks):
                walks.append(self.random_walk(graph, node))
        return walks

    def train_deepwalk_model(self, walks):
        model = Word2Vec(walks, vector_size=self.dimensions, window=5, min_count=1, workers=4)
        return model

    def construct_structure_matrix(self):
        adj_matrix = self.tensor_adjacency.toarray()
        N = adj_matrix.shape[0]
        D = np.diag(np.sum(adj_matrix, axis=1))  #
        L = D - adj_matrix  #

        eigenvalues, eigenvectors = eigsh(L, k=self.num_eigenvectors + 1, which='SM')  #
        U = eigenvectors[:, 1:self.num_eigenvectors + 1]  #

        return torch.tensor(U, dtype=torch.float32)

    def get_embedding(self):
        if self.embedding_method == 'DeepWalk':
            graph = nx.from_scipy_sparse_matrix(self.tensor_adjacency)
            walks = self.deepwalk(graph)
            model = self.train_deepwalk_model(walks)  #
            U = torch.tensor([model.wv[str(i)] for i in range(self.tensor_x.shape[0])], dtype=torch.float32)

        elif self.embedding_method == 'Laplacian':
            U = self.construct_structure_matrix()

        else:
            raise ValueError("Unknown embedding method. Use 'DeepWalk' or 'Laplacian'.")

        combined_matrix = torch.cat([self.tensor_x, U], dim=1)  #
        return combined_matrix



