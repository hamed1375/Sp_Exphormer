import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from bottleneck import argpartition

def sampler_uniform(adj_eidx, P, P_inv, k):
    deg = adj_eidx.shape[0]
    if k > deg:
        extra = np.random.choice(adj_eidx, size=k - deg, replace=True)
        return np.concatenate([adj_eidx, extra])
    idx = np.random.choice(adj_eidx, size=k, replace=False)
    return idx

def sampler_reservoir(adj_eidx, P, P_inv, k):
    deg = adj_eidx.shape[0]
    if k > deg:
        extra = np.random.choice(adj_eidx, size=k - deg, replace=True, p=P)
        return np.concatenate([adj_eidx, extra])
    rsv = -np.log(np.random.rand(deg)) * P_inv
    idx = argpartition(rsv, k-1)[:k]
    return adj_eidx[idx]

# The maximum can be cached for each node to not repeat this process. Caching is not implemented here.
def sampler_get_max(adj_eidx, P, P_inv, k):
    deg = adj_eidx.shape[0]
    if k > deg:
        extra = np.random.choice(adj_eidx, size=k - deg, replace=True, p=P)
        return np.concatenate([adj_eidx, extra])
    idx = argpartition(P, P.shape[0]-k)[P.shape[0]-k:]
    return adj_eidx[idx]


class NeighborSampler():
    def __init__(self, original_graph, edge_index, edge_attr, P, adj_eidx, deg, num_layers, sampler='graph_reservoir') -> None:
        """
        Initialize the NeighborSampler.
        Parameters:
        original_graph (torch_geometric.data.Data): The original graph containing n nodes and m edges.
        edge_index (torch.Tensor): Tensor of shape (2, m) representing the edge indices.
        edge_attr (torch.Tensor): Tensor of shape (m, 1) representing the edge attributes.
        P (list): List of length n, where each element is a 2D numpy array of size (num_layers, d_i). 
                  P[i][l][j] represents the attention score of the j-th neighbor of node i in layer l.
        adj_eidx (list): List of length n, where adj_eidx[i][j] is the index of the edge between node i 
                         and its j-th neighbor in the edge_index.
        deg (list): List representing the number of neighbors to sample per layer.
        num_layers (int): Number of layers in the model.
        sampler (str): Sampling strategy. It consists of two parts: the first part indicates whether 
                       sampling happens on the graph first and then batching is done, and the second 
                       part indicates the strategy of sampling. Default is 'graph_reservoir'.
        Returns:
        None
        """

        self.data = original_graph
        self.edge_index, self.edge_attr, self.P, self.adj_eidx, self.deg, self.num_layers = \
            edge_index, edge_attr, P, adj_eidx, deg, num_layers
        
        self.sampler = sampler
        self.P = tuple(self.P)
        # 1/P values to save some time in calculations
        self.P_inv = tuple([1.0/(p+1e-8) for p in self.P])

        if self.sampler in ['graph_reservoir', 'graph_uniform']:
            self.sample_on_batching = False
            self.sampled_edge_ids = []
            self.all_nodes_neighbor_sampler()
        else:
            self.sample_on_batching = True

    
    def expand_neighborhood_layer(self, layer_nodes, layer_idx):
        k = self.deg[layer_idx]
        
        if self.sample_on_batching:
            if self.sampler == 'batch_reservoir':
                layer_edge_ids = [sampler_reservoir(self.adj_eidx[v], self.P[v][layer_idx], self.P_inv[v][layer_idx], k) for v in layer_nodes]
            elif self.sampler == 'batch_uniform':
                layer_edge_ids = [sampler_uniform(self.adj_eidx[v], self.P[v][layer_idx], self.P_inv[v][layer_idx], k) for v in layer_nodes]
            else:
                raise ValueError(f'Unkown sampler {self.sampler}')
            layer_edge_ids = np.concatenate(layer_edge_ids)
        else:
            layer_edge_ids = self.sampled_edge_ids[layer_idx][layer_nodes].flatten()

        layer_edge_index = self.edge_index[:, layer_edge_ids]

        new_nodes = layer_edge_index[0, :].squeeze()
        new_layer_nodes = np.concatenate((layer_nodes, new_nodes))
        new_layer_nodes = pd.unique(new_layer_nodes)
        layer_edge_index = torch.from_numpy(layer_edge_index)

        return new_layer_nodes, layer_edge_index, layer_edge_ids
    

    def make_batch(self, core_nodes):
        edges = []
        nodes = []
        edge_ids = []
        layer_nodes = np.array(core_nodes)
        for layer_idx in reversed(range(self.num_layers)):
            layer_nodes, layer_edges, layer_edge_ids = self.expand_neighborhood_layer(layer_nodes, layer_idx)
            edges.append(layer_edges)
            nodes.append(layer_nodes)
            edge_ids.append(layer_edge_ids)

        edges = edges[::-1]
        nodes = nodes[::-1]
        edge_ids = edge_ids[::-1]
        data = Data(x=self.data.x[nodes[0]])

        max_idx = np.max(nodes[0])
        index_mapping = torch.zeros(max_idx+1, dtype=torch.int)
        index_mapping[nodes[0]] = torch.arange(nodes[0].shape[0], dtype=torch.int)
        for l in range(self.num_layers):
            mapped_edge_indexes = index_mapping[edges[l]].long()
            setattr(data, f'edge_index_layer_{l}', mapped_edge_indexes)
            setattr(data, f'edge_type_layer_{l}', self.edge_attr[edge_ids[l]])

        num_layer_nodes = [len(layer_nodes) for layer_nodes in nodes]
        num_layer_nodes.append(len(core_nodes))
        data.num_layer_nodes = torch.Tensor(num_layer_nodes).int()
        # print(f'data.num_layer_nodes: {data.num_layer_nodes}')
        data.y = self.data.y[core_nodes]

        return data


    def all_nodes_neighbor_sampler(self, layer_node_deg=None):
        if layer_node_deg is None:
            layer_node_deg = self.deg
        self.sampled_edge_ids = []
        num_nodes = len(self.P)
        for layer_idx in range(self.num_layers):
            if self.sampler == 'graph_reservoir':
                layer_edge_ids = [sampler_reservoir(self.adj_eidx[v], self.P[v][layer_idx], self.P_inv[v][layer_idx], layer_node_deg[layer_idx]) for v in range(num_nodes)]
            elif self.sampler == 'graph_max':
                layer_edge_ids = [sampler_get_max(self.adj_eidx[v], self.P[v][layer_idx], self.P_inv[v][layer_idx], layer_node_deg[layer_idx]) for v in range(num_nodes)]
            elif self.sampler == 'graph_uniform':
                layer_edge_ids = [sampler_uniform(self.adj_eidx[v], self.P[v][layer_idx], self.P_inv[v][layer_idx], layer_node_deg[layer_idx]) for v in range(num_nodes)]
            else:
                raise ValueError(f'unknown neighbor sampler {self.sampler}')
            layer_edge_ids = np.stack(layer_edge_ids)
            self.sampled_edge_ids.append(layer_edge_ids)