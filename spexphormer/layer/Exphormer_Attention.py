import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer



class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, layer_idx, dim_edge=None, use_virt_nodes=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias
        self.layer_idx = layer_idx

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

        self.use_edge_attr = cfg.gt.use_edge_feats
        if self.use_edge_attr:
            self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)


    def propagate_attention(self, batch, edge_index):
        src = batch.K_h[edge_index[0].to(torch.long)]  # (num edges) x num_heads x out_dim
        dest = batch.Q_h[edge_index[1].to(torch.long)]  # (num edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim

        # score = torch.exp(score.sum(-1, keepdim=True))
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))


        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[edge_index[0].to(torch.long)] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        batch.wV.index_add_(0, edge_index[1].to(torch.long), msg)

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.V_h.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        batch.Z.index_add_(0, edge_index[1].to(torch.long), score)


    def forward(self, batch):
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        h = batch.x
        num_node = batch.batch.shape[0]
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        if self.use_edge_attr:
            if cfg.dataset.edge_encoder_name == 'TypeDictEdge2':
                E = self.E(batch.edge_embeddings)[batch.edge_attr]
            else:
                E = self.E(edge_attr)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        h_out = batch.wV / (batch.Z + 1e-6)

        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        batch.virt_h = h_out[num_node:]
        h_out = h_out[:num_node]

        return h_out


register_layer('Exphormer', ExphormerAttention)