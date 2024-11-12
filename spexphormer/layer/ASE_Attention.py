import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer


def separate_by_node(edges, scores):
    num_nodes = edges.max().item() + 1
    neighbors = [[] for _ in range(num_nodes)]
    neighbors_edge_idx = [[] for _ in range(num_nodes)]
    P = [[] for _ in range(num_nodes)]
    sum_scores = torch.zeros(num_nodes, dtype=scores.dtype)
    rev_sum_scores = torch.zeros(num_nodes, dtype=scores.dtype)
    
    for i in range(edges.size(1)):
        u, v = edges[:, i]
        neighbors[v].append(u.item())
        neighbors_edge_idx[v].append(i)
        w = scores[i].item()
        sum_scores[v] += w
        rev_sum_scores[u] += w
        P[v].append(w)

    return neighbors, neighbors_edge_idx, P, sum_scores.tolist(), rev_sum_scores.tolist()


class ASE_Attention_Layer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, layer_idx, use_bias=False, dim_edge=None, use_virt_nodes=False):
        super().__init__()

        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias
        self.layer_idx = layer_idx

        if dim_edge is None:
            dim_edge = in_dim

        self.QKV = nn.Linear(in_dim, self.out_dim * num_heads * 3, bias=use_bias)
        self.V_scale = nn.Parameter(data=torch.Tensor([0.25]), requires_grad=True)

        self.use_edge_attr = cfg.gt.use_edge_feats
        if self.use_edge_attr:
            self.E1 = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
            self.E2 = nn.Linear(dim_edge, num_heads, bias=True)

        self.T = 1.0

    def propagate_attention(self, batch, edge_index):
        src = batch.K_h[edge_index[0].to(torch.long)]  # (num edges) x num_heads x out_dim
        dest = batch.Q_h[edge_index[1].to(torch.long)]  # (num edges) x num_heads x out_dim
        score = torch.einsum('ehd,ehd->eh', src, dest)  # Efficient batch matrix multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.use_edge_attr:
            score = score * batch.E.sum(-1)  # (num real edges) x num_heads
            score = score.unsqueeze(-1) + batch.E2
        else:
            score = score.unsqueeze(-1)

        score = score / self.T
        score = torch.exp(score.clamp(-8, 8))


        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[edge_index[0].to(torch.long)] * score  # (num real edges) x num_heads x out_dim
        
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        batch.wV.index_add_(0, edge_index[1].to(torch.long), msg)  # Using index_add_ as an alternative to scatter

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.V_h.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        batch.Z.index_add_(0, edge_index[1].to(torch.long), score)  # Using index_add_ as an alternative to scatter

        if cfg.train.saving_epoch:
            new_score = score/(batch.Z[edge_index[1]])
            score_np = new_score.cpu().detach().numpy()
            Path(f'Attention_scores/{cfg.dataset.name}').mkdir(parents=True, exist_ok=True)
            with open(f'Attention_scores/{cfg.dataset.name}/seed{cfg.seed}_h{self.out_dim}_layer_{self.layer_idx}.npy', 'wb') as f:
                np.save(f, score_np)


    def forward(self, batch):
        if cfg.train.cur_epoch >= cfg.train.temp_wait:
            self.T = max(cfg.train.temp_min, cfg.train.temp_rdc_ratio ** (cfg.train.cur_epoch - cfg.train.temp_wait))
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        h = batch.x
        num_node = batch.batch.shape[0]
        
        QKV_h = self.QKV(h).view(-1, self.num_heads, 3 * self.out_dim)
        batch.Q_h, batch.K_h, batch.V_h = torch.split(QKV_h, self.out_dim, dim=-1)
        batch.V_h = F.normalize(batch.V_h, p=2.0, dim=-1) * self.V_scale

        if self.use_edge_attr:
            if cfg.dataset.edge_encoder_name == 'TypeDictEdge2':
                E = self.E1(batch.edge_embeddings)[batch.edge_attr]
                E2 = self.E2(batch.edge_embeddings)[batch.edge_attr]
            else:
                E = self.E1(edge_attr)
                E2 = self.E2(edge_attr)
            
            batch.E = E.view(-1, self.num_heads, self.out_dim)
            batch.E2 = E2.view(-1, self.num_heads, 1)

        self.propagate_attention(batch, edge_index)

        # Normalize the weighted sum of values by the normalization coefficient
        h_out = batch.wV / (batch.Z + 1e-6)

        # Reshape the output to combine the heads
        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        # Separate virtual node embeddings if virtual nodes are used
        if self.use_virt_nodes:
            batch.virt_h = h_out[num_node:]
            h_out = h_out[:num_node]

        return h_out


register_layer('ASE_Attention_Layer', ASE_Attention_Layer)
