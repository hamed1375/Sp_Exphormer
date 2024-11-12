import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer


class SpexphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, layer_idx, use_bias=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.layer_idx = layer_idx

        self.edge_index_name = f'edge_index_layer_{layer_idx}'
        self.edge_attr_name = f'edge_type_layer_{layer_idx}'

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E1 = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E2 = nn.Linear(in_dim, num_heads, bias=True)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

    def forward(self, batch):
        edge_index = getattr(batch, self.edge_index_name)
        edge_attr = getattr(batch, self.edge_attr_name)
        
        n1 = batch.num_layer_nodes[self.layer_idx].item()
        n2 = batch.num_layer_nodes[self.layer_idx + 1].item()
        assert batch.x.shape[0] == n1

        Q_h = self.Q(batch.x[:n2]).view(-1, self.num_heads, self.out_dim)
        K_h = self.K(batch.x).view(-1, self.num_heads, self.out_dim)
        V_h = self.V(batch.x).view(-1, self.num_heads, self.out_dim)

        if cfg.dataset.edge_encoder_name == 'TypeDictEdge2':
            E1 = self.E1(batch.edge_embeddings)[edge_attr].view(n2, -1, self.num_heads, self.out_dim)
            E2 = self.E2(batch.edge_embeddings)[edge_attr].view(n2, -1, self.num_heads, 1)
        else:
            E1 = self.E1(edge_attr).view(n2, -1, self.num_heads, self.out_dim)
            E2 = self.E2(edge_attr).view(n2, -1, self.num_heads, 1)
        
        neighbors = edge_index[0, :]
        deg = neighbors.shape[0]//n2
        neighbors = neighbors.reshape(n2, deg)
        
        K_h = K_h[neighbors]
        V_h = V_h[neighbors]
        
        score = torch.mul(E1, K_h)
        
        score = torch.bmm(score.view(-1, deg, self.out_dim), Q_h.view(-1, self.out_dim, 1))
        score = score.view(-1, self.num_heads, deg)

        score = score + E2.squeeze(-1).permute([0, 2, 1])
        score = score.clamp(-8, 8)
        score = F.softmax(score, dim=-1)
        
        V_h = V_h.permute(0, 2, 1, 3)
        score = score.unsqueeze(-1)
        h_out = torch.mul(score, V_h)
        h_out = h_out.sum(dim=2)
        h_out = h_out.reshape(n2, -1)

        return h_out

register_layer('ExphormerFinal', SpexphormerAttention)
register_layer('ExphormerSecond', SpexphormerAttention)
register_layer('ExphormerRegularGraph', SpexphormerAttention)