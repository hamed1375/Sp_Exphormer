import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn

from spexphormer.layer.Exphormer_Attention import ExphormerAttention
from spexphormer.layer.ASE_Attention import ASE_Attention_Layer
from spexphormer.layer.Spexphormer_Attention import SpexphormerAttention
import warnings



class AttentionLayer(nn.Module):
    """
    Attention layer
    """

    def __init__(self, dim_h, exphormer_model_type, num_heads,
                layer_idx, dropout=0.0, layer_norm=False,
                batch_norm=True, exp_edges_cfg=None):

        super().__init__()

        self.dim_h = dim_h
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.num_heads = num_heads
        self.layer_idx = layer_idx
            

        if exphormer_model_type == 'Exphormer':
            self.self_attn = ExphormerAttention(dim_h, dim_h, num_heads, layer_idx=self.layer_idx,
                                          use_virt_nodes= exp_edges_cfg.num_virt_node > 0, use_bias=False)
        elif exphormer_model_type == 'ASE':
            if num_heads != 1:
                warnings.warn('numer of head for the initial network should be always 1, if you want to use the attention scores for a final network')
            self.self_attn = ASE_Attention_Layer(dim_h, dim_h, num_heads, layer_idx=self.layer_idx)
        elif exphormer_model_type == 'Spexphormer':
            self.self_attn = SpexphormerAttention(dim_h, dim_h, num_heads, layer_idx=self.layer_idx)
        else:
            raise ValueError(f"Unsupported exphormer model: "
                             f"{exphormer_model_type}")
        self.exphormer_model_type = exphormer_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for Self-Attention representation.
        if self.layer_norm:
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_attn = self.self_attn(batch)
        h_attn = self.dropout(h_attn)

        if h_attn.shape == h_in1.shape:
            h_attn = h_in1 + h_attn  # Residual connection.
        else:
            h_attn = h_in1[:h_attn.shape[0]] + h_attn

        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batch.batch)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        return h_attn


class SpExphormerFullLayer(nn.Module):
    """Variants of the Exphormer
    """

    def __init__(self, dim_h,
                 layer_type, num_heads, layer_idx, dropout=0.0,
                 layer_norm=False, batch_norm=True, use_ffn=True,
                 exp_edges_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.use_ffn = use_ffn

        # Local message-passing models.
        self.attention_layer = []

        if layer_type in {'Exphormer', 'Spexphormer', 'ASE'}:
            self.attention_layer = AttentionLayer(dim_h=dim_h,
                                        exphormer_model_type=layer_type,
                                        num_heads=self.num_heads,
                                        layer_idx=self.layer_idx,
                                        dropout=dropout,
                                        layer_norm=self.layer_norm,
                                        batch_norm=self.batch_norm,
                                        exp_edges_cfg = exp_edges_cfg)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.activation = F.relu

        if self.use_ffn:
            # Feed Forward block.
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
            
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)
        else:
            self.dropout = nn.Dropout(dropout)
        
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)

    def forward(self, batch):
        h = self.attention_layer(batch)

        if self.use_ffn:
            # Feed Forward block.
            h = h + self._ff_block(h)
        else:
            h = self.dropout(self.activation(h))
        
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'layer_type={self.layer_type}, ' \
            f'heads={self.num_heads}'
        return s