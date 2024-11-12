import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from spexphormer.layer.SpExphormer_full_layer import SpExphormerFullLayer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gt.dim_hidden)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gt.dim_hidden, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            self.dim_in = cfg.gt.dim_hidden
        if cfg.dataset.edge_encoder:
            if  getattr(cfg.gt, 'dim_edge', None) is None:
                cfg.gt.dim_edge = cfg.gt.dim_hidden
            
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gt.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gt.dim_edge, -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class SpExphormer_Network(torch.nn.Module):
    '''
    This model can be used for creating the variants of Exphormer and Spexphormer networks.
    '''
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gt.dim_hidden, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gt.dim_hidden
        else:
            self.pre_mp = None

        layers = []
        
        for i in range(cfg.gt.layers):
            layers.append(SpExphormerFullLayer(
                dim_h=cfg.gt.dim_hidden,
                layer_type=cfg.gt.layer_type,
                num_heads=cfg.gt.n_heads,
                layer_idx=i,
                dropout=cfg.gt.dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                use_ffn=cfg.gt.FFN,
                exp_edges_cfg=cfg.prep
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)

        return batch
    
register_network('Exphormer', SpExphormer_Network)
register_network('Spexphormer', SpExphormer_Network)