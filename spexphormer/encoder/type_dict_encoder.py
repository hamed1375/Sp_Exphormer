import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

@register_node_encoder('TypeDictNode')
class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.node_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'node_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch


@register_edge_encoder('TypeDictEdge')
class TypeDictEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.edge_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'edge_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)

    def forward(self, batch):
        batch.edge_dict = batch.edge_attr
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch


@register_edge_encoder('TypeDictEdge2')
class TypeDictEdgeEncoder2(torch.nn.Module):
    '''
    Edge encoder for the type dictionary that only stores the torch.nn.Embedding weights for the batch.
    '''
    def __init__(self, emb_dim):
        super().__init__()

        self.num_types = cfg.dataset.edge_encoder_num_types
        if self.num_types < 1:
            raise ValueError(f"Invalid 'edge_encoder_num_types': {self.num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=self.num_types,
                                          embedding_dim=emb_dim)

    def forward(self, batch):
        batch.edge_embeddings = self.encoder(torch.arange(self.num_types).to(self.encoder.weight.device))
        return batch