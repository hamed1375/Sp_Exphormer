from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_gt')
def set_cfg_gt(cfg):
    """Configuration for Exphormer Attention Layer
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'ExphormerInitial'

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Size of the edge embedding
    cfg.gt.dim_edge = None

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    cfg.gt.residual = True

    cfg.gt.activation = 'relu'

    cfg.gt.use_edge_feats = True

    # Feed forward network after the Attention layer
    cfg.gt.FFN = True

    # Jumping knowledge concatenate output of all layers to make the final prediction
    cfg.gt.JK = False