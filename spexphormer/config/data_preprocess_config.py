from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_preprocess(cfg):
    """Extend configuration with preprocessing options
    """

    cfg.prep = CN()

    # Argument group for adding expander edges

    # if it's enabled expander edges would be available by e.g. data.expander_edges
    cfg.prep.exp = True
    cfg.prep.exp_deg = 5
    cfg.prep.exp_algorithm = 'Hamiltonian' # options are 'Hamiltonian', 'Random-d', 'Random-d2'

    cfg.prep.use_exp_edges = True
    cfg.prep.replace_combined_exp_edges = False
    cfg.prep.exp_max_num_iters = 100
    cfg.prep.add_edge_index = True
    cfg.prep.num_virt_node = 0

    cfg.prep.add_self_loops = False
    cfg.prep.add_reverse_edges = True
    cfg.prep.layer_edge_indices_dir = None
    cfg.prep.save_edges = False
    cfg.prep.load_edges = False
    cfg.prep.num_edge_sets = 1
    cfg.prep.edge_set_name = None

    cfg.prep.default_initial = False


register_config('preprocess', set_cfg_preprocess)
