import logging
import os.path as osp
import time
import math
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import (Coauthor, GNNBenchmarkDataset, TUDataset,
                                      WikipediaNetwork, ZINC)
from spexphormer.loader.dataset.Amazon_with_split import AmazonWithSplit
from spexphormer.loader.dataset.HeterophilousGraphDataset import HeterophilousGraphDataset
from torch_geometric.datasets import WikiCS
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from spexphormer.loader.planetoid import Planetoid
from spexphormer.loader.heterogeneous_datasets import GeomGCNHeterogeneousDatasets
from spexphormer.loader.dataset.aqsol_molecules import AQSOL
from spexphormer.loader.dataset.coco_superpixels import COCOSuperpixels
from spexphormer.loader.dataset.malnet_tiny import MalNetTiny
from spexphormer.loader.dataset.voc_superpixels import VOCSuperpixels
from spexphormer.loader.dataset.Pokec import Pokec
from spexphormer.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from spexphormer.transform.posenc_stats import compute_posenc_stats
from spexphormer.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos)
from spexphormer.transform.expander_edges import augment_with_expander, load_edges
from spexphormer.transform.dist_transforms import (add_reverse_edges,
                                                 add_self_loops,
                                                 effective_resistance_embedding,
                                                 effective_resistances_from_embedding)


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  avg degree: {dataset.data.edge_index.shape[1]/dataset.data.x.shape[0]}")
    # logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Amazon':
            dataset = AmazonWithSplit(dataset_dir, name)
            if name == "photo" or name == "computers":
                if cfg.prep.add_self_loops:
                  pre_transform_in_memory(dataset, partial(add_self_loops))

        elif pyg_dataset_id == 'Coauthor':
            dataset = Coauthor(dataset_dir, name)
            if name == "physics" or name == "cs":
                if cfg.prep.add_self_loops:
                  pre_transform_in_memory(dataset, partial(add_self_loops))

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name, split='random', train_percent= cfg.prep.train_percent)
            if name == "PubMed":
                pre_transform_in_memory(dataset, partial(typecast_x, type_str='float'))
            if cfg.prep.add_reverse_edges == True:
              pre_transform_in_memory(dataset, partial(add_reverse_edges))
            if cfg.prep.add_self_loops == True:
              pre_transform_in_memory(dataset, partial(add_self_loops))

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)


        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
            
        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        elif pyg_dataset_id in ['Penn94', 'actor', 'squirrel', 'chameleon']:
            dataset = preformat_HeteroDataset(dataset_dir, name)

        elif pyg_dataset_id == "Heterophilous":
            if name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
                dataset = HeterophilousGraphDataset(dataset_dir, name)
            else:
                raise ValueError("dataset is not in the HeterophilousGraphDataset")
        
        elif pyg_dataset_id == 'WikiCS':
            dataset = WikiCS(root=dataset_dir)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)
        elif name.startswith('ogbn'):
            dataset = preformat_ogbn(dataset_dir, name)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    elif format == 'SNAP' and name == 'pokec':
        dataset = Pokec(root=dataset_dir, name='pokec')
    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable and (not key.startswith('posenc_ER')):
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # Other preprocessings:
    # adding expander edges:
    if cfg.prep.exp:
        start = time.perf_counter()
        logging.info(f"Adding expander edges ...")
        pre_transform_in_memory(dataset,
                                partial(augment_with_expander,
                                        degree = cfg.prep.exp_deg,
                                        algorithm = cfg.prep.exp_algorithm,
                                        rng = None,
                                        max_num_iters = cfg.prep.exp_max_num_iters),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                    + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # loading precalculated edges:
    if cfg.prep.load_edges:
        start = time.perf_counter()
        logging.info(f"loading edge sets")
        pre_transform_in_memory(dataset, load_edges, show_progress=True)
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                    + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")


    # adding effective resistance features
    if cfg.posenc_ERN.enable or cfg.posenc_ERE.enable:
        start = time.perf_counter()
        logging.info(f"Precalculating effective resistance for graphs ...")
        
        MaxK = max(
            [
                min(
                math.ceil(data.num_nodes//2), 
                math.ceil(8 * math.log(data.num_edges) / (cfg.posenc_ERN.accuracy**2))
                ) 
                for data in dataset
            ]
            )

        cfg.posenc_ERN.er_dim = MaxK
        logging.info(f"Choosing ER pos enc dim = {MaxK}")

        pre_transform_in_memory(dataset,
                                partial(effective_resistance_embedding,
                                        MaxK = MaxK,
                                        accuracy = cfg.posenc_ERN.accuracy,
                                        which_method = 0),
                                show_progress=True
                                )

        pre_transform_in_memory(dataset,
                        partial(effective_resistances_from_embedding,
                        normalize_per_node = False),
                        show_progress=True
                        )

        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    
    # # adding layer edge indices:
    # if 'LayerETransformer' in cfg.gt.layer_type:
    #     start = time.perf_counter()
    #     logging.info(f"loading edge indices for different layers")
    #     pre_transform_in_memory(dataset,
    #                             partial(add_layer_edge_indices,
    #                                     dir=cfg.prep.layer_edge_indices_dir,
    #                                     layers=cfg.gt.layers),
    #                             show_progress=True
    #                             )
    #     elapsed = time.perf_counter() - start
    #     timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
    #                 + f'{elapsed:.2f}'[-3:]
    #     logging.info(f"Done! Took {timestr}")

    # This could not be done earlier because the training wants 'train_mask' etc.
    # Now after using gnn.head: inductive_node this is ok.
    # if name == 'ogbn-arxiv' or name == 'ogbn-proteins':
    #   return dataset


    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNAConv') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])

    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset = join_dataset_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]

    return dataset


def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-ppa':
        # ogbg-ppa doesn't have any node features, therefore add zeros but do
        # so dynamically as a 'transform' and not as a cached 'pre-transform'
        # because the dataset is big (~38.5M nodes), already taking ~31GB space
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
        dataset.transform = add_zeros

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name):
    """Load and preformat PCQM4Mv2 from OGB LSC.

    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission

    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.

    Note: PygPCQM4Mv2Dataset requires rdkit

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e


    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]
    elif name == 'subset':
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.
        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset


def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from spexphormer.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir, subset='530k')
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset


def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from spexphormer.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from spexphormer.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset

def preformat_ogbn(dataset_dir, name):
    print(name)
    if name == 'ogbn-arxiv' or name == 'ogbn-proteins' or name == 'ogbn-products' or name == 'ogbn-papers100M':
        if name == 'ogbn-products':
             #TODO: add reverse edges for the Amazon or not?
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
        elif name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
            pre_transform_in_memory(dataset, partial(add_reverse_edges))
            if cfg.prep.add_self_loops:
                pre_transform_in_memory(dataset, partial(add_self_loops))
        elif name == 'ogbn-papers100M':
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
            # pre_transform_in_memory(dataset, partial(add_reverse_edges))
            # if cfg.prep.add_self_loops:
            #     pre_transform_in_memory(dataset, partial(add_self_loops))
        elif name == 'ogbn-proteins':
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
            data = dataset.data
            dataset.data.x = torch.zeros((data.num_nodes, data.edge_attr.size(1)), dtype=data.edge_attr.dtype)
            dataset.data.x.index_add_(0, data.edge_index[0], data.edge_attr)
            dataset.data.x = dataset.data.x / torch.bincount(data.edge_index[0]).view(-1, 1)
            dataset.data.edge_attr = None

        split_dict = dataset.get_idx_split()
        # split_dict['val'] = split_dict.pop('valid')
        # dataset.split_idxs = split_dict

        train_mask = torch.zeros(dataset.data.y.shape[0], dtype=torch.bool)
        train_mask[split_dict['train']] = True
        val_mask = torch.zeros(dataset.data.y.shape[0], dtype=torch.bool)
        val_mask[split_dict['valid']] = True
        test_mask = torch.zeros(dataset.data.y.shape[0], dtype=torch.bool)
        test_mask[split_dict['test']] = True
        dataset.data.train_mask = train_mask
        dataset.data.val_mask = val_mask
        dataset.data.test_mask = test_mask

        return dataset

    else:
        ValueError(f"Unknown ogbn dataset '{name}'.")

def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_AQSOL(dataset_dir):
    """Load and preformat AQSOL datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [AQSOL(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_HeteroDataset(dataset_dir, name):
    """Load and preformat datasets from Heterogeneous datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the Heterogeneous datasets class

    Returns:
        PyG dataset object
    """
    dataset = GeomGCNHeterogeneousDatasets(root=dataset_dir, name=name)
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
