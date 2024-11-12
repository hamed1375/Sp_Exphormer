import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

import sys
import math
import time
import pickle as pkl
import scipy as sp
from scipy import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from torch_geometric.data import Data, InMemoryDataset, download_url



class GeomGCNHeterogeneousDatasets(InMemoryDataset):
    r"""Datasets: Penn94, Actor, Squirrel, Chameleon from
    `"Geom-GCN: Geometric Graph Convolutional Networks"
            <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Parts of code are from: <https://github.com/DSL-Lab/Specformer>

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Penn94"`,
            :obj:`"Actor"`, :obj:`"Squirrel"`, :obj:`"Chameleon"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    
    def __init__(self, root: str, name: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, train_percent = 0.6):
        self.name = name

        self.url = 'https://github.com/DSL-Lab/Specformer/tree/main/Node/node_raw_data'
        self.url_penn = 'https://github.com/DSL-Lab/Specformer/raw/main/Node/node_raw_data'
        self.url_penn_split = 'https://github.com/DSL-Lab/Specformer/raw/main/Node/node_raw_data/fb100-Penn94-splits.npy'


        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        data = self.get(0)
        self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.name == 'Penn94':
            names = ['Penn94.mat', 'fb100-Penn94-splits.npy']
        else:
            names = ['out1_graph_edges.txt', 'out1_node_feature_label.txt']
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        if self.name == 'Penn94':
            for name in self.raw_file_names:
                download_url(f'{self.url_penn}/{name}', self.raw_dir)
        else:
            for name in self.raw_file_names:
                download_url(f'{self.url_penn}/{self.name}/{name}', self.raw_dir)

    def process(self):

        def feature_normalize(x):
            x = np.array(x)
            rowsum = x.sum(axis=1, keepdims=True)
            rowsum = np.clip(rowsum, 1, 1e10)
            return x / rowsum

        if self.name == 'Penn94':
            mat = io.loadmat(osp.join(self.raw_dir, 'Penn94.mat'))
            A = mat['A']
            metadata = mat['local_info']
        
            edge_index = A.nonzero()
            metadata = metadata.astype(int)
            label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        
            # make features into one-hot encodings
            feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
            features = np.empty((A.shape[0], 0))
            for col in range(feature_vals.shape[1]):
                feat_col = feature_vals[:, col]
                feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
                features = np.hstack((features, feat_onehot))
        
            node_feat = torch.tensor(features, dtype=torch.float)
            num_nodes = metadata.shape[0]
            label = torch.LongTensor(label)
            edge_index = torch.LongTensor(edge_index).contiguous()
            data = Data(x=node_feat, edge_index=edge_index, y=label)

            split = np.load(osp.join(self.raw_dir, 'fb100-Penn94-splits.npy'), allow_pickle=True)[0]
            train, valid, test = split['train'], split['valid'], split['test']
            train_mask = torch.zeros(data.y.shape, dtype=torch.bool)
            train_mask[train] = True
            val_mask = torch.zeros(data.y.shape, dtype=torch.bool)
            val_mask[valid] = True
            test_mask = torch.zeros(data.y.shape, dtype=torch.bool)
            test_mask[test] = True

            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

        elif self.name in ['chameleon', 'squirrel', 'actor']:
            edge_df = pd.read_csv(osp.join(self.raw_dir, 'out1_graph_edges.txt'), sep='\t')
            node_df = pd.read_csv(osp.join(self.raw_dir, 'out1_node_feature_label.txt'), sep='\t')
            feature = node_df[node_df.columns[1]]
            y = node_df[node_df.columns[2]]
        
            source = list(edge_df[edge_df.columns[0]])
            target = list(edge_df[edge_df.columns[1]])
        
            if self.name == 'actor':
                # for sparse features
                nfeat = 932
                x = np.zeros((len(y), nfeat))
        
                feature = list(feature)
                feature = [feat.split(',') for feat in feature]
                for ind, feat in enumerate(feature):
                    for ff in feat:
                        x[ind, int(ff)] = 1.
                x = feature_normalize(x)
            else:
                feature = list(feature)
                feature = [feat.split(',') for feat in feature]
                new_feat = []
                for feat in feature:
                    new_feat.append([int(f) for f in feat])
                x = np.array(new_feat)
                x = feature_normalize(x)
                
            edge_index = [source, target]
            edge_index = torch.LongTensor(edge_index).contiguous()
            data = Data(x=torch.Tensor(x), edge_index=edge_index, y=torch.LongTensor(y))

            y = data.y
            nclass = 5

            percls_trn = int(round(0.5 * len(y) / nclass))
            val_lb = int(round(0.25 * len(y)))

            indices = []
            for i in range(nclass):
                index = (y == i).nonzero().view(-1)
                index = index[torch.randperm(index.size(0), device=index.device)]
                indices.append(index)

            train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
            rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
            rest_index = rest_index[torch.randperm(rest_index.size(0))]
            valid_index = rest_index[:val_lb]
            test_index = rest_index[val_lb:]

            train_mask = torch.zeros(data.y.shape, dtype=torch.bool)
            train_mask[train_index] = True
            val_mask = torch.zeros(data.y.shape, dtype=torch.bool)
            val_mask[valid_index] = True
            test_mask = torch.zeros(data.y.shape, dtype=torch.bool)
            test_mask[test_index] = True

            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
