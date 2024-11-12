import os.path as osp
import os
from typing import Callable, Optional
import scipy
import numpy as np

import torch

from torch_geometric.data import Data, InMemoryDataset

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    # To have a fixed splits:
    np.random.seed(123)

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

class Pokec(InMemoryDataset):

    # you can download from here manually, google drive downloader did not work :( 
    url = 'https://drive.google.com/file/d/1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return 'pokek.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # I did try a few libraries to download from google drive, it did not work, so please download it manually from:
        # https://drive.google.com/file/d/1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y
        pass

    def process(self):
        """ requires pokec.mat """
        if not osp.exists(f'{self.raw_dir}/pokec.mat'):
            raise Exception("Please download the pokec.mat manually from https://drive.google.com/file/d/1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y")
    
        fulldata = scipy.io.loadmat(f'{self.raw_dir}/pokec.mat')
        edge_index = fulldata['edge_index']
        node_feat = fulldata['node_feat']
        label = fulldata['label']
    
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_feat = torch.tensor(node_feat).float()
        label = torch.tensor(label, dtype=torch.long).view(-1, 1)

        data = Data(x = node_feat, edge_index=edge_index, y=label)
    
        
        train_prop = 0.1
        val_prop = 0.1
        split_dir = f'{self.raw_dir}/split_{train_prop}_{val_prop}'
        tensor_split_idx = {}
        if osp.exists(split_dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(split_dir + '/pokec_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(split_dir + '/pokec_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(split_dir + '/pokec_test.txt'), dtype=torch.long)
        else:
            os.makedirs(split_dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] \
                = rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=val_prop)
            np.savetxt(split_dir + '/pokec_train.txt', tensor_split_idx['train'], fmt='%d')
            np.savetxt(split_dir + '/pokec_valid.txt', tensor_split_idx['valid'], fmt='%d')
            np.savetxt(split_dir + '/pokec_test.txt', tensor_split_idx['test'], fmt='%d')

        train_mask = torch.zeros(data.y.shape, dtype=torch.bool)
        train_mask[tensor_split_idx['train']] = True
        val_mask = torch.zeros(data.y.shape, dtype=torch.bool)
        val_mask[tensor_split_idx['valid']] = True
        test_mask = torch.zeros(data.y.shape, dtype=torch.bool)
        test_mask[tensor_split_idx['test']] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'