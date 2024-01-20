import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data
import torch

class Imbalance:

    def __init__(self, name,data, ratio):
        self.name = name
        self.data = data
        self.total_node = len(data.x)
        self.label = data.y
        self.ratio = int(ratio)
        self.data_train_mask = data.train_mask.clone()
        self.n_cls = data.y.max().item() + 1


    def n_data(self):
        n_data = []
        stats = self.data.y[self.data_train_mask]
        for i in range(self.n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        return n_data


    def class_num_list(self):
        class_num_list = []
        if self.name == 'Cora':
            class_sample_num = 20
            imb_class_num = 3
        elif self.name == 'CiteSeer':
            class_sample_num = 20
            imb_class_num = 3
        elif self.name == 'PubMed':
            class_sample_num = 20
            imb_class_num = 1
        else:
            print("no this dataset: {args.dataset}")
        for i in range(self.n_cls):
            if self.ratio > 1 and i > self.n_cls - 1 - imb_class_num:  # only imbalance the last classes
                class_num_list.append(int(class_sample_num * (1. / self.ratio)))
            else:
                class_num_list.append(class_sample_num)

        return class_num_list



    def get_idx_info(self):
        index_list = torch.arange(len(self.label))
        idx_info = []
        for i in range(self.n_cls):
            cls_indices = index_list[((self.label == i) & self.data_train_mask)]
            idx_info.append(cls_indices)
        return idx_info


    def split_semi_dataset(self):
        n_data = self.n_data()
        class_num_list = self.class_num_list()
        idx_info = self.get_idx_info()
        new_idx_info = []
        _train_mask = idx_info[0].new_zeros(self.total_node, dtype=torch.bool)
        for i in range(self.n_cls):
            if n_data[i] > class_num_list[i]:
                cls_idx = torch.randperm(len(idx_info[i]))
                cls_idx = idx_info[i][cls_idx]
                cls_idx = cls_idx[:class_num_list[i]]
                new_idx_info.append(cls_idx)
            else:
                new_idx_info.append(idx_info[i])
            _train_mask[new_idx_info[i]] = True
        assert _train_mask.sum().long() == sum(class_num_list)
        assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

        return _train_mask

class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"geom-gcn"`,
            :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the `"Revisiting Semi-Supervised Learning with Graph
            Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
            `"Geom-GCN: Geometric Graph Convolutional Networks"
            <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1
            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Cora
              - 2,708
              - 10,556
              - 1,433
              - 7
            * - CiteSeer
              - 3,327
              - 9,104
              - 3,703
              - 6
            * - PubMed
              - 19,717
              - 88,648
              - 500
              - 3
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/master')

    def __init__(self, root: str, name: str, split: str = "public",ratio=1,
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        self.ratio = ratio
        self.split = split.lower()
        assert self.split in ['public', 'full', 'geom-gcn', 'random']

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.train_mask = Imbalance(self.name,self.data,self.ratio).split_semi_dataset()

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'raw')
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'processed')
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)
        if self.split == 'geom-gcn':
            for i in range(10):
                url = f'{self.geom_gcn_url}/splits/{self.name.lower()}'
                download_url(f'{url}_split_0.6_0.2_{i}.npz', self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == 'geom-gcn':
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits['train_mask']))
                val_masks.append(torch.from_numpy(splits['val_mask']))
                test_masks.append(torch.from_numpy(splits['test_mask']))
            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(val_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


















