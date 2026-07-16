import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data

from src.imbalance import Imbalance


class Planetoid(InMemoryDataset):
    """Planetoid datasets with the class-imbalanced split used by UNREAL."""

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    geom_gcn_url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "public",
        ratio=1,
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.ratio = ratio
        self.split = split.lower()
        if self.split not in {"public", "full", "geom-gcn", "random"}:
            raise ValueError(f"Unsupported split: {split}")

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.train_mask = Imbalance(
            self.name,
            self.data,
            self.ratio,
        ).split_semi_dataset()

        if self.split == "full":
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])
        elif self.split == "random":
            data = self.get(0)
            data.train_mask.fill_(False)
            for class_index in range(self.num_classes):
                indices = (data.y == class_index).nonzero(as_tuple=False).view(-1)
                permutation = torch.randperm(indices.size(0))
                data.train_mask[indices[permutation[:num_train_per_class]]] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val : num_val + num_test]] = True
            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "raw")
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "processed")
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name.lower()}.{name}" for name in names]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f"{self.url}/{name}", self.raw_dir)

        if self.split == "geom-gcn":
            base_url = f"{self.geom_gcn_url}/splits/{self.name.lower()}"
            for split_index in range(10):
                download_url(
                    f"{base_url}_split_0.6_0.2_{split_index}.npz",
                    self.raw_dir,
                )

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == "geom-gcn":
            train_masks = []
            validation_masks = []
            test_masks = []
            for split_index in range(10):
                filename = f"{self.name.lower()}_split_0.6_0.2_{split_index}.npz"
                splits = np.load(osp.join(self.raw_dir, filename))
                train_masks.append(torch.from_numpy(splits["train_mask"]))
                validation_masks.append(torch.from_numpy(splits["val_mask"]))
                test_masks.append(torch.from_numpy(splits["test_mask"]))

            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(validation_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"
