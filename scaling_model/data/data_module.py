import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import QM9
from typing import Optional, List, Union
from torch_geometric.transforms import BaseTransform

from scaling_model.data.utils import ProteinData
from scaling_model.data.random_data import SyntheticData


class baselineBatches:
    def __init__(self):
        pass

    def __call__(self, batch):
        return Batch.from_data_list(batch)


class GetTarget(BaseTransform):
    def __init__(self, target: Optional[int] = None) -> None:
        self.target = [target]

    def forward(self, data: Data) -> Data:
        if self.target is not None:
            data.y = data.y[:, self.target]
        return data


class QM9DataModule(pl.LightningDataModule):

    target_types = ["atomwise" for _ in range(19)]
    target_types[0] = "dipole_moment"
    target_types[5] = "electronic_spatial_extent"

    # Specify unit conversions (eV to meV).
    unit_conversion = {
        i: (lambda t: 1000 * t) if i not in [0, 1, 5, 11, 16, 17, 18] else (lambda t: t)
        for i in range(19)
    }

    def __init__(
        self,
        target: int = 0,
        data_dir: str = "qm9_data/",
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [0.8, 0.1, 0.1],
        seed: int = 0,
        subset_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None:
        # Download data
        QM9(root=self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[: self.subset_size]

        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)
        self.data_train = dataset[: split_idx[0]]
        self.data_val = dataset[split_idx[0] : split_idx[1]]
        self.data_test = dataset[split_idx[1] :]

    def get_target_stats(self, remove_atom_refs=False, divide_by_atoms=False):
        atom_refs = self.data_train.atomref(self.target)

        ys = list()
        for batch in self.train_dataloader(shuffle=False):
            y = batch.y.clone()
            if remove_atom_refs and atom_refs is not None:
                y.index_add_(dim=0, index=batch.batch, source=-atom_refs[batch.z])
            if divide_by_atoms:
                _, num_atoms = torch.unique(batch.batch, return_counts=True)
                y = y / num_atoms.unsqueeze(-1)
            ys.append(y)

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std(), atom_refs

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class BaselineDataModule(pl.LightningDataModule):

    target_types = ["atomwise" for _ in range(5)]

    def __init__(
        self,
        sampler: str = "baseline",
        target: int = 0,
        data_dir: str = "data/",
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        max_protein_size: int = torch.inf,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [0.8, 0.1, 0.1],
        seed: int = 0,
        subset_size: Optional[int] = None,
        download: Optional[bool] = False,
        random_data: Optional[bool] = False,
        shadow_depth: Optional[int] = 4,
        shadow_num_neighbors: Optional[int] = 50,
        cluster_num_parts: Optional[int] = 30,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.max_protein_size = max_protein_size
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.download = download

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.random_data = random_data
        collate_dict = {
            "baseline": baselineBatches(),
        }
        self.collate = collate_dict.get(self.sampler, baselineBatches())

    def prepare_data(self) -> None:
        if self.download:
            ProteinData(self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.random_data:
            dataset = SyntheticData(
                root=self.data_dir,
                max_protein_size=self.max_protein_size,
                transform=GetTarget(self.target),
            )
        else:
            dataset = ProteinData(
                root=self.data_dir,
                max_protein_size=self.max_protein_size,
                transform=GetTarget(self.target),
            )

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[: self.subset_size]

        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)
        self.data_train = dataset[: split_idx[0]]
        self.data_val = dataset[split_idx[0] : split_idx[1]]
        self.data_test = dataset[split_idx[1] :]

    def get_target_stats(self, remove_atom_refs=False, divide_by_atoms=False):
        atom_refs = self.data_train.atomref(self.target)

        ys = list()
        for batch in self.train_dataloader(shuffle=False):
            y = batch.y.clone()
            if remove_atom_refs and atom_refs is not None:
                y.index_add_(dim=0, index=batch.batch, source=-atom_refs[batch.z])
            if divide_by_atoms:
                _, num_atoms = torch.unique(batch.batch, return_counts=True)
                y = y / num_atoms.unsqueeze(-1)
            ys.append(y)

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std(), atom_refs

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate,
        )


if __name__ == "__main__":
    cfg = {
        "target": 0,
        "data_dir": "./data_random/",
        "max_protein_size": 2000,
        "batch_size_train": 1,
        "batch_size_inference": 1,
        "num_workers": 12,
        "splits": [0.8, 0.1, 0.1],
        "seed": 69,
        "subset_size": None,
        "random_data": True,
        "shadow_depth": 4,
        "shadow_num_neighbors": 50,
        "cluster_num_parts": 2,
    }
    dm = BaselineDataModule(sampler="cluster", **cfg)
    dm.setup()
    batches = dm.train_dataloader(shuffle=False)
    print(list(batches))
