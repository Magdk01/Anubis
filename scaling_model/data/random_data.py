import os
import random
import math

import torch
from tqdm import tqdm
from typing import Callable, List, Optional
from torch_geometric.data import Data, InMemoryDataset


def get_edge_index(prot_length, pos, cutoff):
    adjecency_matrix = torch.ones(prot_length, prot_length, dtype=int) - torch.eye(
        prot_length, dtype=int
    )
    idx_i, idx_j = torch.block_diag(adjecency_matrix).nonzero().t()

    # Relative_positions pos_ij = pos_j - pos_i and distances
    rel_pos = pos[idx_j] - pos[idx_i]
    rel_dist = torch.linalg.vector_norm(rel_pos, dim=1)

    # Keep only edges shorter than the cutoff
    short_edges = rel_dist < cutoff
    rel_dist = rel_dist[short_edges]

    idx_i = idx_i[short_edges]
    idx_j = idx_j[short_edges]

    return idx_i, idx_j


class SyntheticData(InMemoryDataset):

    def __init__(
        self,
        root: str,
        sampler: str = "baseline",
        sampling_prob: float = 0.5,
        max_protein_size: int = 100_000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        data_path: str = "raw",
        cutoff: float = 5.0,
    ) -> None:
        self.data_path = os.path.join(root, data_path)
        self.max_protein_size = max_protein_size
        self.cutoff_dist = cutoff
        self.sampler = sampler
        self.sampling_prob = sampling_prob
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["output.xml"]

    @property
    def processed_file_names(self) -> str:
        return "data_v0.pt"

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[torch.Tensor]:
        """
        Only for compatibility purposes
        """
        return None

    def download(self) -> None:
        pass

    def process(self):
        data_list = list()
        for i in tqdm(range(1_000)):
            prot_length = random.randint(a=50, b=self.max_protein_size)
            pos = torch.rand(prot_length, 3) * 30
            name = f"{prot_length}_{i}"

            z = torch.randint(low=1, high=9, size=(1, prot_length)).squeeze()

            y = torch.tensor([torch.sum(z.float())])

            idx_i, idx_j = get_edge_index(prot_length, pos, self.cutoff_dist)

            node_idx = torch.cat((idx_i, idx_j), dim=0).unique()
            z = z[node_idx]
            pos = pos[node_idx]

            if self.sampler == "baseline":
                prot_length = len(z)

            elif self.sampler == "random":
                mask = torch.rand(z.shape) > self.sampling_prob

                z = z[mask]
                pos = pos[mask]
                prot_length = len(z)

            elif self.sampler == "density":
                idx_i, idx_j = get_edge_index(len(z), pos, self.cutoff_dist)
                edge_counts = torch.bincount(idx_i)
                sampling_quantile = torch.quantile(
                    edge_counts.type(torch.float), self.sampling_prob
                )
                mask = edge_counts > sampling_quantile

                z = z[mask]
                pos = pos[mask]
                prot_length = len(z)

            idx_i, idx_j = get_edge_index(prot_length, pos, self.cutoff_dist)

            node_idx = torch.cat((idx_i, idx_j), dim=0).unique()
            z = z[node_idx]
            pos = pos[node_idx]

            if len(z) < 1:
                continue

            data = Data(
                z=z,
                pos=pos,
                y=y.unsqueeze(0),
                name=name,
                idx=i,
            )
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
