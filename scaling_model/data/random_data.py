import os
import random
import math

import torch
from tqdm import tqdm
from typing import Callable, List, Optional
from torch_geometric.data import Data, InMemoryDataset


class SyntheticData(InMemoryDataset):

    def __init__(
        self,
        root: str,
        max_protein_size: int = 100_000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        data_path: str = "raw",
    ) -> None:
        self.data_path = os.path.join(root, data_path)
        self.max_protein_size = max_protein_size
        self.cutoff_dist = 3.0
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
        for i in tqdm(range(10_000)):
            prot_length = random.randint(a=50, b=self.max_protein_size)
            pos = torch.rand(prot_length, 3) * 30
            name = f"{prot_length}_{i}"

            z = torch.randint(low=1, high=9, size=(1, prot_length)).squeeze()
            # y = torch.tensor([1 / (1 + math.exp(torch.mean(-z.float())))])

            y = torch.tensor([torch.sum(z.float())])

            adjecency_matrix = torch.ones(
                prot_length, prot_length, dtype=int
            ) - torch.eye(prot_length, dtype=int)
            idx_i, idx_j = torch.block_diag(adjecency_matrix).nonzero().t()

            # Relative_positions pos_ij = pos_j - pos_i and distances
            rel_pos = pos[idx_j] - pos[idx_i]
            rel_dist = torch.linalg.vector_norm(rel_pos, dim=1)

            # Keep only edges shorter than the cutoff
            short_edges = rel_dist < self.cutoff_dist
            rel_dist = rel_dist[short_edges]

            edge_index = torch.stack([idx_i[short_edges], idx_j[short_edges]])

            data = Data(
                z=z,
                pos=pos,
                y=y.unsqueeze(0),
                name=name,
                idx=i,
                edge_index=edge_index,
                num_nodes=prot_length,
            )
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
