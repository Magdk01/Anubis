import os

from typing import Callable, List, Optional

import torch
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset


from periodictable import elements
from tqdm import tqdm


def check_duplicate_coordinates(coordinates):
    """
    Checks if any two atoms in the list share the same coordinates.
    Returns True if duplicates are found, else False.
    """
    unique_coordinates = set(coordinates)
    if len(unique_coordinates) < len(coordinates):
        return True
    else:
        return False


class ProteinData(InMemoryDataset):

    def __init__(
        self,
        root: str,
        sampler: str = "baseline",
        max_protein_size: int = torch.inf,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        data_path: str = "raw",
    ) -> None:
        self.data_path = os.path.join(root, data_path)
        self.max_protein_size = max_protein_size
        self.cutoff_dist = 3.0
        self.sampler = sampler
        self.sampling_prob = 0.5
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
        target_cols = ["Alpha", "Beta"]

        df = pd.read_csv(
            "data/raw/cleaned_data.csv", index_col=False
        )
        if "Alpha" in target_cols:
            df = df.drop(df[df["Alpha"] == 0].index)
            df = df.drop(df[df["Alpha"] == 100].index)

        df[target_cols] = (df[target_cols] - df[target_cols].mean()) / df[
            target_cols
        ].std()
        element_translation = {el.symbol.lower(): el.number for el in elements}
        element_translation["d"] = 1
        data_list = list()
        for j, row in tqdm(df.iterrows(), total=len(df)):
            coords = eval(row.coords)

            name = row["PDB"]
            y = torch.tensor([row[col] for col in target_cols])
            x = torch.tensor([0.0] * 11, dtype=float)

            prot_length = len(coords)
            if prot_length > self.max_protein_size:
                continue

            z = torch.empty((prot_length), dtype=torch.long)
            pos = torch.empty((prot_length, 3))
            for i, x in enumerate(coords):
                z[i] = torch.tensor(
                    int(element_translation[x[0].lower()]), dtype=torch.long
                ).view(-1, 1)
                pos[i] = torch.tensor([x[1], x[2], x[3]])

            if self.sampler == "random":
                mask = torch.rand(z.shape) < self.sampling_prob

                z = z[mask]
                pos = pos[mask]
                prot_length = len(z)

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

            idx_i = idx_i[short_edges]
            idx_j = idx_j[short_edges]

            edge_index = torch.stack([idx_i, idx_j])

            node_idx = torch.cat((idx_i, idx_j), dim=0).unique()
            z = z[node_idx]
            pos = pos[node_idx]
            prot_length = len(z)

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
