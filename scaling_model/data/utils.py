import os
import requests, sys
from lxml import etree
from typing import Callable, List, Optional

import torch
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from biopandas.pdb import PandasPdb
from periodictable import elements
from tqdm import tqdm

bad_ids = [
    #    "1hje"
]


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


def get_atomic_structure(pdb_id, max_protein_size):
    try:
        ppdb = PandasPdb().read_pdb(f"data/raw/pdb_files/{pdb_id}.pdb")
        # ppdb = PandasPdb().fetch_pdb(pdb_id)
        atom = ppdb.df["ATOM"]
        if len(atom) > max_protein_size or pdb_id in bad_ids:
            return None

        if check_duplicate_coordinates(
            list(zip(atom.x_coord, atom.y_coord, atom.z_coord))
        ):
            print(pdb_id)
            return None

        structure = list(
            zip(atom.element_symbol, atom.x_coord, atom.y_coord, atom.z_coord)
        )
        return structure

    except:
        # print(pdb_id)
        return None


def get_prot_analysis(sequence):
    analysis = ProteinAnalysis(sequence)
    mw = analysis.molecular_weight()
    pI = analysis.isoelectric_point()
    instability = analysis.instability_index()
    aromaticity = analysis.aromaticity()
    gravy = analysis.gravy()
    return mw, pI, instability, aromaticity, gravy


class ProteinData(InMemoryDataset):

    def __init__(
        self,
        root: str,
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
        chemesitry_excel = False
        target_cols = ["Alpha", "Beta"]

        if chemesitry_excel:

            df = pd.read_excel("data/raw/MonomericProteinsWithFeatures.xlsx")
            df[target_cols] = (df[target_cols] - df[target_cols].mean()) / df[
                target_cols
            ].std()
            element_translation = {el.symbol.lower(): el.number for el in elements}
            element_translation["d"] = 1
            data_list = list()
            for j, row in tqdm(df.iterrows(), total=len(df)):
                pdb_id = row.PDB
                coords = get_atomic_structure(pdb_id, self.max_protein_size)
                if coords == None:
                    continue

                name = row["PDB"]
                y = torch.tensor([row[col] for col in target_cols])
                x = torch.tensor(
                    [0.0] * 11, dtype=float
                )  # TODO: Figure out what this is

                n_atoms = len(coords)
                z = torch.empty((n_atoms), dtype=torch.long)
                pos = torch.empty((n_atoms, 3))
                for i, x in enumerate(coords):
                    z[i] = torch.tensor(
                        int(element_translation[x[0].lower()]), dtype=torch.long
                    ).view(-1, 1)
                    pos[i] = torch.tensor([x[1], x[2], x[3]])

                data = Data(x=x, z=z, pos=pos, y=y.unsqueeze(0), name=name, idx=j)
                data_list.append(data)

            self.save(data_list, self.processed_paths[0])
            
        else:
            df = pd.read_csv("data/raw/scraped_proteins_dataset_PDB_new_1000.csv")
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
                x = torch.tensor(
                    [0.0] * 11, dtype=float
                )  # TODO: Figure out what this is

                prot_length = len(coords)
                if prot_length != row.atom_len:
                    print('hehe')
                if prot_length>self.max_protein_size:
                    continue

                z = torch.empty((prot_length), dtype=torch.long)
                pos = torch.empty((prot_length, 3))
                for i, x in enumerate(coords):
                    z[i] = torch.tensor(
                        int(element_translation[x[0].lower()]), dtype=torch.long
                    ).view(-1, 1)
                    pos[i] = torch.tensor([x[1], x[2], x[3]])

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
