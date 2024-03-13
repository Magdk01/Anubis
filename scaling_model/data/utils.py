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


def get_atomic_structure(pdb_id):
    ppdb = PandasPdb().fetch_pdb(pdb_id)
    atom = ppdb.df["ATOM"]
    structure = list(zip(atom.element_symbol, atom.x_coord, atom.y_coord, atom.z_coord))
    return structure


def get_prot_analysis(sequence):
    analysis = ProteinAnalysis(sequence)
    mw = analysis.molecular_weight()
    pI = analysis.isoelectric_point()
    instability = analysis.instability_index()
    aromaticity = analysis.aromaticity()
    gravy = analysis.gravy()
    return mw, pI, instability, aromaticity, gravy


class TestData(InMemoryDataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        data_path: str = "raw",
    ) -> None:
        self.data_path = os.path.join(root, data_path)
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
        requestURL = "https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size=100&reviewed=true&isoform=0"

        r = requests.get(requestURL, headers={"Accept": "application/xml"})

        if not r.ok:
            r.raise_for_status()
            sys.exit()

        responseBody = r.text

        file_path = os.path.join(self.data_path, "output.xml")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(responseBody)

    def process(self):
        file_path = os.path.join(self.data_path, "output.xml")
        with open(file_path, "r", encoding="utf-8") as file:
            xml_data_str = file.read()

        xml_data_bytes = xml_data_str.encode("utf-8")
        root = etree.fromstring(xml_data_bytes)

        namespaces = {
            "uniprot": "http://uniprot.org/uniprot",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }

        df = pd.DataFrame(
            columns=[
                "pdb",
                "accession",
                "name",
                "sequence",
                "coords",
                "mw",
                "pI",
                "II",
                "aromaticity",
                "gravy",
            ]
        )

        for entry in root.findall("uniprot:entry", namespaces):
            sequence = entry.find("uniprot:sequence", namespaces).text
            # print(sequence)
            accession = entry.find("uniprot:accession", namespaces).text
            # print(f"Accession: {accession}")
            name = entry.find("uniprot:name", namespaces).text
            # print(f"Name: {name}")
            pdb_ids = entry.findall(".//uniprot:dbReference[@type='PDB']", namespaces)
            for pdb_id in pdb_ids:
                # print(f"PDB ID: {pdb_id.get('id')}")
                df.loc[len(df.index)] = [
                    pdb_id.get("id"),
                    accession,
                    name,
                    sequence,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
        df = df.drop_duplicates()

        for _, row in df.iterrows():
            pdb_id = row.pdb
            row["coords"] = get_atomic_structure(row.pdb)
            row["mw"], row["pI"], row["II"], row["aromaticity"], row["gravy"] = (
                get_prot_analysis(row.sequence)
            )
        element_translation = {el.symbol.lower(): el.number for el in elements}

        data_list = list()
        for j, row in df.iterrows():
            name = row["pdb"]
            y = torch.Tensor(
                [row["gravy"], row["mw"], row["pI"], row["II"], row["aromaticity"]]
            )
            x = torch.Tensor([0.0] * 11, dtype=float)  # TODO: Figure out what this is

            n_atoms = len(row["coords"])
            z = torch.empty((n_atoms), dtype=torch.long)
            pos = torch.empty((n_atoms, 3))
            for i, x in enumerate(row["coords"]):
                z[i] = torch.Tensor(
                    [int(element_translation[x[0].lower()])], dtype=torch.long
                )
                pos[i] = torch.Tensor([x[1], x[2], x[3]])

            data = Data(x=x, z=z, pos=pos, y=y, name=name, idx=j)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
