import requests, sys
from lxml import etree
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from biopandas.pdb import PandasPdb
import torch
import numpy as np
from periodictable import elements
from torch_geometric.data import Data
import os
from tqdm import tqdm
from IPython.utils import io
import re

source_path = "data/ProteinScraping/"

pymol_result_file = "Pymol_results_current.txt"

scraped_csv_path = "scraped_proteins_dataset_PDB_new_1000.csv"



def check_duplicate_coordinates(atom_info):
    """
    Checks if any two atoms in the list share the same coordinates.
    Returns True if duplicates are found, else False.
    """
    coordinates = [(atom[1],atom[2],atom[3]) for atom in atom_info]
    unique_coordinates = set(coordinates)
    if len(unique_coordinates) < len(coordinates):
        return True
    else:
        return False


def get_atomic_structure(pdb_id):
    try:
        if pdb_id == 'pdb' or pdb_id == "6zke":
            return None, None
        # Define the path for the local file storage
        local_file_path = f"{source_path}PyMol/pdb_files_for_pymol/{pdb_id}.cif"
        atom_info = []

        # Open and read the file
        with open(local_file_path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Check if the line starts with 'ATOM'
                if line.startswith("ATOM"):
                    columns = line.split()
                    if not columns[-1] == "1":
                        continue
                    # Extract the desired information
                    atom_number = int(columns[1])
                    atom_type = columns[2]
                    if atom_type == "H":
                        continue
                    x = float(columns[11])
                    y = float(columns[12])
                    z = float(columns[13])

                    # Append a dictionary with the extracted info to our list
                    if not re.search("[a-zA-Z]",atom_type):
                        print(pdb_id)
                        return None, None
                    atom_info.append((atom_type,x,y,z))
    except BaseException as err:
        print(pdb_id)
        print(err)
        return None, None

    if len(atom_info)>1000 or check_duplicate_coordinates(atom_info):
        return None, None
    return atom_info, len(atom_info)





# df = pd.DataFrame()
# with open("data/ProteinScraping//prots_from_pdb.txt","r") as file:
#     data_list = [line.strip().split(',') for line in file][0]

# try:
#     current_pymol_results = pd.read_table(source_path+"PyMol/"+pymol_result_file)
#     not_done_in_pymol = list((set(data_list)-set(current_pymol_results.PDB)))

# except:
#     print("PyMol results file not found")
#     not_done_in_pymol = list(set(data_list))



# # Determine the number of batches
# batch_size = 100
# num_batches = (len(not_done_in_pymol) + batch_size - 1) // batch_size

# for batch in range(num_batches):
#     output = open(source_path+"PyMol/pdb_ids.txt", "w")
#     # Calculate start and end indices for each batch
#     start_index = batch * batch_size
#     end_index = min((batch + 1) * batch_size, len(not_done_in_pymol))

#     # Process each id in the current batch
#     for id in not_done_in_pymol[start_index:end_index]:
#         output.write(f"{id}\n")
#     output.close()

#     os.system(f'pymol {source_path}PyMol/PyMOL_Analysis.py')



res_df = pd.read_table(source_path + "PyMol/" + pymol_result_file)
res_df = res_df.drop_duplicates()
# res_df = res_df[res_df['PDB'].isin(data_list)]


res_df["coords"] = None
res_df["atom_len"] = None
coords_list = []
atom_len_list = []


# Iterate over each row in the DataFrame
for index, row in tqdm(res_df.iterrows(),total=len(res_df)):
    # Apply the get_atomic_structure function to the lowercase PDB value of the current row
    # print(row['PDB'])
    coords,atom_len = get_atomic_structure(str(row['PDB']).lower())
    # Append the result to the coords_list
    coords_list.append(coords)
    atom_len_list.append(atom_len)


# Assign the list of coordinates to the 'coords' column of the DataFrame
res_df['coords'] = coords_list
res_df['atom_len'] = atom_len_list

print('Len before reduce of DF')
print(len(res_df))
output_df = res_df[~pd.isna(res_df["coords"])]
print(len(output_df))

output_df = output_df[output_df["atom_len"]!=0]
print(len(output_df))

output_df = output_df[~pd.isna(output_df.Alpha)]
print(len(output_df))

output_df = output_df[~pd.isna(output_df["Salt bridges"])]
print(len(output_df))

output_df = output_df[~pd.isna(output_df["H-bonds"])]
print(len(output_df))

# df = df.drop(df[df['pdb']=="pdb"].index)
output_df.to_csv(source_path+scraped_csv_path)