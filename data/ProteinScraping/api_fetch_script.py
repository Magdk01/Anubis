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

scraped_csv_path = "scraped_proteins_dataset_80_current.csv"

def build_xml(offset=0,size=200):
    requestURL = f"https://www.ebi.ac.uk/proteins/api/proteins?offset={offset}&size={size}&reviewed=true&isoform=0&seqLength=1-80"

    r = requests.get(requestURL, headers={ "Accept" : "application/xml"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    responseBody = r.text
    
    file_path = source_path+'output.xml'
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(responseBody)


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
            return None
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

                    # Extract the desired information
                    atom_number = int(columns[1])
                    atom_type = columns[2]
                    x = float(columns[11])
                    y = float(columns[12])
                    z = float(columns[13])

                    # Append a dictionary with the extracted info to our list
                    if not re.search("[a-zA-Z]",atom_type):
                        print(pdb_id)
                        return None
                    atom_info.append((atom_type,x,y,z))
    except BaseException as err:
        print(pdb_id)
        print(err)
        return None

    if len(atom_info) >2000 or check_duplicate_coordinates(atom_info):
        return None
    return atom_info


def read_xml(filepath = source_path+"output.xml"):
    with open(filepath, 'r', encoding='utf-8') as file:
        xml_data_str = file.read()
        
    xml_data_bytes = xml_data_str.encode('utf-8')
    root = etree.fromstring(xml_data_bytes)
    
    namespaces = {
        'uniprot': "http://uniprot.org/uniprot",
        'xsi': "http://www.w3.org/2001/XMLSchema-instance"
    }
    
    # for protein_name in root.findall('.//uniprot:protein/uniprot:recommendedName/uniprot:fullName', namespaces):
    #     print(protein_name.text)
    
    df = pd.DataFrame(columns=['pdb','accession','name','sequence','coords',
                               #'mw','pI','II','aromaticity','gravy'
                               ])
    
    for entry in root.findall('uniprot:entry', namespaces):
        sequence = entry.find('uniprot:sequence', namespaces).text
        # print(sequence)
        accession = entry.find('uniprot:accession', namespaces).text
        # print(f"Accession: {accession}")
        name = entry.find('uniprot:name', namespaces).text
        # print(f"Name: {name}")
        pdb_ids = entry.findall(".//uniprot:dbReference[@type='PDB']", namespaces)
        for pdb_id in pdb_ids:
            # print(f"PDB ID: {pdb_id.get('id')}")
            df.loc[len(df.index)] = [pdb_id.get('id'),accession,name,sequence,None]
    df = df.drop_duplicates()

    return df


df = pd.DataFrame()
for i in tqdm(range(500)):
    build_xml(0+i*200,200)
    df = pd.concat([df,read_xml()])
df

df = df.drop_duplicates("pdb")
try:
    current_pymol_results = pd.read_table(source_path+"PyMol/"+pymol_result_file)
    not_done_in_pymol = list((set(df.pdb)-set(current_pymol_results.PDB)))

except:
    print("PyMol results file not found")
    not_done_in_pymol = list(set(df.pdb))



# Determine the number of batches
batch_size = 100
num_batches = (len(not_done_in_pymol) + batch_size - 1) // batch_size

for batch in range(num_batches):
    output = open(source_path+"PyMol/pdb_ids.txt", "w")
    # Calculate start and end indices for each batch
    start_index = batch * batch_size
    end_index = min((batch + 1) * batch_size, len(not_done_in_pymol))

    # Process each id in the current batch
    for id in not_done_in_pymol[start_index:end_index]:
        output.write(f"{id}\n")
    output.close()

    os.system(f'pymol {source_path}PyMol/PyMOL_Analysis.py')



res_df = pd.read_table(source_path + "PyMol/" + pymol_result_file)

res_df["coords"] = None

coords_list = []



# Iterate over each row in the DataFrame
for index, row in tqdm(res_df.iterrows(),total=len(res_df)):
    # Apply the get_atomic_structure function to the lowercase PDB value of the current row
    # print(row['PDB'])
    coords = get_atomic_structure(str(row['PDB']).lower())
    # Append the result to the coords_list
    coords_list.append(coords)



# Assign the list of coordinates to the 'coords' column of the DataFrame
res_df['coords'] = coords_list

output_df = res_df[~pd.isna(res_df["coords"])]
output_df = output_df[~pd.isna(output_df.Alpha)]
output_df = output_df[~pd.isna(output_df["Salt bridges"])]
output_df = output_df[~pd.isna(output_df["H-bonds"])]
# df = df.drop(df[df['pdb']=="pdb"].index)
output_df.to_csv(source_path+scraped_csv_path)