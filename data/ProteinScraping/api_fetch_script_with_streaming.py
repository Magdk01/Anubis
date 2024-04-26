import pandas as pd
from tqdm import tqdm
import os
from joblib import Memory
import re

# Define paths and file names
source_path = "data/ProteinScraping/"
pymol_result_file = "Pymol_results_current.txt"
scraped_csv_path = "scraped_proteins_dataset_PDB_new_1000_streaming.csv"

# Setup caching with joblib
cachedir = './joblib_cache'
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir, verbose=0)

def check_duplicate_coordinates(atom_info):
    """
    Checks for duplicate coordinates in the atom info list.
    """
    coordinates = set()
    for atom in atom_info:
        if atom in coordinates:
            return True
        coordinates.add(atom)
    return False

@memory.cache
def get_atomic_structure(pdb_id):
    """
    Fetches atomic structure from a local CIF file, avoiding hydrogens and duplicate coordinates.
    """
    local_file_path = f"{source_path}PyMol/pdb_files_for_pymol/{pdb_id}.cif"
    atom_info = []

    try:
        with open(local_file_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    columns = line.split()
                    if columns[-1] != "1" or columns[2] == "H":
                        continue
                    x, y, z = map(float, columns[11:14])
                    atom_type = columns[2]
                    if re.search("[a-zA-Z]", atom_type):  # Verify that atom_type is valid
                        atom_info.append((atom_type, x, y, z))
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None, None

    if check_duplicate_coordinates(atom_info):
        return None, None
    return atom_info, len(atom_info)

def process_data_stream(input_df_path, output_path):
    """
    Processes each row in the input DataFrame, one at a time, and appends results to a CSV.
    Filters for non-empty 'Salt bridges', 'Alpha', and 'H-bonds'.
    """
    # Read the input DataFrame
    df = pd.read_table(input_df_path)
    df = df.drop_duplicates()

    # Prepare the output file
    if os.path.exists(output_path):
        output_df = pd.read_csv(output_path)
        processed_ids = set(output_df['PDB'].str.lower())
    else:
        processed_ids = set()
        df["coords"] = None
        df["atom_len"] = None
        df.iloc[0:0].to_csv(output_path, index=False)  # Write empty df with headers

    # Stream process each row
    for index, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = row['PDB'].lower()
        if pdb_id in processed_ids:
            continue  # Skip already processed IDs

        coords, atom_len = get_atomic_structure(pdb_id)
        if coords is not None and atom_len is not None and not pd.isna(row['Alpha']) and not pd.isna(row['Salt bridges']) and not pd.isna(row['H-bonds']):
            row['coords'] = coords
            row['atom_len'] = atom_len
            # Append row to CSV file
            with open(output_path, 'a') as f:
                row.to_frame().T.to_csv(f, header=f.tell()==0, index=False)


if __name__ == "__main__":
    input_df_path = os.path.join(source_path, "PyMol", pymol_result_file)
    output_path = os.path.join(source_path, scraped_csv_path)
    process_data_stream(input_df_path, output_path)
