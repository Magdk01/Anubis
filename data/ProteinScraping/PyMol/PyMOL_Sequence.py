#!/usr/bin/env python

# Run PyMOL with no GUI
import __main__
__main__.pymol_argv = [ 'pymol', '-q' ]

import pymol
import sys, time, os
import pandas as pd

from pymol import cmd

pymol.finish_launching()

# Set path for loaded PDBs. 
cmd.set('fetch_path', cmd.exp_path(r"C:\Users\shaji\test\experiment"), quiet=0)


# Output file
output = open("Sequence_33.txt", "w")
output.write("PDB" + "\t" + "Sequence")

#Load PDB entries 
filename = 'Datasets_33.txt'
df = pd.read_csv(filename, sep="\t", header=0)

# Change PDB entries to list
PDBs = list(df['pdb_id'])

for pdb in PDBs:

	# Load protein
	cmd.fetch(pdb)

	# Remove hetatm, all but chain A, and alternative locations
	cmd.remove('hetatm')
	cmd.remove('not (chain A or segment A)')
	cmd.do('''removealt''')

	# Get FASTA sequence from PDB 
	fasta = cmd.get_fastastr('all', quiet=1)
	output.write(fasta)

	cmd.do('''delete all''')

output.close()
cmd.quit()