#!/usr/bin/env python

# Run PyMOL with no GUI
import __main__
__main__.pymol_argv = [ 'pymol', '-q' ]

import pymol
import sys, time, os
import pandas as pd

import os
from pymol import cmd

pymol.finish_launching()



source_path = "data/ProteinScraping/"
pymol_result_file = "PyMol_results_max_80_seq.txt"


# Set path for loaded PDBs. 
cmd.set('fetch_path', cmd.exp_path(os.getcwd()+"/"+source_path + r"PyMol/pdb_files_for_pymol"), quiet=0)
# Output file
output = open(source_path + "PyMol/"+ pymol_result_file, "a")
output.write("PDB" + "\t" + "No. a.a." + "\t" + "Glycine" + "\t" + "S.S." + "\t" + "Long SS" + "\t" + "Charge" + "\t"+ "SASA" + "\t" + "No. pos." + "\t" + "No. Surf. pos." + "\t" + "Pos. area" + "\t" + "No. neg." + "\t" + "No. Surf. neg." + "\t" + "Neg. area" + "\t" + "No. hyd." + "\t" + "No. Surf. hyd." + "\t" + "hyd. area" + "\t" + "Alpha" + "\t" + "Beta" + "\t" + "Salt bridges" + "\t" + "H-bonds" + "\n")
 #Load PDB entries
filename = source_path + 'PyMol/pdb_ids.txt'

# Initialize an empty list to store PDB IDs
PDBs = []

# Open and read the file
with open(filename, 'r') as file:
	for line in file:
		# Strip whitespace and add to the list
		pdb_id = line.strip()
		PDBs.append(pdb_id)

# Load scripts
cmd.do(f'''run {source_path}PyMol/removealt.py''')
cmd.do(f'''run {source_path}PyMol/findSurfaceResidues.py''')
cmd.do(f'''run {source_path}PyMol/get_raw_distances.py''')

for pdb in PDBs: 
	
	# Load protein
	print(pdb)
	cmd.fetch(pdb)

	# Remove hetatm, all but chain A, and alternative locations
	cmd.remove('hetatm')
	cmd.remove('not (chain A or segment A)')
	cmd.do('''removealt''') 

	# Number of amino acids 
	no_aa = cmd.select('name ca') 
	no_aa = str(no_aa) 

	# Number of glycines 
	no_gly = cmd.select('resname GLY and name ca')
	no_gly = str(no_gly)

	# Calculate SASA
	cmd.do('''set dot_solvent, 1''')
	cmd.do('''set dot_density, 2''')
	sasa = cmd.get_area()
	sasa = str(sasa) 

	cmd.do('''findSurfaceResidues''')

	# Find positively charged residues 
	no_pos = cmd.select('resname ARG+LYS and name ca')
	surf_pos = cmd.select('(byres(exposed_res_01 and resname ARG+LYS)) and name ca')
	surf_pos = str(surf_pos)
	surf_pos_all = cmd.get_area('byres(exposed_res_01 and resname ARG+LYS)')
	surf_pos_area = str(surf_pos_all)

	# Find negatively charged residues 
	no_neg = cmd.select('resname GLU+ASP and name ca')
	surf_neg = cmd.select('(byres(exposed_res_01 and resname GLU+ASP)) and name ca')
	surf_neg = str(surf_neg)
	surf_neg_all = cmd.get_area('byres(exposed_res_01 and resname GLU+ASP)')
	surf_neg_area = str(surf_neg_all)

	# Find hydrophobic residues 
	no_hyd = cmd.select('resname ALA+VAL+ILE+LEU+MET+PHE+TYR+TRP and name ca')
	no_hyd = str(no_hyd) 
	surf_hyd = cmd.select('(byres(exposed_res_01 and resname ALA+VAL+ILE+LEU+MET+PHE+TYR+TRP)) and name ca')
	surf_hyd = str(surf_hyd)
	surf_hyd_all = cmd.get_area('byres(exposed_res_01 and resname ALA+VAL+ILE+LEU+MET+PHE+TYR+TRP)')
	surf_hyd_area = str(surf_hyd_all)

	# Charge
	no_pos = cmd.select('resname ARG+LYS and name ca')
	no_neg = cmd.select('resname GLU+ASP and name ca')
	charge = no_pos - no_neg
	charge = str(charge)
	no_neg = str(no_neg) 
	no_pos = str(no_pos) 

	# Number of SS
	no_ssx = cmd.select('CYS/SG and bound_to CYS/SG')
	no_ss = no_ssx/2
	no_ss = str(no_ss)


	# Number of long-range SS
	count = 0
	if no_ssx >= 2:
		cmd.do('''select long_ss, CYS/SG and bound_to CYS/SG''')
		cmd.do('''myspace={'l': []}''')
		cmd.iterate('long_ss', 'l.append(resv)', space=myspace)
		lss = myspace['l']
		cmd.do('''myspace={'d': []}''')

		count = 0
		for i in lss:

			t = []
			a = i

			cmd.select('d', '(not resid %s) and bound_to %s/SG' % (i, i))
			cmd.iterate('d', 't.append(resv)')

			if t:  # Check if t is not empty
				t_i = int("".join(map(str, t)))  # Safely convert elements to string and then to integer
				c = t_i - a

				if abs(c) >= 12:
					count += 1
				# No else needed; if t is empty or the condition is not met, we simply don't increment count

		count = count // 2  # Using integer division
		long_ss = str(count)
	else:
		long_ss = "N"
		
	# Secondary structure - percentage
	cmd.do('''s = [i.ss for i in cmd.get_model("n. ca").atom]''')
	if len(s) != 0:
		alp = 100.0*s.count("H")/len(s)
		bet  = 100.0*s.count("S")/len(s)
		alp = str(alp)
		bet  = str(bet)
	else:
		alp = "None"
		bet = "None"

	# Salt-bridges (https://sourceforge.net/p/pymol/mailman/message/29141454/)
	cmd.do('''select roi, chain A''')
	cmd.do('''select positive, resn ARG+LYS and not name N+O''')
	cmd.do('''select negative, resn GLU+ASP and not name N+O''')

	cmd.do('''set h_bond_cutoff_center, 5.0''')
	cmd.do('''set h_bond_cutoff_edge, 5.0''')

	cmd.delete('saltbridges')
	cmd.do('''distance saltbridges, roi and negative, roi and positive, mode=2''')
	cmd.hide('label')

	try:
		cmd.do('''get_raw_distances''')
		cmd.do('''sb = get_raw_distances('saltbridges')''')
		sb = len(sb)
		sb = str(sb)
	except:
		sb = "None"
	cmd.do('''set h_bond_cutoff_center, 3.6''')
	cmd.do('''set h_bond_cutoff_edge, 3.6''')

	try:
		# Hydrogen bonds - all
		cmd.do('''distance H_bonds, roi, roi, quiet=1, mode=2, label=0, reset=1''')
		cmd.do('''get_raw_distances''')
		cmd.do('''hb = get_raw_distances('H_bonds')''')
		hb = len(hb)
		hb = str(hb)
	except:
		hb = "None"
	output.write(pdb + "\t" + no_aa + "\t" + no_gly + "\t" + no_ss + "\t" + long_ss + "\t" + charge + "\t" + sasa + "\t" + no_pos + "\t" + surf_pos + "\t" + surf_pos_area + "\t" + no_neg + "\t" + surf_neg + "\t" + surf_neg_area + "\t" + no_hyd + "\t" + surf_hyd + "\t" + surf_hyd_area + "\t" + alp + "\t" + bet + "\t" + sb + "\t" + hb + "\n")
	

	cmd.do('''delete all''')

output.close()

cmd.quit()	