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
output = open("Results.txt", "w")

# Load PDB entries 
#filename = 'Datasets_test.txt'
#df = pd.read_csv(filename, sep="\t", header=0)

# Change PDB entries to list
#PDBs = "1ryo"

# Load scripts
cmd.do('''run removealt.py''')
cmd.do('''run findSurfaceResidues.py''')
cmd.do('''run get_raw_distances.py''')



# Remove hetatm, all but chain A, and alternative locations
cmd.remove('hetatm')
cmd.remove('not (chain A or segment A)')
cmd.do('''removealt''') 

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
	
		cmd.select('d', '(not resid %s) and bound_to %s/SG' %(i,i))
		cmd.iterate('d', 't.append(resv)')
		t = [str(integer) for integer in t]
		t_i = "".join(t)
		t_i = int(t_i)

		c = t_i - a
		print(t_i, a, c)
		
		if abs(c) >= 12:
			count = count + 1
		
		else: 
			count = count

	count = count/2
	long_ss = str(count)				

else:
	long_ss = "N"

output.write(no_ss + "\t" + long_ss + "\t" + "\n")
	

	#cmd.do('''delete all''')

output.close()

#cmd.quit()	