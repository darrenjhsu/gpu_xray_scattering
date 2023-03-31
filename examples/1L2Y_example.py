
import numpy as np
from gpu_xray_scattering import XS
from gpu_xray_scattering.Molecule import Molecule

def readPDB(fname): 
    # Just a temporary reader 
    # Use your favorite molecule/trajectory processor
    coords = []
    elements = []
    with open(fname, 'r') as f:
        cont = f.readlines()
    for line in cont:
        if 'ATOM' in line:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elements.append(line[76:78].strip())
    return np.array(coords), np.array(elements) 

pro_coord, pro_ele = readPDB('1L2Y.pdb') # Or generate numpy arrays yourself

# Molecule class takes numpy array for coordinates
# and any kind of lists as elements
pro = Molecule(coordinates=pro_coord, elements=pro_ele)

scatter = XS.Scatter(use_oa=0)
S_calc = scatter.scatter(pro, timing=True)
S_calc = scatter.scatter(pro, timing=True) # Second time is much faster

scatter_oa = XS.Scatter(use_oa=1)
S_calc_oa = scatter_oa.scatter(pro, timing=True)

scatter_oa2 = XS.Scatter(use_oa=2)
S_calc_oa2 = scatter_oa.scatter(pro, timing=True)

print(S_calc[:5])
print(S_calc_oa[:5])
print(S_calc_oa2[:5])
