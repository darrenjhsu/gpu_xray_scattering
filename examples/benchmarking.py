
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

scatter = XS.Scatter()
S_calc = scatter.scatter(pro, timing=True)

for i in np.unique(np.logspace(0, 2.7, 50, dtype=int)):
    pro_coord_stack = np.empty((0, 3))
    for j in range(i):
        pro_coord_stack = np.concatenate([pro_coord_stack, pro_coord + np.array([100, 0, 0]) * i])
    pro_ele_stack = np.repeat(pro_ele, i)
    pro = Molecule(coordinates=pro_coord_stack, elements=pro_ele_stack)
    print(f'Num of atoms: {len(pro_ele_stack)}')
    S_calc = scatter.scatter(pro, timing=True) # Second time is much faster

