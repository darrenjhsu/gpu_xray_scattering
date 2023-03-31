
import numpy as np
from gpu_xray_scattering import XS
from gpu_xray_scattering.Molecule import Molecule
import glob

def readPDB(fname): 
    # Just a temporary reader 
    # Use your favorite molecule/trajectory processor
    coords = []
    elements = []
    with open(fname, 'r') as f:
        cont = f.readlines()
    for line in cont:
        if 'ATOM' in line[:10]:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elements.append(line[76:78].strip())
    return np.array(coords), np.array(elements) 



scatter = XS.Scatter(use_oa=0)
scatter_oa = XS.Scatter(use_oa=1)
pro_coord, pro_ele = readPDB('1L2Y.pdb') # Or generate numpy arrays yourself
pro = Molecule(coordinates=pro_coord, elements=pro_ele)
S_calc = scatter.scatter(pro, timing=True)

for prot in glob.glob('data/*.pdb'):
    
    pro_coord, pro_ele = readPDB(prot) # Or generate numpy arrays yourself
    print(f"Protein {prot} has {len(pro_ele)} atoms")
    pro = Molecule(coordinates=pro_coord, elements=pro_ele)
    S_calc = np.array(scatter.scatter(pro, timing=True))
    print(S_calc)
    print('Orientational average 1')
    S_calc_oa = np.array(scatter_oa.scatter(pro, timing=True))
    print(S_calc_oa)
    #print(S_calc[:5], S_calc_oa[:5])
    rel_diff_max = np.max(np.abs((S_calc_oa - S_calc) / S_calc))
    rel_diff_mean = np.mean(np.abs((S_calc_oa - S_calc) / S_calc))
    ref_diff_max_idx = np.argmax(np.abs((S_calc_oa - S_calc) / S_calc))
    print(f'Relative difference: mean {rel_diff_mean:.3f}, max {rel_diff_max:.3f} at idx {ref_diff_max_idx}: vanilla is {S_calc[ref_diff_max_idx]} and oa is {S_calc_oa[ref_diff_max_idx]}')
    print()


