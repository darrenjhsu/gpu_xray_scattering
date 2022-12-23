
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
pro2 = Molecule(coordinates=np.concatenate([pro_coord, pro_coord+10]), elements=np.concatenate([pro_ele, pro_ele]))

scatter = XS.Scatter(c1=1, c2=2)
scatter_oa = XS.Scatter(c1=1, c2=2, use_oa=1)
S_calc = scatter.scatter(pro, timing=True)
#print(S_calc[:10])
S_calc = scatter.scatter(pro2, timing=True)
#print(S_calc[:10])

#for i in np.unique(np.logspace(1, 2.9, 50, dtype=int)):
#    pro_coord_stack = np.empty((0, 3))
#    for j in range(i):
#        pro_coord_stack = np.concatenate([pro_coord_stack, pro_coord + j*1])
#    pro_ele_stack = np.repeat(pro_ele, i)
#    pro = Molecule(coordinates=pro_coord_stack, elements=pro_ele_stack)
#    print(f'Num of atoms: {len(pro_ele_stack)}, electrons: {(pro.electrons).sum()}')
#    print('Orientational average 1')
#    S_calc_oa = np.array(scatter_oa.scatter(pro, timing=True))
#    print(S_calc_oa[:10])
#    #print(S_calc[:5], S_calc_oa[:5])

for i in np.unique(np.logspace(0, 2.7, 40, dtype=int)):
    pro_coord_stack = np.empty((0, 3))
    for j in range(i):
        pro_coord_stack = np.concatenate([pro_coord_stack, pro_coord + j*50])
    pro_ele_stack = np.repeat(pro_ele, i)
    pro = Molecule(coordinates=pro_coord_stack, elements=pro_ele_stack)
    print(f'Num of atoms: {len(pro_ele_stack)} ({i} copies), electrons: {(pro.electrons).sum()}')
    print('Vanilla')
    S_calc = np.array(scatter.scatter(pro, timing=True))
    print('Orientational average 1')
    S_calc_oa = np.array(scatter_oa.scatter(pro, timing=True))
    print(S_calc[:10])
    print(S_calc_oa[:10])
    #print(S_calc[:5], S_calc_oa[:5])
    rel_diff_max = np.max(np.abs((S_calc_oa - S_calc) / S_calc))
    rel_diff_mean = np.mean(np.abs((S_calc_oa - S_calc) / S_calc))
    ref_diff_max_idx = np.argmax(np.abs((S_calc_oa - S_calc) / S_calc))
    print(f'Relative difference: mean {rel_diff_mean:.3f}, max {rel_diff_max:.3f} at idx {ref_diff_max_idx}: vanilla is {S_calc[ref_diff_max_idx]} and oa is {S_calc_oa[ref_diff_max_idx]}')
    print()


