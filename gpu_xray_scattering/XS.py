
import numpy as np
import time
from .xs_helper import xray_scatter 
from array import array

class Scatter:
    def __init__(self, q=np.linspace(0, 1, 200), c1=1.0, c2=2.0, r_m=1.62, sol_s=1.8, num_raster=512, rho=0.334):
        self.q = q
        self.c1 = c1
        self.c2 = c2
        self.r_m = r_m
        self.sol_s = sol_s
        self.num_raster = num_raster
        self.rho = rho

    def scatter(self, protein=None, ligand=None, timing=False):
        if protein is None and ligand is None:
            print("No input, return None")
            return None
        # prepare coordinates from protein and optionally ligand
        coords = np.empty((0, 3))
        ele = np.empty(0)
        if protein is not None:
            coords = protein.coordinates
            ele = np.concatenate([ele, protein.electrons - 1])
        if ligand is not None:
            # do things with ligand
            coords = np.vstack([coords, ligand.coordinatess])
            ele = np.concatenate([ele, ligand.electrons - 1])
                    
        
        # array-ize np arrays
        coords_a = array('f', coords.flatten())
        ele_a = array('I', ele.astype(int))
        q_a = array('f', self.q)

        t0 = time.time()
        S_calc = xray_scatter(coords_a, ele_a, q_a, 
                              num_raster=self.num_raster, sol_s=self.sol_s, 
                              r_m=self.r_m, rho=self.rho, c1=self.c1, c2=self.c2)
        t1 = time.time()
        
        if timing:
            print(f'Elapsed time = {(t1-t0)*1000:.3f} ms')

        return S_calc

