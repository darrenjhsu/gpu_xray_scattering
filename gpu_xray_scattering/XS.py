
import numpy as np
import time
from .xs_helper import xray_scatter 
from array import array

class Scatter:
    def __init__(self, q=np.linspace(0, 1, 200), num_q_raster=1024, use_oa=1, centering=True):
        self.q = q
        self.use_oa = use_oa
        self.num_q_raster = num_q_raster
        self.centering = centering

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
        if self.centering:
            coords = coords - coords.mean(0)            
        
        # array-ize np arrays
        coords_a = array('f', coords.flatten())
        ele_a = array('I', ele.astype(int))
        q_a = array('f', self.q)

        t0 = time.time()
        S_calc = xray_scatter(coords_a, ele_a, q_a, 
                              use_oa=self.use_oa, num_q_raster=self.num_q_raster)
        t1 = time.time()
        
        if timing:
            print(f'Elapsed time = {(t1-t0)*1000:.3f} ms')

        return S_calc

