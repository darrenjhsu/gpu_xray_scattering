
import numpy as np
import time
from .xs_helper import xray_scatter, cross_xray_scatter 
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

    def cross_scatter(self, protein=None, prior=None, weight=None, trial=None, timing=False):
   
        # protein is a Molecule object, with a molecule of M atoms and their element information
        # prior is a numpy array of N*3 coordinate
        # weight is a N vector, denoting the weights of each prior point
        # trial is a numpy array of T*3 coordinate
        # prior and trial are treated as pseudo-carbon atoms (using its form factor) - can change
        # prior is "fraction of carbon atoms" and trial is always "one carbon atom"
        # Downstream math figures out how to scale trial to fit data, and add/subtract the prior weights

        if protein is None or prior is None or trial is None:
            print("No input, return None")
            return None

        # Concatenate protein and prior coordinates
        coords1 = np.vstack([protein.coordinates, prior.coordinates])
        # prior points are treated as carbons
        ele1 = np.concatenate([protein.electrons - 1, np.ones(len(prior)) * 5])

        coords2 = trial.coordinates
        # trial points are treated as carbons as well
        ele2 = np.ones(len(trial)) * 5
        
        # array-ize np arrays
        coords1_a = array('f', coords1.flatten())
        coords2_a = array('f', coords2.flatten())
        # protein has full weight for all its atoms, while the prior has fraction weights as input from user
        weight_a = array('f', np.concatenate([np.ones(len(protein.coordinates)), weight]))
        ele1_a = array('I', ele1.astype(int))
        ele2_a = array('I', ele2.astype(int))
        q_a = array('f', self.q)

        t0 = time.time()
        S_calc = cross_xray_scatter(coords1_a, coord2_a, ele1_a, ele2_a, weight_a, q_a, 
                                    num_q_raster=self.num_q_raster)
        t1 = time.time()
        
        if timing:
            print(f'Elapsed time = {(t1-t0)*1000:.3f} ms')

        return S_calc
