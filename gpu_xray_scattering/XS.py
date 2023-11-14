
import numpy as np
import time
from .xs_helper import xray_scatter, cross_xray_scatter 
from array import array
from .Molecule import Molecule

class Scatter:
    def __init__(self, q=np.linspace(0, 1, 200), num_q_raster=1024, use_oa=1, centering=True):
        self.q = q
        self.use_oa = use_oa
        self.num_q_raster = num_q_raster
        self.centering = centering

    def scatter(self, protein=None, ligand=None, rho=0.334, timing=False):
        if protein is None and ligand is None:
            print("No input, return None")
            return None
        # prepare coordinates from protein and optionally ligand
        coords = np.empty((0, 3))
        ele = np.empty(0)
        vdW = np.empty(0)
        if protein is not None:
            coords = protein.coordinates
            ele = np.concatenate([ele, protein.electrons - 1])
            vdW = np.concatenate([vdW, protein.radius])
        if ligand is not None:
            # do things with ligand
            coords = np.vstack([coords, ligand.coordinatess])
            ele = np.concatenate([ele, ligand.electrons - 1])
            vdW = np.concatenate([vdW, ligand.radius])
        if self.centering:
            coords = coords - coords.mean(0)            
        
        # array-ize np arrays
        coords_a = array('f', coords.flatten())
        ele_a = array('I', ele.astype(int))
        vdW_a = array('f', vdW.astype(float))
        q_a = array('f', self.q)
        
        # TODO: Add input vdW determined by denss
        t0 = time.time()
        S_calc = xray_scatter(coords_a, ele_a, vdW_a, q_a, 
                              use_oa=self.use_oa, num_q_raster=self.num_q_raster, rho=rho)
        t1 = time.time()
        
        if timing:
            print(f'Elapsed time = {(t1-t0)*1000:.3f} ms')
        print(np.array(S_calc).shape)
        return np.array(S_calc)

    def cross_scatter(self, protein=None, prior=np.empty((0, 3)), weight_p=np.empty(0), trial=None, weight_t=None, rho=0.334, timing=False):
   
        # protein is a Molecule object, with a molecule of M atoms and their element information
        # prior is a numpy array of N*3 coordinate or a Molecule object
        # weight is a N vector, denoting the weights of each prior point
        # trial is a numpy array of T*3 coordinate or a Molecule object
        # If prior or trial is numpy array:
        #   prior and trial are treated as pseudo-carbon atoms (using its form factor) - can change
        #   prior is "fraction of carbon atoms" and trial is always "one carbon atom"
        # if prior or trial is Molecule:
        #   prior and trial are treated as regular molecules 
        #   their elements are passed to scattering calculation
        # Downstream math figures out how to scale trial to fit data, and add/subtract the prior weights
        # Returns S_calc_pro, S_calc_trial, S_calc_cross, where cross is NOT MULTIPLIED BY 2

        if protein is None or prior is None or trial is None:
            print("No input, return None")
            return None

        # Concatenate protein and prior coordinates
        if isinstance(prior, Molecule):
            coords1 = np.vstack([protein.coordinates, prior.coordinates])
            ele1 = np.concatenate([protein.electrons - 1, prior.electrons - 1])
            vdW1 = np.concatenate([protein.radius, prior.radius])
        else:
            coords1 = np.vstack([protein.coordinates, prior])
            ele1 = np.concatenate([protein.electrons - 1, np.ones(len(prior)) * 5])
            vdW1 = np.concatenate([protein.radius, np.ones(len(prior)) * 1.4])
        # prior points are treated as carbons

        if isinstance(trial, Molecule):
            coords2 = trial.coordinates
            ele2 = trial.electrons - 1
            vdW2 = np.ones(len(ele2)) * 1.4
        else:
            coords2 = trial
            ele2 = np.ones(len(trial)) * 5
            vdW2 = np.ones(len(ele2)) * 1.4
        # trial points are treated as carbons as well
        
        # array-ize np arrays
        coords1_a = array('f', coords1.flatten())
        coords2_a = array('f', coords2.flatten())
        # protein has full weight for all its atoms, while the prior has fraction weights as input from user
        weight1_a = array('f', np.concatenate([np.ones(len(protein.coordinates)), weight_p]))
        if weight_t is not None:
            weight2_a = array('f', weight_t)
        elif isinstance(trial, Molecule):
            weight2_a = array('f', np.ones(len(trial.electrons)))
        else:
            weight2_a = array('f', np.ones(len(trial)))
        ele1_a = array('I', ele1.astype(int))
        ele2_a = array('I', ele2.astype(int))
        vdW1_a = array('f', vdW1.astype(float))
        vdW2_a = array('f', vdW2.astype(float))
        q_a = array('f', self.q)

        # TODO: Add input vdW determined by denss
        t0 = time.time()
        S_calc = cross_xray_scatter(coords1_a, coords2_a, ele1_a, ele2_a, vdW1_a, vdW2_a, weight1_a, weight2_a, q_a, 
                                    num_q_raster=self.num_q_raster, rho=rho)
        t1 = time.time()
        
        if timing:
            print(f'Elapsed time = {(t1-t0)*1000:.3f} ms')

        return np.array(S_calc[0]), np.array(S_calc[1]), np.array(S_calc[2])

