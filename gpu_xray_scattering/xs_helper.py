import ctypes
from ctypes import *
import sys, os

lib_path = os.path.dirname(os.path.realpath(__file__)) + '/bin/XS.so' 

try:
    xs = CDLL(lib_path)
#    print('XS loaded successfully')
except Exception as e:
    print('CDLL failed')
    print(e)
    exit()

xs_calc = xs.xray_scattering
xs_calc.restype = None

lib_path = os.path.dirname(os.path.realpath(__file__)) + '/bin/XXS.so' 

try:
    xxs = CDLL(lib_path)
#    print('XS loaded successfully')
except Exception as e:
    print('CDLL failed')
    print(e)
    exit()

xxs_calc = xxs.cross_xray_scattering
xxs_calc.restype = None

def xray_scatter(coord, ele, q, 
                 use_oa=0, num_q_raster=1024):
    
    assert len(coord) == 3 * len(ele)
    num_atom = len(ele)
    num_coord = 3 * num_atom
    num_q = len(q)
    c_coord = (c_float * num_coord)(*coord)
    c_ele = (c_int * num_atom)(*ele)
    c_q = (c_float * num_q)(*q)
    c_S_calc = (c_float * num_q)()

    xs_calc(c_int(num_atom), c_coord, c_ele, c_int(num_q), c_q, c_S_calc, 
            c_int(use_oa), c_int(num_q_raster))

    return c_S_calc[:]

def cross_xray_scatter(coord1, coord2, ele1, ele2, weight1, weight2, q, num_q_raster=1024):
    
    assert len(coord1) == 3 * len(ele1) and len(coord2) == 3 * len(ele2)
    num_atom1 = len(ele1)
    num_coord1 = 3 * num_atom1
    num_atom2 = len(ele2)
    num_coord2 = 3 * num_atom2
    num_q = len(q)
    c_coord1 = (c_float * num_coord1)(*coord1)
    c_ele1 = (c_int * num_atom1)(*ele1)
    c_coord2 = (c_float * num_coord2)(*coord2)
    c_ele2 = (c_int * num_atom2)(*ele2)
    c_weight1 = (c_float * num_atom1)(*weight1)
    c_weight2 = (c_float * num_atom2)(*weight2)
    c_q = (c_float * num_q)(*q)
    c_S_calc1 = (c_float * num_q)()
    c_S_calc2 = (c_float * num_q)()
    c_S_calc12 = (c_float * num_q)()

    xxs_calc(c_int(num_atom1), c_coord1, c_int(num_atom2), c_coord2, c_ele1, c_ele2, 
             c_weight1, c_weight2,
             c_int(num_q), c_q, 
             c_S_calc1, c_S_calc2, c_S_calc12,
             c_int(num_q_raster))

    return c_S_calc1[:], c_S_calc2[:], c_S_calc12[:]
