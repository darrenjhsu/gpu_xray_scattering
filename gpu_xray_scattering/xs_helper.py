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
