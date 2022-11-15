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

def xray_scatter(coord, ele, q, use_oa=False, num_q_raster=1024, 
                 num_raster=512, sol_s=1.8, r_m=1.62, rho=0.334, c1=1.00, c2=2.00):
    
    assert len(coord) == 3 * len(ele)
    num_atom = len(ele)
    num_coord = 3 * num_atom
    num_q = len(q)
    #print(f'num atom: {num_atom}, num_coord: {num_coord}, num q: {num_q}')
    #c_coord = (c_float * num_coord).from_buffer(coord)
    #c_ele = (c_int * num_atom).from_buffer(ele)
    #c_q = (c_float * num_q).from_buffer(q)
    c_coord = (c_float * num_coord)(*coord)
    c_ele = (c_int * num_atom)(*ele)
    c_q = (c_float * num_q)(*q)
    c_S_calc = (c_float * num_q)()

    xs_calc(c_int(num_atom), c_coord, c_ele, c_int(num_q), c_q, c_S_calc, 
            c_int(num_raster), c_float(sol_s), c_float(r_m), c_float(rho), c_float(c1), c_float(c2),
            c_int(use_oa), c_int(num_q_raster))
    return c_S_calc[:]
