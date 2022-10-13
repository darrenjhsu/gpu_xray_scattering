import ctypes
from ctypes import c_double, c_int, CDLL
import sys

lib_path = '/ccs/home/djh992/GitHub/gpu_xray_scattering/src/test.so' 

try:
    xs = CDLL(lib_path)
except:
    print('CDLL failed')

xs_calc = xs.xray_scattering
xs_calc.restype = None

def do_square_using_c(list_in):
    """Call C function to calculate squares"""
    n = len(list_in)
    c_arr_in = (c_double * n).from_buffer(list_in)
    c_arr_out = (c_double * n)()

    python_c_square(c_int(n), c_arr_in, c_arr_out)
    return c_arr_out[:]
