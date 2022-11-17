
import gpu_xray_scattering
from gpu_xray_scattering.xs_helper import xray_scatter 
import numpy as np
from array import array
import time

coords = np.array([[0.0,0.0,0.0],[2.0,1.0,1.0]]).flatten()
coords_a = array('f',coords)

q = np.linspace(0, 1, 41)
q_a = array('f',q)

ele_a = array('I',[5,5])

for i in [10]:
    print(f'x = {i}')
    coords_a = array('f',[0.0,0.0,0.0,i,0.0,0.0])
    S_calc1 = xray_scatter(coords_a, ele_a, q_a, c1=1.0, c2=2.0, use_oa=0)
    t0 = time.time()
    S_calc1 = xray_scatter(coords_a, ele_a, q_a, c1=1.0, c2=2.0, use_oa=0)
    t1 = time.time()
    print(f'C = {(t1-t0)*1000:.3f} ms')
    t0 = time.time()
    S_calc2 = xray_scatter(coords_a, ele_a, q_a, c1=1.0, c2=2.0, use_oa=1)
    t1 = time.time()
    print(f'C = {(t1-t0)*1000:.3f} ms')
    t0 = time.time()
    S_calc3 = xray_scatter(coords_a, ele_a, q_a, c1=1.0, c2=2.0, use_oa=1, num_q_raster=512)
    t1 = time.time()
    print(f'C = {(t1-t0)*1000:.3f} ms')
    t0 = time.time()
    S_calc4 = xray_scatter(coords_a, ele_a, q_a, c1=1.0, c2=2.0, use_oa=2, num_q_raster=1024)
    t1 = time.time()
    print(f'C = {(t1-t0)*1000:.3f} ms')


print(S_calc1)
print()
print(S_calc2)
print()
print(S_calc3)
print()
print(S_calc4)
