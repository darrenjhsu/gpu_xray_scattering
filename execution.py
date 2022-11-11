
from xs_helper import xray_scatter 
import numpy as np
from array import array
import time

coords = np.array([[0.0,0.0,0.0],[2.0,1.0,1.0]]).flatten()
coords_a = array('f',coords)

q = np.linspace(0, 1, 201)
q_a = array('f',q)

ele_a = array('I',[5,5])

t0 = time.time()
for i in range(1, 10):
    print(f'x = {i}')
    coords_a = array('f',[0.0,0.0,0.0,i,0.0,0.0])
    S_calc = xray_scatter(coords_a, ele_a, q_a, c1=1.0, c2=2.0)
t1 = time.time()

print(f'C = {(t1-t0)*1000:.3f} ms')

print(S_calc)
