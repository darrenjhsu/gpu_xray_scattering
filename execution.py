
from square_helper import do_square_using_c
import numpy as np
from array import array
import time

my_list = [i for i in range(10000000)] #np.arange(1e4)
my_list_a = array('d',my_list)
my_list_n = np.array(my_list)

t0 = time.time()
squared_list_a = do_square_using_c(my_list_a)
t1 = time.time()

squared_list = np.square(my_list_n)
t2 = time.time()
print(f'C = {(t1-t0)*1000:.3f} ms, np = {(t2-t1)*1000:.3f} ms')

print(squared_list[:10], squared_list_a[:10])
