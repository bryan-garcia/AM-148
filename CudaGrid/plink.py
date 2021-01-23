import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import numpy.ctypeslib as npct

def get_f():

    # Dont forget to set me!
    SO_PATH = os.environ['SO_PATH']
    lib = ctypes.cdll.LoadLibrary(SO_PATH + '/lib.so')
    f = lib.bicubic_main
    f.restype = None

    return f

def Bicubic_Benchmark(rmin, rmax, zmin, zmax, nr_crude, nz_crude, nr_fine, nz_fine, option):

    bicubic = get_f()

    crude_data = np.zeros((nr_crude, nz_crude), dtype=np.float32)
    fine_data = np.zeros((nr_fine, nz_fine), dtype=np.float32)

    bicubic(ctypes.c_float(rmin), ctypes.c_float(rmax), ctypes.c_float(zmin), ctypes.c_float(zmax), 
        ctypes.c_int(nr_crude), ctypes.c_int(nz_crude), ctypes.c_int(nr_fine), ctypes.c_int(nz_fine),
        npct.as_ctypes(crude_data.ravel()), npct.as_ctypes(fine_data.ravel()), ctypes.c_int(option))

    plt.imsave("crude_data.png", crude_data)
    plt.imsave("fine.png", fine_data)

rmin = -1
rmax = 1
zmin = -1
zmax = 1

nr_crude = 10
nz_crude = 10

nr_fine = 100
nz_fine = 100

option = 1

Bicubic_Benchmark(rmin, rmax, zmin, zmax, nr_crude, nz_crude, nr_fine, nz_fine, option)