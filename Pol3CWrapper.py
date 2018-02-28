__doc__ = """
Python wrapper for pol3_calculate.c

Note: You must compile the C shared library
       gcc -O3 -shared -o pol3_calculate.so pol3_calculate.c -lm -fopenmp
"""
import os
import ctypes
from ctypes import c_double, c_int, POINTER

try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/pol3_calculate.so")
except OSError:
    raise NotImplementedError(
        """The library is absent. You must compile the C shared library using the commands:
              gcc -O3 -shared -o pol3_calculate.so pol3_calculate.c -lm -fopenmp
        """
    )

############################################################################################
#
#
#
############################################################################################


# specify the parameters of the c-function
c_calc_pol3_a2 = lib.eval
c_calc_pol3_a2.argtypes = (
    POINTER(c_double),        # double* out_r
    POINTER(c_double),        # double* out_i
    c_int,                    # int size_out
    POINTER(c_int),           # int* index
    c_int,                           # int x_num
    c_double,                        # double x_min
    c_double,                        # double x_max
    c_double,                        # double h_min
    c_double,                        # double h_max
    POINTER(MoleculeParams),  # struct params
)
c_calc_pol3_a2.restype = c_int
