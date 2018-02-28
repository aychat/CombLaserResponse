__doc__ = """
Python wrapper for eval_pol3.c

Note: You must compile the C shared library
       gcc -O3 -shared -o eval_pol3.so eval_pol3.c -lm -fopenmp
"""
import os
import ctypes
from ctypes import c_double, c_int, POINTER, Structure


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [('real', c_double), ('imag', c_double)]

try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/eval_pol3.so")
except OSError:
    raise NotImplementedError(
        """The library is absent. You must compile the C shared library using the commands:
              gcc -O3 -shared -o eval_pol3.so eval_pol3.c -lm -fopenmp
        """
    )

############################################################################################
#
#   Declaring the function pol3_a2
#
############################################################################################

lib.pol3.argtypes = (
    POINTER(c_complex), # cmplx* out, # Array to save the polarizability
    POINTER(c_double),  # double* freq, frequency arrays
    c_int,      # const int freq_size,
    c_int,      # const int comb_size,
    c_double,   # const double delta_freq,
    c_double,   # const double gamma,
    c_double,   # const double M_field1,
    c_double,   # const double M_field2,
    c_double,   # const double M_field3, // Comb parameters
    c_complex,  # const cmplx wg_nv,
    c_complex,  # const cmplx wg_mv,
    c_complex   # const cmplx wg_vl // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
lib.pol3.restype = None


def pol3(out, freq, params, M_field1, M_field2, M_field3, wg_nv, wg_mv, wg_vl):
    return lib.pol3(
        out.ctypes.data_as(POINTER(c_complex)),
        freq.ctypes.data_as(POINTER(c_double)),
        freq.size,
        params.comb_size,
        params.delta_freq,
        params.gamma,
        M_field1,
        M_field2,
        M_field3,
        c_complex(wg_nv.real, wg_nv.imag),
        c_complex(wg_mv.real, wg_mv.imag),
        c_complex(wg_vl.real, wg_vl.imag)
    )
