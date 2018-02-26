from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from types import MethodType, FunctionType
import numexpr as ne
from itertools import permutations
import time


class PolarizationTerms:
    """
    Calculates NL Polarizations for an ensemble of near identical molecules.
    """
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)
        self.molecules = [
            dict(
                w_10=self.w_excited_1 + _ * self.w_spacing_10 * 10**(-self.N_order_energy),
                w_20=self.w_excited_1 + self.w_excited_2 + _ * self.w_spacing_20 * 10**(-self.N_order_energy),
                w_30=self.w_excited_1 + self.w_excited_2 + self.w_excited_3 + _ * self.w_spacing_20 * 10**(-self.N_order_energy),

                g_10=_ * self.g_spacing_10 * 10**(-self.N_order),
                g_20=_ * self.g_spacing_20 * 10**(-self.N_order),
                g_30=_ * self.g_spacing_30 * 10**(-self.N_order),

                g_12=_ * self.g_spacing_12 * 10**(-self.N_order),
                g_13=_ * self.g_spacing_13 * 10**(-self.N_order),
                g_23=_ * self.g_spacing_23 * 10**(-self.N_order),
                omega_M1=self.omega_M1,
                omega_M2=self.omega_M2
            ) for _ in range(1, self.N_molecules+1)
        ]

        [self.molecules[_].update(
            {
                'w_12': self.molecules[_]['w_20'] - self.molecules[_]['w_10'],
                'w_13': self.molecules[_]['w_30'] - self.molecules[_]['w_10'],
                'w_23': self.molecules[_]['w_30'] - self.molecules[_]['w_20'],

            }
        ) for _ in range(self.N_molecules)]

        [self.molecules[_].update(
            {
                'w_01': self.molecules[_]['w_10'],
                'w_02': self.molecules[_]['w_20'],
                'w_03': self.molecules[_]['w_30'],

                'w_21': self.molecules[_]['w_12'],
                'w_31': self.molecules[_]['w_13'],
                'w_32': self.molecules[_]['w_23'],

                'g_21': self.molecules[_]['g_12'],
                'g_31': self.molecules[_]['g_13'],
                'g_32': self.molecules[_]['g_23'],

                'g_01': self.molecules[_]['g_10'],
                'g_02': self.molecules[_]['g_20'],
                'g_03': self.molecules[_]['g_30'],

            }
        ) for _ in range(self.N_molecules)]

        self.frequency = np.linspace(
            self.w_excited_1 - self.freq_halfwidth,
            self.w_excited_1 + self.freq_halfwidth - 2*self.freq_halfwidth/self.N_frequency,
            self.N_frequency
        )

        self.frequency_detection = np.linspace(
            self.detection - self.freq_halfwidth,
            self.detection + self.freq_halfwidth - 2 * self.freq_halfwidth / self.N_frequency,
            self.N_frequency
        )

        self.del_omega = self.omega_del * np.arange(-self.N_comb, self.N_comb)

        self.omega = self.frequency[:, np.newaxis, np.newaxis, np.newaxis]
        self.comb_omega1 = self.del_omega[np.newaxis, :, np.newaxis, np.newaxis]
        self.comb_omega2 = self.del_omega[np.newaxis, np.newaxis, :, np.newaxis]
        self.comb_omega3 = self.del_omega[np.newaxis, np.newaxis, np.newaxis, :]

        self.field1 = ne.evaluate(
            "sum(gamma / ((omega - omega_M1 - comb_omega1)**2 + gamma**2), axis=3)",
            local_dict=vars(self)
        ).sum(axis=(1, 2))
        self.field2 = ne.evaluate(
            "sum(gamma / ((omega - omega_M2 - comb_omega2)**2 + gamma**2), axis=3)",
            local_dict=vars(self)
        ).sum(axis=(1, 2))

# x = np.linspace(-10, 10, 300)
# H = np.linspace(-5, 5, 10)

######################################################################
#
#   Perform the same evaluation using C shared library
#
#   Note: You must compile the C shared library
#       gcc -O3 -shared -o fastest_evaluate.so fastest_evaluate.c -lm -fopenmp
#
######################################################################

if __name__ == '__main__':

    import pickle
    import os
    import ctypes
    import matplotlib.pyplot as plt

    start = time.time()
    w_excited_1 = 0.6e-6
    w_excited_2 = 2.354
    w_excited_3 = 1.7e-6
    detection = w_excited_1 + w_excited_2 - w_excited_3

    parameters = dict(
        N_molecules=1,
        N_order=5,
        N_order_energy=9,
        N_comb=20,
        N_frequency=2000,
        freq_halfwidth=4.e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,
        w_excited_3=w_excited_3,

        detection=detection,

        omega_M1=-3.2e-7,
        omega_M2=-6.4e-7,

        gamma=1e-9,
        omega_del=2.*16e-7,

        w_spacing_10=2.*0.50,
        w_spacing_20=2.*0.65,
        w_spacing_30=2.*0.80,

        g_spacing_10=0.45,
        g_spacing_20=0.35,
        g_spacing_30=0.40,
        g_spacing_12=0.25,
        g_spacing_13=0.20,
        g_spacing_23=0.30
    )

    ensemble = PolarizationTerms(**parameters)

    class MoleculeParams(ctypes.Structure):
        _fields_ = [(k, ctypes.c_double) for k, v in ensemble.molecules[0].items()]

    print(os.getcwd() + "/fastest_evaluate.so")

    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/fastest_evaluate.so")

    # specify the parameters of the c-function
    c_calc_pol3_a2 = lib.eval
    c_calc_pol3_a2.argtypes = (
        ctypes.POINTER(ctypes.c_double),        # double* out_r
        ctypes.POINTER(ctypes.c_double),        # double* out_i
        ctypes.c_int,                           # int size_out
        ctypes.POINTER(ctypes.c_int),           # int* index
        ctypes.c_int,                           # int x_num
        ctypes.c_double,                        # double x_min
        ctypes.c_double,                        # double x_max
        ctypes.c_double,                        # double h_min
        ctypes.c_double,                        # double h_max
        ctypes.POINTER(MoleculeParams),  # struct params
    )
    c_calc_pol3_a2.restype = ctypes.c_int

    result_c_r = np.zeros_like(ensemble.frequency)
    result_c_i = np.zeros_like(ensemble.frequency)
    index = np.array([1, 2, 2])
    t0 = time.time()

    for i in range(1):
        molecule = MoleculeParams()
        molecule.w_21 = ensemble.molecules[i]['w_21']
        molecule.g_20 = ensemble.molecules[i]['g_20']
        molecule.g_32 = ensemble.molecules[i]['g_32']
        molecule.g_31 = ensemble.molecules[i]['g_31']
        molecule.w_23 = ensemble.molecules[i]['w_23']
        molecule.g_10 = ensemble.molecules[i]['g_10']
        molecule.g_13 = ensemble.molecules[i]['g_13']
        molecule.g_12 = ensemble.molecules[i]['g_12']
        molecule.g_02 = ensemble.molecules[i]['g_02']
        molecule.g_03 = ensemble.molecules[i]['g_03']
        molecule.g_01 = ensemble.molecules[i]['g_01']
        molecule.w_20 = ensemble.molecules[i]['w_20']
        molecule.w_32 = ensemble.molecules[i]['w_32']
        molecule.w_31 = ensemble.molecules[i]['w_31']
        molecule.w_30 = ensemble.molecules[i]['w_30']
        molecule.g_21 = ensemble.molecules[i]['g_21']
        molecule.omega_M1 = ensemble.molecules[i]['omega_M1']
        molecule.w_13 = ensemble.molecules[i]['w_13']
        molecule.w_12 = ensemble.molecules[i]['w_12']
        molecule.w_02 = ensemble.molecules[i]['w_02']
        molecule.w_03 = ensemble.molecules[i]['w_03']
        molecule.w_01 = ensemble.molecules[i]['w_01']
        molecule.g_30 = ensemble.molecules[i]['g_30']
        molecule.w_10 = ensemble.molecules[i]['w_10']
        molecule.g_23 = ensemble.molecules[i]['g_23']
        molecule.omega_M2 = ensemble.molecules[i]['omega_M2']
        molecule.gamma = ensemble.gamma

    c_calc_pol3_a2(
        result_c_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        result_c_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        result_c_r.size,
        index.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ensemble.del_omega.size,
        ensemble.del_omega.min(),
        ensemble.del_omega.max(),
        ensemble.frequency.min(),
        ensemble.frequency.max(),
        molecule
    )

    print result_c_r
    print
    print result_c_i

    print("running C-library time: {} seconds".format(time.time() - t0))