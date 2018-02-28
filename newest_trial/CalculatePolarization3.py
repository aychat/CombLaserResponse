from itertools import permutations, product
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array

from eval_pol3_wrapper import pol3

############################################################################################
#
#   declare types
#
############################################################################################

class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)

CTransition = namedtuple("CTransition", ["g", "mu"])

############################################################################################
#
#   declare parameters
#
############################################################################################


def get_polarization3(molecule, params):
    """
    Return the third order polarization for a specified molecule 
    :param molecule: an instance of ADict describing molecule 
    :param params: an instance of ADict specifying calculation parameters
    :return: numpy.array containing polarization
    """

    # introducing aliases
    transition = molecule.transitions
    energy = molecule.energies

    # initialize output array with zeros
    polarization = np.zeros(params.N_frequency, dtype=np.complex)
    polarization_mnv = np.zeros_like(polarization)

    for m, n, v in permutations(range(1, len(energy))):
        try:
            # calculate the product of the transition dipole if they are not zeros
            mu_product = transition[(0, v)].mu * transition[(v, n)].mu * \
                         transition[(n, m)].mu * transition[(m, 0)].mu

            for M_field1, M_field2, M_field3 in product(*(3 * [[params.omega_M1, params.omega_M2]])):
                pol3(
                    polarization_mnv, params,
                    M_field1, M_field2, M_field3,
                    energy[n] - energy[v] + 1j * transition[(n, v)].g,
                    energy[m] - energy[v] + 1j * transition[(m, v)].g,
                    energy[v] - energy[0] + 1j * transition[(v, 0)].g
                )

            polarization_mnv *= mu_product
            polarization += polarization_mnv
        except KeyError:
            # Not allowed transition, this diagram is not calculated
            pass

    return polarization

############################################################################################
#
#   Run test
#
############################################################################################
if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    # Energy difference of levels
    E_10 = 0.6e-6
    E_21 = 2.354
    E_32 = 1.7e-6

    molecule = ADict(

        energies=[0., E_10, E_21 + E_10, E_32 + E_21 + E_10],

        # dipole value and line width for each transition
        transitions={
            (0, 1): CTransition(0.45, 1.),
            (0, 2): CTransition(0.35, 1.),
            (0, 3): CTransition(0.40, 1.),
            (1, 0): CTransition(0.45, 1.),
            (1, 2): CTransition(0.25, 1.),
            (1, 3): CTransition(0.20, 1.),
            (2, 0): CTransition(0.35, 1.),
            (2, 1): CTransition(0.25, 1.),
            (2, 3): CTransition(0.30, 1.),
            (3, 0): CTransition(0.40, 1.),
            (3, 1): CTransition(0.20, 1.),
            (3, 2): CTransition(0.30, 1.)
        }
    )

    params = ADict(
        comb_size=20,
        N_frequency=2000,
        freq_halfwidth=4.e-5,
        central_freq=E_10,
        omega_M1=-3.2e-7,
        omega_M2=-6.4e-7,
        gamma=1e-9,
        delta_freq=2. * 16e-7,
    )

    frequency = np.linspace(
        params.central_freq - params.freq_halfwidth,
        params.central_freq + params.freq_halfwidth,
        params.N_frequency
    )

    plt.figure()

    import time
    t0 = time.time()

    plt.plot(frequency, get_polarization3(molecule, params), '*-')

    print("Running time: {:.1f} sec".format(time.time() - t0))

    plt.show()

# class CMolecule(Structure):
#     _fields_ = [
#         ("NE", c_int),   # the number of levels
#         ("E", POINTER(c_double)), # Array of energy levels
#
#         ("Nt", c_int), # the number of transitions
#         ("transitions", POINTER(Transition)), # Array of transitions
#     ]
#
# class Transition(Structure):
#     _fields_ = [
#         ("i", c_int),  # the level number i
#         ("j", c_int),  # the level number j
#         ("g", c_double),  # the linewidth g_ij
#         ("mu", c_double),  # the dipole moment mu_ij
#     ]

