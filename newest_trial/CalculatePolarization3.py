from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array

from eval_pol3_wrapper import pol3_new

############################################################################################
#                                                                                          #
#   Declare new types: ADict to access dictionary elements with a (.) rather than ['']     #
#                                                                                          #
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
#                                                                                          #
#           Declare uniform and non-uniform (localized around comb frequencies)            #
#                                                                                          #
############################################################################################


def uniform_frequency_range(params):
    """
    Generate uniformly spaced frequency range
    :type params: object
    :param params: 
    :return: 
    """
    return np.linspace(
        params.central_freq - params.freq_halfwidth,
        params.central_freq + params.freq_halfwidth - 2.*params.freq_halfwidth/params.N_frequency,
        params.N_frequency
    )


def nonuniform_frequency_range_3(params, freq_offset=None):
    """
    Generation of nonuniform frequency range taylored to the 3d order optical effects 
    :param params: 
    :param freq_offset:
    :return: 
    """
    omega_M1 = params.omega_M1
    omega_M2 = params.omega_M2

    # If freq_offset has not been specified, generate all unique third order combinations
    if not freq_offset:
        freq_offset = np.array([
            sum(_) for _ in combinations_with_replacement([omega_M1, omega_M2, -omega_M1, -omega_M2], 3)
        ])

        # get number of digits to round to
        decimals = np.log10(np.abs(freq_offset[np.nonzero(freq_offset)]).min())
        decimals = int(np.ceil(abs(decimals))) + 3

        # round of
        np.round(freq_offset, decimals, out=freq_offset)

        freq_offset = np.unique(freq_offset)
        print freq_offset

    # def points_for_lorentzian(mean):
    #     L = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 1.])
    #     L = np.append(-L[::-1], L[1::])
    #
    #     return mean + 4. * params.gamma * L
    #
    # lorentzians_per_comb_line = np.hstack(points_for_lorentzian(_) for _ in freq_offset)
    lorentzians_per_comb_line = freq_offset

    lorentzians_per_comb_line = lorentzians_per_comb_line[:, np.newaxis]

    # Positions of all comb lines
    position_comb_lines = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]

    freq = lorentzians_per_comb_line + position_comb_lines
    freq = freq.reshape(-1)
    freq = freq[np.nonzero(
        (params.central_freq - params.freq_halfwidth < freq) & (freq < params.central_freq + params.freq_halfwidth)
    )]
    freq.sort()

    return np.ascontiguousarray(freq)


def get_polarization3(molecule, params):
    """
    Return the third order polarization for a specified molecule 
    :param molecule: an instance of ADict describing molecule 
    :param params: an instance of ADict specifying calculation parameters
    :return: numpy.array -- polarization
    """


    # introducing aliases
    transition = molecule.transitions
    energy = molecule.energies

    # initialize output array with zeros
    polarization = np.zeros(params.freq.size, dtype=np.complex)
    polarization_mnv = np.zeros_like(polarization)

    # for m, n, v in permutations(range(1, len(energy))):
    for m, n, v in [(1, 2, 3)]:
        try:
            # calculate the product of the transition dipole if they are not zeros
            mu_product = transition[(0, v)].mu * transition[(v, n)].mu * \
                         transition[(n, m)].mu * transition[(m, 0)].mu

            # reset the polarization because C-code performs "+="
            polarization_mnv[:] = 0.

            for M_field_h, M_field_i, M_field_j in product(*(3 * [[params.omega_M1, params.omega_M2]])):
                pol3_new(
                    polarization_mnv, params,
                    M_field_h, M_field_i, M_field_j,
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


def comb_plot(frequency, value, *args, **kwargs):
    """
    Plotting with comb structure 
    :param frequency: 
    :param value: 
    :param kwargs: 
    :return: 
    """
    for omega, val in zip(frequency, value):
        plt.plot((omega, omega), (0, val), *args, **kwargs)

############################################################################################
#
#   Run test
#
############################################################################################
if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    # Energy difference of levels
    E_10 = 0.6e-6
    E_21 = 2.354
    E_32 = 1.7e-6
    central_freq = E_10 + E_21 - E_32

    molecule = ADict(

        energies=[0., E_10, E_21 + E_10, E_32 + E_21 + E_10],

        # dipole value and line width for each transition
        transitions={
            (0, 1): CTransition(0.45, 1.),
            (1, 0): CTransition(0.45, 1.),
            (0, 2): CTransition(0.35, 1.),
            (2, 0): CTransition(0.35, 1.),
            (0, 3): CTransition(0.40, 1.),
            (3, 0): CTransition(0.40, 1.),
            (1, 2): CTransition(0.25, 1.),
            (2, 1): CTransition(0.25, 1.),
            (1, 3): CTransition(0.20, 1.),
            (3, 1): CTransition(0.20, 1.),
            (2, 3): CTransition(0.30, 1.),
            (3, 2): CTransition(0.30, 1.)
        }
    )

    params = ADict(
        N_frequency=500,
        comb_size=20,
        freq_halfwidth=5.e-5,
        central_freq=central_freq,
        omega_M1=central_freq-1e-6,
        omega_M2=central_freq-3e-6,
        gamma=1e-9,
        delta_freq=2.5e-6,
    )

    # choosing the frequency range
    # frequency = nonuniform_frequency_range_3(params)
    frequency = uniform_frequency_range(params)
    params['freq'] = frequency
    print params.freq.size

    pol3 = get_polarization3(molecule, params)

    omega = frequency[:, np.newaxis]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (params.gamma / ((omega - params.omega_M1 - comb_omega)**2 + params.gamma**2)).sum(axis=1)
    field2 = (params.gamma / ((omega - params.omega_M2 - comb_omega)**2 + params.gamma**2)).sum(axis=1)

    #plt.figure()

    comb_plot(frequency, field1, 'b*-')
    # plt.plot(frequency, field1, 'b*-')
    comb_plot(frequency, field2, 'r*-')
    # plt.plot(frequency, field2, 'r*-')

    comb_plot(frequency, (pol3/(pol3.max()/field1.max())).real, 'k*-')
    data = pickle.load(open("Pol3_data.pickle", "rb"))
    pol3_ = data['molecules_pol3']
    # plt.plot(frequency, pol3_ / (pol3_.max() / field1.max()), 'g*-')
    # plt.xlim([30e-6, 35e-6])

    plt.show()

