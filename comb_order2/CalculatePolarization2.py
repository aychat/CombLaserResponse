from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array

from eval_pol2_wrapper import pol2_total

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


def nonuniform_frequency_range_2(params, freq_offset=None):
    """
    Generation of nonuniform frequency range tailored to the 2nd order optical effects 
    :param params: 
    :param freq_offset:
    :return: 
    """
    omega_M1 = params.omega_M1
    omega_M2 = params.omega_M2

    # If freq_offset has not been specified, generate all unique third order combinations
    if not freq_offset:
        freq_offset = np.array([
            sum(_) for _ in combinations_with_replacement(
                [omega_M1, omega_M2, -omega_M1, -omega_M2], 2
            )
        ])

        # get number of digits to round to
        decimals = np.log10(np.abs(freq_offset[np.nonzero(freq_offset)]).min())
        decimals = int(np.ceil(abs(decimals))) + 4

        # round of
        np.round(freq_offset, decimals, out=freq_offset)

        freq_offset = np.unique(freq_offset)
        print freq_offset

    # def points_for_lorentzian(mean):
    #     # L = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 1.])
    #     L = np.array([0, 0.05, 0.4, 1.])
    #     L = np.append(-L[::-1], L[1::])
    #
    #     return mean + 4. * params.gamma * L

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


def get_polarization2(molecule, params):
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
    polarization_mn = np.zeros_like(polarization)

    for m, n in permutations(range(1, len(energy))):
        print(m, n)
        try:
            # calculate the product of the transition dipole if they are not zeros
            mu_product = transition[(0, n)].mu * transition[(n, m)].mu * transition[(m, 0)].mu

            # reset the polarization because C-code performs "+="
            polarization_mn[:] = 0.

            all_modulations = product(*(2 * [[params.omega_M1, params.omega_M2]]))

            for M_field_i, M_field_j in all_modulations:
                pol2_total(
                    polarization_mn, params,
                    M_field_i, M_field_j,
                    energy[n] - energy[0] - 1j * transition[(n, 0)].g,
                    energy[m] - energy[0] - 1j * transition[(m, 0)].g,
                    energy[m] - energy[n] - 1j * transition[(m, n)].g,
                    energy[n] - energy[m] - 1j * transition[(n, m)].g,
                )

                print(M_field_i, M_field_j)

            polarization_mn *= mu_product
            polarization += polarization_mn
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
    E_10 = 2.354
    E_21 = 2.354
    central_freq = 0.0

    molecule = ADict(

        energies=[0., E_10, E_21 + E_10],

        # dipole value and line width for each transition
        transitions={
            (0, 1): CTransition(0.45e-7, 1.),
            (1, 0): CTransition(0.45e-7, 1.),
            (0, 2): CTransition(0.35e-7, 1.),
            (2, 0): CTransition(0.35e-7, 1.),
            (1, 2): CTransition(0.25e-7, 1.),
            (2, 1): CTransition(0.25e-7, 1.),
        }
    )

    params = ADict(
        N_frequency=2000,
        comb_size=20,
        freq_halfwidth=1e-5,
        central_freq=central_freq,
        omega_M1=central_freq-1e-7,
        omega_M2=central_freq-3e-7,
        gamma=1e-9,
        delta_freq=2.5e-7,
    )

    import time
    start = time.time()
    # choosing the frequency range
    frequency = nonuniform_frequency_range_2(params)
    # frequency = uniform_frequency_range(params)
    params['freq'] = frequency
    print params.freq.size

    pol2 = get_polarization2(molecule, params)

    omega = frequency[:, np.newaxis]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (params.gamma / ((omega - (params.omega_M1 - comb_omega))**2 + params.gamma**2)).sum(axis=1)
    field2 = (params.gamma / ((omega - (params.omega_M2 - comb_omega))**2 + params.gamma**2)).sum(axis=1)

    plt.figure()
    # comb_plot(frequency, field1/field1.max(), 'b-', alpha=0.5)
    # comb_plot(frequency, field2/field1.max(), 'r-', alpha=0.5)

    comb_plot(frequency, -(pol2/(np.abs(pol2).max())).real, 'k-')
    plt.plot(frequency, np.zeros_like(frequency), 'k-')
    plt.xlabel("$\\omega_1 + \\omega_2 + \\Delta \\omega$ (in GHz)")
    plt.ylabel("$\mathcal{R}e[P^{(2)}(\\omega)]$")

    with open("Pol2_data.pickle", "wb") as output_file:
        pickle.dump(
            {
                "molecules_pol2": pol2,
                "freq": frequency,
                "field1": field1,
                "field2": field2,
            }, output_file
        )

    print time.time() - start
    plt.show()