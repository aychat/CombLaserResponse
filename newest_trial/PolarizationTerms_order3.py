from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from types import MethodType, FunctionType
import numexpr as ne
from itertools import permutations, product


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

        self.frequency = np.linspace(
            self.energies[1] - self.freq_halfwidth,
            self.energies[1] + self.freq_halfwidth - 2*self.freq_halfwidth/self.N_frequency,
            self.N_frequency
        )

        self.del_omega = self.omega_del * np.arange(-self.N_comb, self.N_comb)

        self.omega = self.frequency[:, np.newaxis, np.newaxis, np.newaxis]
        self.comb_omega1 = self.del_omega[np.newaxis, :, np.newaxis, np.newaxis]
        self.comb_omega2 = self.del_omega[np.newaxis, np.newaxis, :, np.newaxis]
        self.comb_omega3 = self.del_omega[np.newaxis, np.newaxis, np.newaxis, :]

        self.field1 = (self.gamma / ((self.omega - self.omega_M1 - self.comb_omega1) ** 2 + self.gamma ** 2)).sum(axis=(1,2,3))
        self.field2 = (self.gamma / ((self.omega - self.omega_M2 - self.comb_omega2) ** 2 + self.gamma ** 2)).sum(axis=(1,2,3))

    def calculate_pol_a2(self, m, n, v, M_field1, M_field2, M_field3, **params):

        local_vars = vars(self).copy()
        local_vars['pi'] = np.pi
        local_vars['w_mv'] = params['energies'][m] - params['energies'][v]
        local_vars['w_nv'] = params['energies'][n] - params['energies'][v]
        local_vars['w_v0'] = params['energies'][v] - params['energies'][0]
        local_vars['g_mv'] = params['g_'+str(m)+str(v)]
        local_vars['g_nv'] = params['g_'+str(n)+str(v)]
        local_vars['g_v0'] = params['g_'+str(v)+str(0)]
        local_vars['omega_M1'] = M_field1
        local_vars['omega_M2'] = M_field2
        local_vars['omega_M3'] = M_field3

        A = ne.evaluate("1./(w_nv - omega - 1j*g_nv)", local_dict=local_vars)
        P = ne.evaluate("omega_M1 + comb_omega1 -1j*gamma", local_dict=local_vars)
        R = ne.evaluate("w_mv - omega_M2 - comb_omega2 - 1j*(gamma + g_mv)", local_dict=local_vars)

        Q = ne.evaluate("omega - omega_M2 - comb_omega2 - omega_M3 - comb_omega3 - 2j*gamma", local_dict=local_vars)

        E = ne.evaluate("-w_v0 - 1j*g_v0", local_dict=local_vars)

        term_s = ne.evaluate("omega - omega_M1 - comb_omega1 - omega_M2 - comb_omega2 - omega_M3 - comb_omega3", local_dict=local_vars)
        local_vars['term_s'] = term_s
        W = ne.evaluate("-2.*term_s/(term_s**2+9*gamma**2)", local_dict=local_vars)
        B = ne.evaluate("1. / (omega - omega_M3 - comb_omega3 - w_mv + 1j * (gamma + g_mv))", local_dict=local_vars)

        return ne.evaluate(
            "sum((B * A / (2j * (conj(P) - conj(Q)) * (conj(Q) - E)))*((conj(Q) - R)/(conj(P) - Q)*(conj(P) - R) + W), axis=3)"
        ).sum(axis=(1, 2))

    def calculate_total_pol3(self, **params):
        pol3_tot = np.zeros(self.N_frequency, dtype=np.complex)
        # for m, n, v in permutations(range(1, 4)):
        for m, n, v in [(1, 2, 3)]:
            for M_field1, M_field2, M_field3 in product(*(3 * [[self.omega_M1, self.omega_M2]])):
                pol3_tot += ensemble.calculate_pol_a2(m, n, v, M_field1, M_field2, M_field3, **params)
                print m, n, v
        return pol3_tot


if __name__ == '__main__':

    import time
    import pickle
    start = time.time()
    E_10 = 0.6e-6
    E_21 = 2.354
    E_32 = 1.7e-6

    parameters = dict(
        N_comb=20,
        N_frequency=2000,
        freq_halfwidth=5.e-5,

        energies=[0., E_10, E_21 + E_10, E_32 + E_21 + E_10],

        omega_M1=-1e-6,
        omega_M2=-3e-6,

        gamma=1e-9,
        omega_del=2.5e-6,

        g_10=0.45,
        g_01=0.45,
        g_20=0.35,
        g_02=0.35,
        g_30=0.40,
        g_03=0.40,
        g_12=0.25,
        g_21=0.25,
        g_13=0.20,
        g_31=0.20,
        g_23=0.30,
        g_32=0.30
    )

    ensemble = PolarizationTerms(**parameters)

    pol3 = ensemble.calculate_total_pol3(**parameters)
    fld1 = ensemble.field1
    fld2 = ensemble.field2
    plt.plot(ensemble.frequency, fld1, 'b*-')
    plt.plot(ensemble.frequency, fld2, 'r*-')
    plt.plot(ensemble.frequency, pol3/(pol3.max()/fld1.max()), 'k*-')
    with open("Pol3_data.pickle", "wb") as output_file:
        pickle.dump(
            {
                "molecules_pol3": pol3,
                "freq": ensemble.frequency
            }, output_file
        )
    plt.show()



