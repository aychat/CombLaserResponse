from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from types import MethodType, FunctionType
import numexpr as ne
from itertools import permutations


class PolarizationTerms:
    """
    Calculates NL Polarizations for an ensemble of near identical molecules.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified:

        N_molecules: NUMBER OF MOLECULES IN THE DISCRIMINATION PROBLEM
        N_comb: NUMBER OF COMB LINES USED IN THE CALCULATION
        N_frequency: NUMBER OF FREQUENCY POINTS DESCRIBING POLARIZATION AND FIELDS

        ---------------------------------------------------
        THE FOLLOWING PARAMETERS ALL HAVE THE UNITS OF fs-1
        ---------------------------------------------------

        freq_halfwidth: HALF LENGTH OF FREQUENCY ARRAY CENTERED AROUND w_excited_1
            s.t. FREQ = [w_excited_1 - freq_halfwidth --> w_excited_1 + freq_halfwidth]

        w_excited_1: omega_10 = (1/hbar) * (E_1 - E_0)
        w_excited_2: omega_20 = (1/hbar) * (E_2 - E_0)

        omega_M1: MODULATION FREQUENCY 1 = omega_10 + OFFSET_1
        omega_M2: MODULATION FREQUENCY 2 = omega_10 + OFFSET_2
        gamma: ELECTRIC FIELD LINE-WIDTH
        omega_del_1: COMB-SPACING FOR MODULATOR 1
        omega_del_2: COMB-SPACING FOR MODULATOR 2

        w_spacing_10: omega_10 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        w_spacing_20: omega_20 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        g_spacing_10: gamma_10 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        g_spacing_12: gamma_12 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        g_spacing_20: gamma_20 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        """

        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.N_molecules
        except AttributeError:
            raise AttributeError("Number of molecules not specified")

        try:
            self.N_comb
        except AttributeError:
            raise AttributeError("Number of comb lines not specified")

        try:
            self.N_frequency
        except AttributeError:
            raise AttributeError("Number of frequency points not specified")

        try:
            self.w_excited_1
        except AttributeError:
            raise AttributeError("omega_10 not specified")

        try:
            self.w_excited_2
        except AttributeError:
            raise AttributeError("omega_20 not specified")

        try:
            self.omega_M1
        except AttributeError:
            raise AttributeError("Modulation frequency 1 not specified")

        try:
            self.omega_M2
        except AttributeError:
            raise AttributeError("Modulation frequency 2 not specified")

        try:
            self.gamma
        except AttributeError:
            raise AttributeError("Field line-width not specified")

        try:
            self.omega_del
        except AttributeError:
            raise AttributeError("Frequency Comb spacing is not specified")

        try:
            self.w_spacing_10
        except AttributeError:
            raise AttributeError("Molecule omega_10 increments not specified")

        try:
            self.g_spacing_10
        except AttributeError:
            raise AttributeError("Molecule gamma_10 increments not specified")

        try:
            self.w_spacing_20
        except AttributeError:
            raise AttributeError("Molecule omega_20 increments not specified")

        try:
            self.g_spacing_20
        except AttributeError:
            raise AttributeError("Molecule gamma_20 increments not specified")

        try:
            self.g_spacing_12
        except AttributeError:
            raise AttributeError("Molecule gamma_12 increments not specified")

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

    def calculate_pol_a2(self, m, n, v, a, b, c, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (a2) OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A FOUR-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_30,
        w_12, w_23, w_31, g_10, g_20, g_30, g_12, g_23, g_31
        :param m: indices (integer)
        :param n: indices (integer)
        :param v: indices (integer)
        :return: P^(3)(omega)_(a2)
        """

        local_vars = vars(self).copy()
        local_vars['pi'] = np.pi
        local_vars['w_mv'] = params['w_'+str(m)+str(v)]
        local_vars['w_nv'] = params['w_'+str(n)+str(v)]
        local_vars['w_v0'] = params['w_'+str(v)+str(0)]
        local_vars['g_mv'] = params['g_'+str(m)+str(v)]
        local_vars['g_nv'] = params['g_'+str(n)+str(v)]
        local_vars['g_v0'] = params['g_'+str(v)+str(0)]
        local_vars['omega_M1_lv'] = params['omega_M'+str(a)]
        local_vars['omega_M2_lv'] = params['omega_M'+str(b)]
        local_vars['omega_M3_lv'] = params['omega_M'+str(c)]

        print a, b, c, local_vars['omega_M1_lv'], local_vars['omega_M2_lv'], local_vars['omega_M3_lv']

        K = ne.evaluate("-pi * pi * gamma / (omega - omega_M3_lv - comb_omega3 - w_mv + 1j * (gamma + g_mv))", local_dict=local_vars)
        Q = ne.evaluate("omega - omega_M2_lv - comb_omega2 - omega_M3_lv - comb_omega3 - 2j*gamma", local_dict=local_vars)
        R = ne.evaluate("w_mv - omega_M2_lv - comb_omega2 - 1j*(gamma + g_mv)", local_dict=local_vars)
        G = ne.evaluate("1./(w_nv - omega - 1j*g_nv)", local_dict=local_vars)
        D = ne.evaluate("omega_M1_lv + comb_omega1 -1j*gamma", local_dict=local_vars)
        E = ne.evaluate("-w_v0 - 1j*g_v0", local_dict=local_vars)
        delta = ne.evaluate("omega - omega_M1_lv - comb_omega1 - omega_M2_lv - comb_omega2 - omega_M3_lv - comb_omega3", local_dict=local_vars)
        local_vars['delta'] = delta
        H = ne.evaluate("delta/(delta**2+9*gamma**2)", local_dict=local_vars)

        return ne.evaluate(
            "sum((K * G / (2j * (conj(D) - conj(Q)) * (conj(Q) - E)))*((conj(Q) - R)/(conj(D) - Q)*(conj(D) - R) - H), axis=3)"
        ).sum(axis=(1, 2))

    def calculate_pol_b1(self, m, n, v, a, b, c, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (b1) OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A FOUR-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_30,
        w_12, w_23, w_31, g_10, g_20, g_30, g_12, g_23, g_31
        :param m: indices
        :param n: indices
        :param v: indices
        :return: P^(3)(omega)_(a2)
        """

        local_vars = vars(self).copy()
        local_vars['pi'] = np.pi
        local_vars['w_vm'] = params['w_' + str(v) + str(m)]
        local_vars['w_nv'] = params['w_' + str(n) + str(v)]
        local_vars['w_m0'] = params['w_' + str(m) + str(0)]
        local_vars['g_vm'] = params['g_' + str(v) + str(m)]
        local_vars['g_nv'] = params['g_' + str(n) + str(v)]
        local_vars['g_m0'] = params['g_' + str(m) + str(0)]
        local_vars['omega_M1_lv'] = params['omega_M'+str(a)]
        local_vars['omega_M2_lv'] = params['omega_M'+str(b)]
        local_vars['omega_M3_lv'] = params['omega_M'+str(c)]

        print a, b, c, local_vars['omega_M1_lv'], local_vars['omega_M2_lv'], local_vars['omega_M3_lv']

        K = ne.evaluate("-pi * pi * gamma / (omega - omega_M3_lv - comb_omega3 + w_vm + 1j * (gamma - g_vm))",
                        local_dict=local_vars)
        Q = ne.evaluate("omega - omega_M2_lv - comb_omega2 - omega_M3_lv - comb_omega3 - 2j*gamma", local_dict=local_vars)
        R = ne.evaluate("-w_vm - omega_M2_lv - comb_omega2 - 1j*(gamma - g_vm)", local_dict=local_vars)
        G = ne.evaluate("1./(w_nv - omega - 1j*g_nv)", local_dict=local_vars)
        D = ne.evaluate("omega_M1_lv + comb_omega1 -1j*gamma", local_dict=local_vars)
        E = ne.evaluate("w_m0 - 1j*g_m0", local_dict=local_vars)
        delta = ne.evaluate("omega - omega_M1_lv - comb_omega1 - omega_M2_lv - comb_omega2 - omega_M3_lv - comb_omega3", local_dict=local_vars)
        local_vars['delta'] = delta
        H = ne.evaluate("delta/(delta**2+9*gamma**2)", local_dict=local_vars)
        return ne.evaluate(
            "sum((K * G / (2j * (conj(D) - conj(Q)) * (conj(Q) - E)))*((conj(Q) - R)/(conj(D) - Q)*(conj(D) - R) - H), axis=3)"
        ).sum(axis=(1, 2))

    def calculate_pol_c1(self, m, n, v, a, b, c, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (c1) OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A FOUR-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_30,
        w_12, w_23, w_31, g_10, g_20, g_30, g_12, g_23, g_31
        :param m: indices
        :param n: indices
        :param v: indices
        :return: P^(3)(omega)_(a2)
        """

        local_vars = vars(self).copy()
        local_vars['pi'] = np.pi
        local_vars['w_n0'] = params['w_' + str(n) + str(0)]
        local_vars['w_vn'] = params['w_' + str(v) + str(n)]
        local_vars['w_m0'] = params['w_' + str(m) + str(0)]
        local_vars['g_n0'] = params['g_' + str(n) + str(0)]
        local_vars['g_vn'] = params['g_' + str(v) + str(n)]
        local_vars['g_m0'] = params['g_' + str(m) + str(0)]
        local_vars['omega_M1_lv'] = params['omega_M'+str(a)]
        local_vars['omega_M2_lv'] = params['omega_M'+str(b)]
        local_vars['omega_M3_lv'] = params['omega_M'+str(c)]

        print a, b, c, local_vars['omega_M1_lv'], local_vars['omega_M2_lv'], local_vars['omega_M3_lv']
        K = ne.evaluate("-pi * pi * gamma / (omega - omega_M3_lv - comb_omega3 - w_n0 + 1j * (gamma + g_n0))",
                        local_dict=local_vars)

        Q = ne.evaluate("omega - omega_M2_lv - comb_omega2 - omega_M3_lv - comb_omega3 - 2j*gamma", local_dict=local_vars)
        R = ne.evaluate("w_n0 - omega_M2_lv - comb_omega2 - 1j*(gamma + g_n0)", local_dict=local_vars)
        G = ne.evaluate("1./(-w_vn - omega - 1j*g_vn)", local_dict=local_vars)
        D = ne.evaluate("omega_M1_lv + comb_omega1 -1j*gamma", local_dict=local_vars)
        E = ne.evaluate("w_m0 - 1j*g_m0", local_dict=local_vars)
        delta = ne.evaluate("omega - omega_M1_lv - comb_omega1 - omega_M2_lv - comb_omega2 - omega_M3_lv - comb_omega3",
                            local_dict=local_vars)
        local_vars['delta'] = delta
        H = ne.evaluate("delta/(delta**2+9*gamma**2)", local_dict=local_vars)
        return ne.evaluate(
            "sum((K * G / (2j * (conj(D) - conj(Q)) * (conj(Q) - E)))*((conj(Q) - R)/(conj(D) - Q)*(conj(D) - R) - H), axis=3)"
        ).sum(axis=(1, 2))

    def calculate_total_pol3(self, a, b, c, permute, **params):
        if permute == 1:
            return sum(
                (
                    self.calculate_pol_a2(m, n, v, a, b, c, **params) +
                    self.calculate_pol_b1(m, n, v, a, b, c, **params) +
                    self.calculate_pol_c1(m, n, v, a, b, c, **params)
                 ) for m, n, v in permutations([1, 2, 3])
            )
        else:
            return self.calculate_pol_a2(1, 2, 3, a, b, c, **params)

    def calculate_chi_3(self, m, n, v, z, **params):
        local_vars = vars(self).copy()
        local_vars['pi'] = np.pi
        local_vars['w_n0'] = params['w_' + str(n) + str(0)]
        local_vars['w_vn'] = params['w_' + str(v) + str(n)]
        local_vars['w_m0'] = params['w_' + str(m) + str(0)]
        local_vars['g_n0'] = params['g_' + str(n) + str(0)]
        local_vars['g_vn'] = params['g_' + str(v) + str(n)]
        local_vars['g_m0'] = params['g_' + str(m) + str(0)]
        f1 = self.frequency[:, np.newaxis]
        f2 = self.frequency[np.newaxis, :]
        f3 = z
        local_vars['f1'] = f1
        local_vars['f2'] = f2
        local_vars['f3'] = f3

        return ne.evaluate("1. / ((w_vn - 1j*g_vn - f1 - f2 - f3)*(w_m0 - 1j*g_m0 - f1 - f2)*(w_n0 - 1j*g_n0 - f1))", local_dict=local_vars)


if __name__ == '__main__':

    import time
    import pickle
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

    # pol3_mat = np.asarray([
    #     (
    #         ensemble.calculate_total_pol3(1, 1, 1, permute=0, **m)
    #         + ensemble.calculate_total_pol3(1, 1, 2, permute=0, **m)
    #         + ensemble.calculate_total_pol3(1, 2, 1, permute=0, **m)
    #         + ensemble.calculate_total_pol3(1, 2, 2, permute=0, **m)
    #         + ensemble.calculate_total_pol3(2, 1, 1, permute=0, **m)
    #         + ensemble.calculate_total_pol3(2, 1, 2, permute=0, **m)
    #         + ensemble.calculate_total_pol3(2, 2, 1, permute=0, **m)
    #         + ensemble.calculate_total_pol3(2, 2, 2, permute=0, **m)
    #                         ) for m in ensemble.molecules])
    # factor = np.abs(pol3_mat).max() / np.abs(ensemble.field1.max())
    # pol3_mat /= factor

    plt.figure()
    plt.plot(ensemble.frequency, ensemble.field1/ensemble.field1.max(), 'g')
    plt.plot(ensemble.frequency, ensemble.field2/ensemble.field1.max(), 'y')
    factor = 3.*np.abs(ensemble.calculate_total_pol3(1, 1, 1, permute=0, **ensemble.molecules[0])).real.max()
    print factor
    plt.plot(ensemble.frequency, ensemble.calculate_total_pol3(1, 1, 1, permute=0, **ensemble.molecules[0]).real/factor, label='$P^{(3)}_{111}$')
    plt.plot(ensemble.frequency, ensemble.calculate_total_pol3(1, 1, 2, permute=0, **ensemble.molecules[0]).real/factor
             + ensemble.calculate_total_pol3(1, 2, 1, permute=0, **ensemble.molecules[0]).real/factor
             + ensemble.calculate_total_pol3(2, 1, 1, permute=0, **ensemble.molecules[0]).real/factor
             , label='$P^{(3)}_{211} + P^{(3)}_{121} + P^{(3)}_{112}$')
    plt.plot(ensemble.frequency,
             ensemble.calculate_total_pol3(2, 2, 1, permute=0, **ensemble.molecules[0]).real/factor
             + ensemble.calculate_total_pol3(2, 1, 2, permute=0, **ensemble.molecules[0]).real/factor
             + ensemble.calculate_total_pol3(1, 2, 2, permute=0, **ensemble.molecules[0]).real/factor
             , label='$P^{(3)}_{221} + P^{(3)}_{212} + P^{(3)}_{122}$')
    plt.plot(ensemble.frequency, ensemble.calculate_total_pol3(2, 2, 2, permute=0, **ensemble.molecules[0]).real/factor, label='$P^{(3)}_{222}$')
    plt.legend()
    # plt.plot(pol3_mat.real[0])
    plt.ylabel("Polarizations (arbitrary units)")
    plt.xlabel("Frequency (in fs$^{-1}$)")
    colors = ['r', 'k', 'b']
    # plt.figure()
    # [(plt.subplot(311+i), plt.plot(ensemble.frequency, pol3_mat[i], color=colors[i])) for i in range(ensemble.N_molecules)]

    # with open("Polarization_order3_data.pickle", "wb") as output_file:
    #     pickle.dump(
    #         {
    #             "frequency": ensemble.frequency,
    #             "molecules_pol3": pol3_mat,
    #             "field1": ensemble.field1,
    #             "field2": ensemble.field2,
    #             'params': parameters
    #         }, output_file
    #     )
    print time.time() - start
    plt.show()

