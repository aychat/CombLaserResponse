import numpy as np
import matplotlib.pyplot as plt
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
            self.omega_del_1
        except AttributeError:
            raise AttributeError("Frequency Comb spacing 1 not specified")

        try:
            self.omega_del_2
        except AttributeError:
            raise AttributeError("Frequency Comb spacing 2 not specified")

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

        self.detection = self.w_excited_1 + self.w_excited_2 - self.w_excited_3

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

        self.del_omega1 = self.omega_del_1 * np.arange(-self.N_comb, self.N_comb)
        self.del_omega2 = self.omega_del_2 * np.arange(-self.N_comb, self.N_comb)
        self.del_omega3 = self.omega_del_3 * np.arange(-self.N_comb, self.N_comb)

        self.omega = self.frequency[:, np.newaxis, np.newaxis, np.newaxis]
        self.comb_omega1 = self.del_omega1[np.newaxis, :, np.newaxis, np.newaxis]
        self.comb_omega2 = self.del_omega2[np.newaxis, np.newaxis, :, np.newaxis]
        self.comb_omega3 = self.del_omega3[np.newaxis, np.newaxis, np.newaxis, :]

        self.field1 = ne.evaluate(
            "sum(gamma / ((omega - omega_M1 - comb_omega1)**2 + gamma**2), axis=3)",
            local_dict=vars(self)
        ).sum(axis=(1, 2))
        self.field2 = ne.evaluate(
            "sum(gamma / ((omega - omega_M2 - comb_omega2)**2 + gamma**2), axis=3)",
            local_dict=vars(self)
        ).sum(axis=(1, 2))
        self.field3 = ne.evaluate(
            "sum(gamma / ((omega - omega_M3 - comb_omega3)**2 + gamma**2), axis=3)",
            local_dict=vars(self)
        ).sum(axis=(1, 2))

    def calculate_pol_a2(self, m, n, v, **params):
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

        K = ne.evaluate("pi * gamma / (omega - omega_M3 - comb_omega3 - w_mv + 1j * (gamma + g_mv))", local_dict=local_vars)
        Q = ne.evaluate("omega - omega_M2 - comb_omega2 - omega_M3 - comb_omega3 - 2j*gamma", local_dict=local_vars)
        R = ne.evaluate("w_mv - omega_M2 - comb_omega2 - 1j*(gamma + g_mv)", local_dict=local_vars)
        G = ne.evaluate("1./(w_nv - omega - 1j*g_nv)", local_dict=local_vars)
        D = ne.evaluate("omega_M1 + comb_omega1 -1j*gamma", local_dict=local_vars)
        E = ne.evaluate("-w_v0 - 1j*g_v0", local_dict=local_vars)
        delta = ne.evaluate("omega - omega_M1 - comb_omega1 - omega_M2 - comb_omega2 - omega_M3 - comb_omega3", local_dict=local_vars)
        local_vars['delta'] = delta
        H = ne.evaluate("delta/(delta**2+9*gamma**2)", local_dict=local_vars)

        return ne.evaluate(
            "sum((K * G / (2j * (conj(D) - conj(Q)) * (conj(Q) - E)))*((conj(Q) - R)/(conj(D) - Q)*(conj(D) - R) - H), axis=3)"
        ).sum(axis=(1, 2))

    def calculate_pol_b1(self, m, n, v, **params):
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
        K = ne.evaluate("pi * gamma / (omega - omega_M3 - comb_omega3 + w_vm + 1j * (gamma - g_vm))",
                        local_dict=local_vars)
        Q = ne.evaluate("omega - omega_M2 - comb_omega2 - omega_M3 - comb_omega3 - 2j*gamma", local_dict=local_vars)
        R = ne.evaluate("-w_vm - omega_M2 - comb_omega2 - 1j*(gamma - g_vm)", local_dict=local_vars)
        G = ne.evaluate("1./(w_nv - omega - 1j*g_nv)", local_dict=local_vars)
        D = ne.evaluate("omega_M1 + comb_omega1 -1j*gamma", local_dict=local_vars)
        E = ne.evaluate("w_m0 - 1j*g_m0", local_dict=local_vars)
        delta = ne.evaluate("omega - omega_M1 - comb_omega1 - omega_M2 - comb_omega2 - omega_M3 - comb_omega3", local_dict=local_vars)
        local_vars['delta'] = delta
        H = ne.evaluate("delta/(delta**2+9*gamma**2)", local_dict=local_vars)
        return ne.evaluate(
            "sum((K * G / (2j * (conj(D) - conj(Q)) * (conj(Q) - E)))*((conj(Q) - R)/(conj(D) - Q)*(conj(D) - R) - H), axis=3)"
        ).sum(axis=(1, 2))

    def calculate_pol_c1(self, m, n, v, **params):
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
        K = ne.evaluate("pi * gamma / (omega - omega_M3 - comb_omega3 - w_n0 + 1j * (gamma + g_n0))",
                        local_dict=local_vars)

        Q = ne.evaluate("omega - omega_M2 - comb_omega2 - omega_M3 - comb_omega3 - 2j*gamma", local_dict=local_vars)
        R = ne.evaluate("w_n0 - omega_M2 - comb_omega2 - 1j*(gamma + g_n0)", local_dict=local_vars)
        G = ne.evaluate("1./(-w_vn - omega - 1j*g_vn)", local_dict=local_vars)
        D = ne.evaluate("omega_M1 + comb_omega1 -1j*gamma", local_dict=local_vars)
        E = ne.evaluate("w_m0 - 1j*g_m0", local_dict=local_vars)
        delta = ne.evaluate("omega - omega_M1 - comb_omega1 - omega_M2 - comb_omega2 - omega_M3 - comb_omega3",
                            local_dict=local_vars)
        local_vars['delta'] = delta
        H = ne.evaluate("pi*delta/(delta**2+9*gamma**2)", local_dict=local_vars)
        return ne.evaluate(
            "sum((K * G / (2j * (conj(D) - conj(Q)) * (conj(Q) - E)))*((conj(Q) - R)/(conj(D) - Q)*(conj(D) - R) - H), axis=3)"
        ).sum(axis=(1, 2))

    def calculate_total_pol3(self, **params):
        return self.calculate_pol_a2(1, 2, 3, **params)

        # return sum(
        #     (
        #         self.calculate_pol_a2(m, n, v, **params) +
        #         self.calculate_pol_b1(m, n, v, **params) +
        #         self.calculate_pol_c1(m, n, v, **params)
        #      ) for m, n, v in permutations([1, 2, 3])
        # )


if __name__ == '__main__':

    w_excited_1 = 0.6e-6
    w_excited_2 = 2.354
    w_excited_3 = 1.7e-6

    ensemble = PolarizationTerms(
        N_molecules=1,
        N_order=5,
        N_order_energy=9,
        N_comb=30,
        N_frequency=500,
        freq_halfwidth=2e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,
        w_excited_3=w_excited_3,

        omega_M1=w_excited_1 - 3.2e-7,
        omega_M2=w_excited_1 - 6.4e-7,
        omega_M3=w_excited_1 + 1.6e-7,

        gamma=1e-9,
        omega_del_1=16e-7,
        omega_del_2=16e-7,
        omega_del_3=16e-7,

        w_spacing_10=0.50,
        w_spacing_20=0.65,
        w_spacing_30=0.80,

        g_spacing_10=0.45,
        g_spacing_20=0.35,
        g_spacing_30=0.40,
        g_spacing_12=0.25,
        g_spacing_13=0.20,
        g_spacing_23=0.30
    )

    pol3_mat = np.asarray([ensemble.calculate_total_pol3(**m).real for m in ensemble.molecules])
    # print pol3_mat[0]
    factor = np.abs(pol3_mat).max() / np.abs(ensemble.field1.max())
    pol3_mat /= factor
    plt.figure()

    plt.plot(ensemble.frequency_detection, ensemble.field1, 'g')
    plt.plot(ensemble.frequency_detection, ensemble.field2, 'y')
    plt.plot(ensemble.frequency_detection, ensemble.field3, 'm')

    plt.plot(ensemble.frequency_detection, np.abs(pol3_mat[0]), 'r')
    # plt.plot(ensemble.frequency_detection, np.abs(pol3_mat[1]), 'b')
    # plt.plot(ensemble.frequency_detection, np.abs(pol3_mat[2]), 'k')
    plt.ylabel("Polarizations (arbitrary units)")
    plt.xlabel("Frequency (in fs$^{-1}$)")
    plt.show()