import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
import numexpr as ne


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
                w_10=2.354 + _ * self.w_spacing_10 * 10**(-self.N_order_energy),
                g_10=_ * self.g_spacing_10 * 10**(-self.N_order),
                w_20=4.708 + _ * self.w_spacing_20 * 10**(-self.N_order_energy),
                g_20=_ * self.g_spacing_20 * 10**(-self.N_order),
                g_12=_ * self.g_spacing_12 * 10**(-self.N_order),
            ) for _ in range(1, self.N_molecules+1)
        ]

        [self.molecules[_].update(
            {
                'w_12': self.molecules[_]['w_20'] - self.molecules[_]['w_10']
            }
        ) for _ in range(self.N_molecules)]

        [self.molecules[_].update(
            {
                'w_21': self.molecules[_]['w_12'],
                'g_21': self.molecules[_]['g_12']

            }
        ) for _ in range(self.N_molecules)]

        self.frequency = np.linspace(
            2.*self.w_excited_1 - self.freq_halfwidth,
            2.*self.w_excited_1 + self.freq_halfwidth - 2*self.freq_halfwidth/self.N_frequency,
            self.N_frequency
        )

        self.del_omega1 = self.omega_del_1 * np.arange(-self.N_comb, self.N_comb)
        self.del_omega2 = self.omega_del_2 * np.arange(-self.N_comb, self.N_comb)

        self.omega = self.frequency[:, np.newaxis, np.newaxis]
        self.comb_omega1 = self.del_omega1[np.newaxis, :, np.newaxis]
        self.comb_omega2 = self.del_omega2[np.newaxis, np.newaxis, :]

        self.field1 = ne.evaluate(
            "sum(gamma / ((omega - 2*omega_M1 - 2*comb_omega1)**2 + gamma**2), axis=2)",
            local_dict=vars(self)
        ).sum(axis=1)
        self.field2 = ne.evaluate(
            "sum(gamma / ((omega - 2*omega_M2 - 2*comb_omega2)**2 + gamma**2), axis=2)",
            local_dict=vars(self)
        ).sum(axis=1)
        # self.comb_basis = ne.evaluate(
        #     "sum(gamma / ((omega - omega_M1 - comb_omega1 - omega_M2 - comb_omega2)**2 + gamma**2), axis=2)",
        #     local_dict=vars(self)
        # ).sum(axis=1)
        self.pol2_freq_matrix = np.asarray([self.calculate_total_pol(**instance) for instance in self.molecules]).T

    def calculate_pol_12_21(self, m, n, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (a2') OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A THREE-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_12,
        w_21, g_10, g_20, g_12, g_21
        :return: P^(2)(omega)_(a1)
        """
        term_J1 = np.pi*self.gamma*(self.omega - 2*params['w_'+str(m)+str(0)] + self.omega_M1 - self.omega_M2 + self.comb_omega1 - self.comb_omega2)
        term_J2 = np.pi*self.gamma*(self.omega - 2*params['w_'+str(m)+str(0)] + self.omega_M2 - self.omega_M1 + self.comb_omega2 - self.comb_omega1)

        term_K1 = (self.omega - self.omega_M2 - self.comb_omega2 - params['w_'+str(m)+str(0)]) + 1j*(self.gamma + params['g_'+str(m)+str(0)])
        term_K2 = (self.omega_M1 + self.comb_omega1 - params['w_'+str(m)+str(0)]) + 1j*(self.gamma + params['g_'+str(m)+str(0)])

        term_K1_d = (self.omega - self.omega_M1 - self.comb_omega1 - params['w_'+str(m)+str(0)]) + 1j * (self.gamma + params['g_'+str(m)+str(0)])
        term_K2_d = (self.omega_M2 + self.comb_omega2 - params['w_'+str(m)+str(0)]) + 1j * (self.gamma + params['g_'+str(m)+str(0)])

        delta = (self.omega - self.omega_M1 - self.omega_M2 - self.comb_omega1 - self.comb_omega2)**2 + 4.*self.gamma**2

        A_012 = params['w_'+str(n)+str(0)] - self.omega - 1j*params['g_'+str(n)+str(0)]
        return ne.evaluate(
            "sum((term_J1/(term_K1*term_K2) + term_J2/(term_K1_d*term_K2_d))/(delta*A_012), axis=2)"
        ).sum(axis=1)

        # return (np.pi*self.gamma/delta).sum(axis=(1, 2))

    def calculate_total_pol(self, **params):
        return self.calculate_pol_12_21(1, 2, **params) + self.calculate_pol_12_21(2, 1, **params)


if __name__ == '__main__':

    w_excited_1 = 2.354
    w_excited_2 = 4.708

    ensemble = PolarizationTerms(
        N_molecules=3,
        N_order=6,
        N_order_energy=9,
        N_comb=30,
        N_frequency=1000,
        freq_halfwidth=1e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,

        omega_M1=w_excited_1 + .5e-7,
        omega_M2=w_excited_1 + 1.5e-7,
        gamma=2.5e-9,
        omega_del_1=2.4e-7,
        omega_del_2=2.4e-7,

        w_spacing_10=0.60,
        w_spacing_20=0.70,
        g_spacing_10=0.25,
        g_spacing_12=0.40,
        g_spacing_20=0.35,
    )

    # print ensemble.__init__.__doc__

    # pol_order = np.zeros(20)
    # order = np.zeros(20)
    # for i in range(20):
    #     ensemble.gamma=2.5*10**(5-i)
    #     order[i]=ensemble.gamma
    #     pol_order[i] = np.abs(ensemble.calculate_total_pol(**ensemble.molecules[0]).max())
    #     print order[i], pol_order[i]
    #
    # plt.figure()
    # plt.plot(-np.log10(order), np.log10(pol_order))
    # plt.show()
    pol2_mat = np.asarray([ensemble.calculate_total_pol(**m).real for m in ensemble.molecules])

    factor = np.abs(ensemble.calculate_total_pol(**ensemble.molecules[0]).max()) / np.abs(ensemble.field1.max())
    pol2_mat /= factor
    plt.figure()

    # plt.subplot(221)
    plt.plot(ensemble.frequency, pol2_mat[0], 'r')
    plt.plot(ensemble.frequency, pol2_mat[1], 'b')
    plt.plot(ensemble.frequency, pol2_mat[2], 'k')

    # plt.subplot(222)
    # plt.plot(ensemble.frequency, ensemble.calculate_pol_12_21_021(**ensemble.molecules[0]) / factor, 'r')
    # plt.plot(ensemble.frequency, ensemble.calculate_pol_12_21_021(**ensemble.molecules[1]) / factor, 'b')
    # plt.plot(ensemble.frequency, ensemble.calculate_pol_12_21_021(**ensemble.molecules[2]) / factor, 'k')

    # plt.subplot(212)
    plt.plot(ensemble.frequency, ensemble.field1, 'g')
    plt.plot(ensemble.frequency, ensemble.field2, 'y')
    plt.show()