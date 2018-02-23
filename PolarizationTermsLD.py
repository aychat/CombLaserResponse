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
            "sum(gamma / ((omega - 2*omega_M2 - 2*comb_omega2)**2 + gamma**2), axis=1)",
            local_dict=vars(self)
        ).sum(axis=1)

        self.pol2_freq_matrix = np.asarray([self.calculate_total_pol(**instance) for instance in self.molecules]).T
        self.freq_axis_0 = self.frequency[:, np.newaxis]
        self.freq_axis_1 = self.frequency[np.newaxis, :]

    def calculate_pol_12_21_012(self, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (a2') OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A THREE-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_12,
        w_21, g_10, g_20, g_12, g_21
        :return: P^(2)(omega)_(a1)
        """
        term_J1 = np.pi*self.gamma*(self.omega - 2*params['w_10'] + self.omega_M1 - self.omega_M2 + self.comb_omega1 - self.comb_omega2)
        term_J2 = np.pi*self.gamma*(self.omega - 2*params['w_10'] + self.omega_M2 - self.omega_M1 + self.comb_omega2 - self.comb_omega1)

        term_K1 = (self.omega - self.omega_M2 - self.comb_omega2 - params['w_10']) + 1j*(self.gamma + params['g_10'])
        term_K2 = (self.omega_M1 + self.comb_omega1 - params['w_10']) + 1j*(self.gamma + params['g_10'])

        term_K1_d = (self.omega - self.omega_M1 - self.comb_omega1 - params['w_10']) + 1j * (self.gamma + params['g_10'])
        term_K2_d = (self.omega_M2 + self.comb_omega2 - params['w_10']) + 1j * (self.gamma + params['g_10'])

        delta = (self.omega - self.omega_M1 - self.omega_M2 - self.comb_omega1 - self.comb_omega2)**2 + 4.*self.gamma**2

        A_012 = params['w_20'] - self.omega - 1j*params['g_20']
        return ne.evaluate(
            "sum((term_J1/(term_K1*term_K2) + term_J2/(term_K1_d*term_K2_d))/(delta*A_012), axis=2)"
        ).sum(axis=1)

    def calculate_pol_12_21_021(self, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (a2') OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A THREE-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_12,
        w_21, g_10, g_20, g_12, g_21
        :return: P^(2)(omega)_(a1)
        """
        term_J1 = np.pi * self.gamma * (self.omega - 2 * params['w_20'] + self.omega_M1 - self.omega_M2 + self.comb_omega1 - self.comb_omega2)
        term_J2 = np.pi * self.gamma * (self.omega - 2 * params['w_20'] + self.omega_M2 - self.omega_M1 + self.comb_omega2 - self.comb_omega1)

        term_K1 = (self.omega - self.omega_M2 - self.comb_omega2 - params['w_20']) + 1j * (self.gamma + params['g_20'])
        term_K2 = (self.omega_M1 + self.comb_omega1 - params['w_20']) + 1j * (self.gamma + params['g_20'])

        term_K1_d = (self.omega - self.omega_M1 - self.comb_omega1 - params['w_20']) + 1j * (self.gamma + params['g_20'])
        term_K2_d = (self.omega_M2 + self.comb_omega2 - params['w_20']) + 1j * (self.gamma + params['g_20'])

        delta = (
                self.omega - self.omega_M1 - self.omega_M2 - self.comb_omega1 - self.comb_omega2) ** 2 + 4. * self.gamma ** 2

        A_021 = params['w_10'] - self.omega - 1j * params['g_10']

        return ne.evaluate(
            "sum((term_J1/(term_K1*term_K2) + term_J2/(term_K1_d*term_K2_d))/(delta*A_021), axis=2)"
        ).sum(axis=1)

    def calculate_total_pol(self, **params):
        return self.calculate_pol_12_21_012(**params) + self.calculate_pol_12_21_021(**params)

    def calculate_chi2_full_a1(self, m, n):
        return [1./(
            (self.molecules[i]['w_'+str(n)+'0'] - self.freq_axis_0 - 1j*self.molecules[i]['g_'+str(n)+'0']) *
            (self.molecules[i]['w_'+str(m)+'0'] - self.freq_axis_1 - 1j*self.molecules[i]['g_'+str(m)+'0']))
                for i in range(self.N_molecules)]


if __name__ == '__main__':

    w_excited_1 = 2.354
    w_excited_2 = 4.708

    ensemble = PolarizationTerms(
        N_molecules=3,
        N_order=5,
        N_order_energy=9,
        N_comb=80,
        N_frequency=1000,
        freq_halfwidth=1e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,

        omega_M1=w_excited_1 + .5e-7,
        omega_M2=w_excited_1 + .7e-7,
        gamma=2.5e-9,
        omega_del_1=.6e-7,
        omega_del_2=.6e-7,

        w_spacing_10=0.60,
        w_spacing_20=0.70,
        g_spacing_10=0.25,
        g_spacing_12=0.40,
        g_spacing_20=0.35,
    )

    import matplotlib.cm as cm

    fig, ax = plt.subplots()

    cax = ax.imshow(ensemble.calculate_chi2_full_a1(1, 2)[0].T.real.reshape(ensemble.frequency.size, ensemble.frequency.size), interpolation='nearest', cmap=cm.gist_rainbow)
    cbar = fig.colorbar(cax)

    plt.figure()
    plt.plot(ensemble.frequency, ensemble.field1, 'g')
    plt.plot(ensemble.frequency, ensemble.field2, 'y')
    plt.plot(ensemble.frequency, ensemble.calculate_chi2_full_a1(1, 2)[0].sum(axis=1))
    plt.ylabel("Electric fields and $\chi^{(2)}(\omega_1)$")
    plt.xlabel("Frequency ($fs^{-1}$)")
    plt.show()
