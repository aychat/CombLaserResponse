import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
from PolarizationTerms import PolarizationTerms


class FrequencyComb(PolarizationTerms):
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
        PolarizationTerms.__init__(self, **kwargs)
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

        self.pol2_freq_matrix = np.asarray([self.calculate_total_pol(**instance) for instance in self.molecules]).T.real
        # self.pol2_freq_matrix /= np.abs(self.pol2_freq_matrix).max()

    def heterodyne_fields_frequency_basis(self, k):
        """
        CALCULATES THE HETERODYNE FIELDS TO DETECT THE n POLARIZATIONS
        :param k: k_th POLARIZATION TO WHICH THE HETERODYNE FIELD IS NON-ORTHOGONAL
        :return: (N_frequency - N_molecules) NUMBER OF HETERODYNE FIELD VECTORS FROM THE QR DECOMPOSITION OF THE
        POLARIZATION MATRIX DUE ALL BUT THE k_th POLARIZATIONS
        """
        return np.delete(
            np.linalg.qr(np.delete(self.pol2_freq_matrix.real, k, 1), mode='complete')[0],
            np.s_[:self.N_molecules-1],
            1
        )

    def pol2_basis_change_freq2comb(self):
        """
        CALCULATES BASIS CHANGE MATRIX FROM FREQUENCY BASIS TO COMB BASIS: DIM - (N_frequency x N_comb)
        :return: POLARIZATION MATRIX IN THE COMB BASIS: DIM - (N_frequency x N_molecules)
        """
        self.omega_cb = self.frequency[:, np.newaxis]
        self.del_omega1_cb = self.omega_del_1 * np.arange(-2*self.N_comb, 2*self.N_comb)
        self.comb_omega1_cb = self.del_omega1_cb[np.newaxis, :]

        self.freq2comb_basis = self.gamma / (
            (self.omega_cb - self.omega_M1 - self.omega_M2 - self.comb_omega1_cb) ** 2 + self.gamma ** 2
        )
        self.freq2comb_basis /= self.freq2comb_basis.max()
        assert self.freq2comb_basis.shape == (self.N_frequency, 4*self.N_comb), "Mismatching dims for basis-change matrix"
        self.pol2_comb_matrix = self.freq2comb_basis.T.dot(self.pol2_freq_matrix.real)
        return self.pol2_comb_matrix

    def heterodyne_fields_comb_basis(self, k):
        """
        CALCULATES THE HETERODYNE FIELDS TO DETECT THE n POLARIZATIONS
        :param k: k_th POLARIZATION TO WHICH THE HETERODYNE FIELD IS NON-ORTHOGONAL
        :return: (N_frequency - N_molecules) NUMBER OF HETERODYNE FIELD VECTORS FROM THE QR DECOMPOSITION OF THE
        POLARIZATION MATRIX DUE TO ALL BUT THE k_th POLARIZATIONS
        """
        try:
            self.pol2_comb_matrix
        except:
            self.pol2_basis_change_freq2comb()

        return np.delete(np.linalg.qr(np.delete(self.pol2_comb_matrix.real, k, 1), mode='complete')[0], np.s_[:self.N_molecules-1], 1)


if __name__ == '__main__':
    from matplotlib import cm

    w_excited_1 = 2.354
    w_excited_2 = 4.708

    ensemble = FrequencyComb(
        N_molecules=3,
        N_order=6,
        N_order_energy=6,
        N_comb=50,
        N_frequency=1000,
        freq_halfwidth=1.5e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,

        omega_M1=w_excited_1 + .3e-7,
        omega_M2=w_excited_1 + 1.8e-7,
        gamma=2.5e-9,
        omega_del_1=3e-7,
        omega_del_2=3e-7,

        w_spacing_10=0.60,
        w_spacing_20=0.70,
        g_spacing_10=0.25,
        g_spacing_12=0.40,
        g_spacing_20=0.35,
    )

    ensemble.pol2_basis_change_freq2comb()

    plt.figure()
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 4*ensemble.N_comb))

    plt.subplot(211)
    [plt.plot(ensemble.frequency, ensemble.freq2comb_basis[:, i], color=colors[i]) for i in range(4*ensemble.N_comb)]
    plt.subplot(212)
    plt.plot(ensemble.frequency, ensemble.freq2comb_basis.sum(axis=1))
    plt.plot(ensemble.frequency, ensemble.pol2_freq_matrix/ensemble.pol2_freq_matrix.max())

    P_f = ensemble.pol2_freq_matrix
    P_c = ensemble.pol2_comb_matrix
    CB = ensemble.freq2comb_basis
    P_f /= P_f.max()
    P_c /= P_c.max()
    CB /= CB.max()

    print P_f.shape, P_f.max(), P_f.min()
    print P_c.shape, P_c.max(), P_c.min()
    print CB.shape, CB.max(), CB.min()

    print np.allclose(P_c, CB.T.dot(P_f))
    Q_f, R_f = np.linalg.qr(P_f, mode='complete')

    def gaussian(x, x_0, sigma):
        return (1./(sigma*np.sqrt(np.pi)))*np.exp(-(x-x_0)**2/(2*sigma**2))

    colors1 = cm.Reds(np.linspace(0.5, 1, 3))
    colors2 = cm.Blues(np.linspace(0.5, 1, 3))
    f, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    f.suptitle("$2^{nd}$ order non-linear polarization $P^{(2)}(\\omega)$ and corresponding \n heterodyne fields $E^{het}(\\omega)$ for 3 near-identical atomic systems")
    for i in range(ensemble.N_molecules):
        Q_c, R_c = np.linalg.qr(np.delete(P_c, i, 1), mode='complete')
        het_fields = Q_c[:, ensemble.N_molecules:]
        ax[0, i].plot(ensemble.frequency, CB.dot(P_c[:, i]), color=colors1[i])
        # ax[0, i].set_xlabel("Frequency (in $fs$)")
        G = gaussian(np.linspace(0., 1., het_fields.shape[1]), 0.5, .05)
        heterodyne = CB.dot(np.asarray([G[j] * het_fields[:, j] for j in range(het_fields.shape[1])]).sum(axis=0))
        heterodyne /= heterodyne.max()
        ax[1, i].plot(ensemble.frequency, heterodyne, color=colors2[i])
        ax[1, i].set_xlabel("Frequency (in $fs$)")
        print [heterodyne.dot(P_f[:, i]) for i in range(3)]
    ax[0, 0].set_ylabel("Normalized Polarization \n $P^{(2)}(\\omega)$")
    ax[1, 0].set_ylabel("Heterodyne fields \n $E^{het}(\\omega)$")

    for i in range(2):
        for j in range(3):
            for label in ax[i, j].get_xmajorticklabels() + ax[i, j].get_ymajorticklabels():
                label.set_rotation(30)

    plt.show()

