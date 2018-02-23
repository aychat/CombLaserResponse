import numpy as np
from types import MethodType, FunctionType
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PolarizationTerms import PolarizationTerms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class ChiPlot(PolarizationTerms):
    """
    Calculates NL Polarizations for an ensemble of near identical molecules.
    """
    def __init__(self, **kwargs):
        PolarizationTerms.__init__(self, **kwargs)
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

    def total_chi_2(self, **params):
        return self.calculate_chi(1, 2, **params) + self.calculate_chi(2, 1, **params)


if __name__ == "__main__":
    w_excited_1 = 2.354
    w_excited_2 = 4.708

    atomic = ChiPlot(
        N_molecules=1,
        N_order=7,
        N_order_energy=9,
        N_comb=20,
        N_frequency=1000,
        freq_halfwidth=.01e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,

        omega_M1=w_excited_1 + .5e-7,
        omega_M2=w_excited_1 + 1.5e-7,
        gamma=2.5e-9,
        omega_del_1=2.4e-7,
        omega_del_2=2.4e-7,

        w_spacing_10=0.60*3.,
        w_spacing_20=0.70*3.,
        g_spacing_10=0.25*3.,
        g_spacing_12=0.40*3.,
        g_spacing_20=0.35*3.,
    )

    laser_dye = ChiPlot(
        N_molecules=1,
        N_order=6,
        N_order_energy=9,
        N_comb=20,
        N_frequency=1000,
        freq_halfwidth=1e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,

        omega_M1=w_excited_1 + .5e-7,
        omega_M2=w_excited_1 + 1.5e-7,
        gamma=2.5e-9,
        omega_del_1=2.4e-7,
        omega_del_2=2.4e-7,

        w_spacing_10=0.60*3.,
        w_spacing_20=0.70*3.,
        g_spacing_10=0.25*3.,
        g_spacing_12=0.40*3.,
        g_spacing_20=0.35*3.,
    )

    plot_chi_atomic = [atomic.total_chi_2(**m) for m in atomic.molecules]
    plot_chi_laser_dye = [laser_dye.total_chi_2(**m) for m in laser_dye.molecules]

    X_a = atomic.frequency
    X_l = laser_dye.frequency
    Y_a = atomic.frequency
    Y_l = laser_dye.frequency

    N_levels = 15
    f, ax = plt.subplots(2, 4)
    for i in range(atomic.N_molecules):
        level = np.linspace(plot_chi_atomic[i].real.min(), plot_chi_atomic[i].real.max(), N_levels)
        cax = ax[1, 0].contourf(X_a, Y_a, plot_chi_atomic[i].real,
                                levels=level,
                                cmap=cm.coolwarm)
        ax[1, 0].set_ylabel("Frequency (in $fs$)")
        cbar = f.colorbar(cax, ticks=level[::2], ax=ax[1, 0])

    for i in range(atomic.N_molecules):
        level = np.linspace(plot_chi_atomic[i].imag.min(), plot_chi_atomic[i].imag.max(), N_levels)
        cax = ax[1, 1].contourf(X_a, Y_a, plot_chi_atomic[i].imag,
                                levels=level,
                                cmap=cm.coolwarm)
        ax[1, 1].set_ylabel("Frequency (in $fs$)")
        cbar = f.colorbar(cax, ticks=level[::2], ax=ax[1, 1])

    ax[0, 0].plot(atomic.frequency, atomic.total_chi_2(**atomic.molecules[0]).real.sum(axis=0), 'b')
    ax[0, 0].grid()
    ax[0, 1].plot(atomic.frequency, atomic.total_chi_2(**atomic.molecules[0]).imag.sum(axis=0), 'b')
    ax[0, 1].grid()
    ax[0, 2].plot(laser_dye.frequency, laser_dye.total_chi_2(**laser_dye.molecules[0]).real.sum(axis=0), 'b')
    ax[0, 2].grid()
    ax[0, 3].plot(laser_dye.frequency, laser_dye.total_chi_2(**laser_dye.molecules[0]).imag.sum(axis=0), 'b')
    ax[0, 3].grid()

    for i in range(laser_dye.N_molecules):
        level = np.linspace(plot_chi_laser_dye[i].real.min(), plot_chi_laser_dye[i].real.max(), N_levels)
        cax = ax[1, 2].contourf(X_l, Y_l, plot_chi_laser_dye[i].real,
                                levels=level,
                                cmap=cm.coolwarm)
        ax[1, 2].set_ylabel("Frequency (in $fs$)")
        cbar = f.colorbar(cax, ticks=level[::2], ax=ax[1, 2])

    for i in range(laser_dye.N_molecules):
        level = np.linspace(plot_chi_laser_dye[i].imag.min(), plot_chi_laser_dye[i].imag.max(), N_levels)
        cax = ax[1, 3].contourf(X_l, Y_l, plot_chi_laser_dye[i].imag,
                                levels=level,
                                cmap=cm.coolwarm)
        ax[1, 3].set_ylabel("Frequency (in $fs$)")
        cbar = f.colorbar(cax, ticks=level[::2], ax=ax[1, 3])
    plt.subplots_adjust(left=0.07, bottom=0.15, right=0.95, top=0.95, wspace=0.90, hspace=0.75)
    for i in range(2):
        for j in range(4):
            for label in ax[i, j].get_xmajorticklabels() + ax[i, j].get_ymajorticklabels():
                label.set_rotation(30)

    # plt.figure()
    # plt.plot(laser_dye.frequency, laser_dye.calculate_broad_response(**laser_dye.molecules[0]).sum(axis=(1, 2)))
    # plt.figure()
    # pol2_laser_real = laser_dye.calculate_total_pol(**laser_dye.molecules[0]).real
    # pol2_laser_imag = laser_dye.calculate_total_pol(**laser_dye.molecules[0]).imag
    # pol2_laser_real /= pol2_laser_real.max()/laser_dye.field1.max()
    # pol2_laser_imag /= pol2_laser_imag.max()/laser_dye.field1.max()

    # plt.subplot(211)
    # plt.plot(laser_dye.frequency, pol2_laser_real, 'k')
    # plt.plot(laser_dye.frequency, laser_dye.field1, 'b')
    # plt.plot(laser_dye.frequency, laser_dye.field2, 'r')

    # plt.subplot(212)
    # plt.plot(laser_dye.frequency, pol2_laser_imag, 'k')
    # plt.plot(laser_dye.frequency, laser_dye.field1, 'b')
    # plt.plot(laser_dye.frequency, laser_dye.field2, 'r')
    plt.show()
