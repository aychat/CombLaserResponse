import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fmin, curve_fit


def spectra_freq(filename):
    abs_interpolate_num = 500
    lamb, absorption_lamb = np.loadtxt(filename, unpack=True)
    freq_w2f = 2. * np.pi * 3e2 / lamb
    absorption_freq_w2f = absorption_lamb * 2. * np.pi * 3e2 / (freq_w2f*freq_w2f)

    interpolation_function = interp1d(freq_w2f, absorption_freq_w2f, kind='cubic')
    d_freq = (freq_w2f[1] - freq_w2f[0]) / abs_interpolate_num
    freq = np.linspace(freq_w2f[0], freq_w2f[-1] - d_freq, abs_interpolate_num)
    absorption = interpolation_function(freq)

    return freq, absorption


freq_BFP, absorption_BFP = spectra_freq("DataFiles/TagBFP_spectra.dat")
freq_CFP, absorption_CFP = spectra_freq("DataFiles/TagCFP_spectra.dat")
freq_YFP, absorption_YFP = spectra_freq("DataFiles/TagYFP_spectra.dat")
freq_RFP, absorption_RFP = spectra_freq("DataFiles/TagRFP_spectra.dat")
freq_mKate2, absorption_mKate2 = spectra_freq("DataFiles/mKate2_spectra.dat")

absorption_BFP /= absorption_BFP.max()
absorption_CFP /= absorption_CFP.max()
absorption_YFP /= absorption_YFP.max()
absorption_RFP /= absorption_RFP.max()
absorption_mKate2 /= absorption_mKate2.max()

plt.figure()
# plt.subplot(211)
plt.plot(freq_mKate2, absorption_mKate2, 'm', label='mKate2')
plt.plot(freq_BFP, absorption_BFP, 'b', label='BFP')
plt.plot(freq_CFP, absorption_CFP, 'c', label='CFP')
# plt.plot(freq_YFP, absorption_YFP, 'y', label='YFP')
# plt.plot(freq_RFP, absorption_RFP, 'r', label='RFP')
plt.xlabel("Frequency (fs$^{-1}$)")
plt.ylabel("Normalized absorption")
plt.legend()


def func(x, abs, freq):
    return np.linalg.norm(abs - (
        x[0] * (1. / (x[1] - freq - 1j * x[2]) + 1. / (x[1] + freq + 1j * x[2])) +
        x[3] * (1. / (x[4] - freq - 1j * x[5]) + 1. / (x[4] + freq + 1j * x[5])) +
        x[6] * (1. / (x[7] - freq - 1j * x[8]) + 1. / (x[7] + freq + 1j * x[8])) +
        x[9] * (1. / (x[10] - freq - 1j * x[11]) + 1. / (x[10] + freq + 1j * x[11])) +
        x[12] * (1. / (x[13] - freq - 1j * x[14]) + 1. / (x[13] + freq + 1j * x[14]))
    ).imag*freq, 2)


def func_plot(x, freq):
    return (
        x[0] * (1. / (x[1] - freq - 1j * x[2]) + 1. / (x[1] + freq + 1j * x[2])) +
        x[3] * (1. / (x[4] - freq - 1j * x[5]) + 1. / (x[4] + freq + 1j * x[5])) +
        x[6] * (1. / (x[7] - freq - 1j * x[8]) + 1. / (x[7] + freq + 1j * x[8])) +
        x[9] * (1. / (x[10] - freq - 1j * x[11]) + 1. / (x[10] + freq + 1j * x[11])) +
        x[12] * (1. / (x[13] - freq - 1j * x[14]) + 1. / (x[13] + freq + 1j * x[14]))
    ).imag*freq


def func_plot_truncate(x, freq):
    return (
        x[0] * (1. / (x[1] - freq - 1j * x[2]) + 1. / (x[1] + freq + 1j * x[2])) +
        x[3] * (1. / (x[4] - freq - 1j * x[5]) + 1. / (x[4] + freq + 1j * x[5])) +
        x[6] * (1. / (x[7] - freq - 1j * x[8]) + 1. / (x[7] + freq + 1j * x[8])) +
        x[9] * (1. / (x[10] - freq - 1j * x[11]) + 1. / (x[10] + freq + 1j * x[11]))
    ).imag*freq


xopt_BFP = fmin(func, [.02, 4.5, .1, .022, 4.7, .085, .02, 4.9, .1, .022, 4.75, .085, .022, 4.85, .085],
                xtol=1e-6, args=(absorption_BFP, freq_BFP))
xopt_CFP = fmin(func, [.02, 4.2, .1, .022, 4.45, .085, .02, 4.9, .1, .022, 3.75, .085, .022, 4.5, .085],
                xtol=1e-6, args=(absorption_CFP, freq_CFP))
xopt_mKate2 = fmin(func, [.02, 3.2, .1, .022, 3.5, .085, .02, 2.9, .1, .022, 3.6, .085, .022, 3.85, .085],
                   xtol=1e-6, args=(absorption_mKate2, freq_mKate2))

plt.plot(freq_BFP, func_plot(xopt_BFP, freq_BFP), 'b-.', label='BFP --')
plt.plot(freq_CFP, func_plot(xopt_CFP, freq_CFP), 'c-.', label='CFP --')
# plt.plot(freq_YFP, func_plot(xopt_YFP, freq_YFP), 'y-.', label='YFP --')
# plt.plot(freq_RFP, func_plot(xopt_RFP, freq_RFP), 'r-.', label='RFP --')
plt.plot(freq_mKate2, func_plot(xopt_mKate2, freq_mKate2), 'm-.', label='mKate2 --')

plt.xlabel("Frequency (fs$^{-1}$)")
plt.ylabel("Normalized absorption")
plt.legend()

print xopt_BFP
print xopt_BFP
print xopt_mKate2
plt.show()
