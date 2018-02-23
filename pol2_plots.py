import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
import matplotlib.tri as mtri

N = 4

molecules = [
    dict(
        w_in=2.354 + np.random.uniform(1.0, 1.5)*1e-6,
        g_in=np.random.uniform(0.0, 1.0)*1e-6,
        w_out=4.708 + np.random.uniform(1.5, 2.0)*1e-6,
        g_out=np.random.uniform(1.0, 2.0)*1e-6,
    ) for _ in range(N)
]

omegaM1 = 2.354 + 7e-8
omegaM2 = 2.354 + 1e-8
gamma = 1.5e-9

N_freq = 1000
freq = np.linspace(2.*2.354-5e-6, 2.*2.354+5e-6, N_freq)

omega_del1 = 12e-8
omega_del2 = 12e-8

N_comb = 20
del_omega1 = omega_del1 * np.arange(-N_comb, N_comb)
del_omega2 = omega_del2 * np.arange(-N_comb, N_comb)

omega = freq[:, np.newaxis, np.newaxis]
comb_omega1 = del_omega1[np.newaxis, :, np.newaxis]
comb_omega2 = del_omega2[np.newaxis, np.newaxis, :]

field1 = gamma / ((omega - 2*omegaM1 - 2*comb_omega1)**2 + 4*gamma**2)
field2 = gamma / ((omega - 2*omegaM2 - 2*comb_omega2)**2 + 4*gamma**2)

########################################################################################################################
#                                                                                                                      #
#         ----------------- SECOND ORDER POLARIZATION FUNCTIONS FOR THE OFF-DIAGONAL TERM --------------------         #
#                                                                                                                      #
########################################################################################################################


def calculate_chi_a1(w, w1, w_in, g_in, w_out, g_out):
    return 1./((w_out - w - 1j*g_out)*(w_in - w1 - 1j*g_in))


X, Y = np.meshgrid(freq, freq)
# X, Y = X.flatten(), Y.flatten()
Z = np.asarray([calculate_chi_a1(X, Y, **m) for m in molecules])

fig = plt.figure()

ax11 = fig.add_subplot(4, 4, 1)
ax12 = fig.add_subplot(4, 4, 2)
ax13 = fig.add_subplot(4, 4, 5)
ax14 = fig.add_subplot(4, 4, 6)

ax21 = fig.add_subplot(4, 4, 3)
ax22 = fig.add_subplot(4, 4, 4)
ax23 = fig.add_subplot(4, 4, 7)
ax24 = fig.add_subplot(4, 4, 8)

ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

im = ax11.imshow(Z[0].real, extent=[freq.min(), freq.max(), freq.min(), freq.max()])
im = ax12.imshow(Z[1].real, extent=[freq.min(), freq.max(), freq.min(), freq.max()])
im = ax13.imshow(Z[2].real, extent=[freq.min(), freq.max(), freq.min(), freq.max()])
im = ax14.imshow(Z[3].real, extent=[freq.min(), freq.max(), freq.min(), freq.max()])

im = ax21.imshow(Z[0].imag, extent=[freq.min(), freq.max(), freq.min(), freq.max()])
im = ax22.imshow(Z[1].imag, extent=[freq.min(), freq.max(), freq.min(), freq.max()])
im = ax23.imshow(Z[2].imag, extent=[freq.min(), freq.max(), freq.min(), freq.max()])
im = ax24.imshow(Z[3].imag, extent=[freq.min(), freq.max(), freq.min(), freq.max()])

[(ax3.plot(Z[i].real.sum(axis=0)), ax4.plot(Z[i].imag.sum(axis=0))) for i in range(N)]

for ax in (ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24):
    plt.setp(ax.get_xticklabels(), rotation=25, fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=25, fontsize=10)

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7], aspect=20)
fig.colorbar(im, cax=cbar_ax)


plt.show()