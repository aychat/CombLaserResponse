import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors, ticker, cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.tri as mtri

N = 3

molecules = [
    dict(
        w_in=2.354 + _*0.5*1e-6,
        g_in=_*0.25*1e-6,
        w_out=4.708 + _*0.75*1e-6,
        g_out=_*0.35*1e-6,
    ) for _ in range(1, N+1)
]

omegaM1 = 2.354 + .5e-7
omegaM2 = 2.354 + 1.5e-7
gamma = 2.5e-9

N_freq = 1000
freq = np.linspace(2.*2.354-1e-5, 2.*2.354+1e-5 - 2e-5/N_freq, N_freq)

omega_del1 = 2e-7
omega_del2 = 2e-7

N_comb = 30
del_omega1 = omega_del1 * np.arange(-N_comb, N_comb)
del_omega2 = omega_del2 * np.arange(-N_comb, N_comb)

omega = freq[:, np.newaxis, np.newaxis]
comb_omega1 = del_omega1[np.newaxis, :, np.newaxis]
comb_omega2 = del_omega2[np.newaxis, np.newaxis, :]

field1 = 8.*gamma / ((omega - 2*omegaM1 - 2*comb_omega1)**2 + gamma**2)
field2 = 8.*gamma / ((omega - 2*omegaM2 - 2*comb_omega2)**2 + gamma**2)
comb_basis = 8.*gamma / ((omega - omegaM1 - omegaM2 - comb_omega1)**2 + gamma**2)
print comb_basis.shape


def calculate_pol_a1_12(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM2 - comb_omega2 - w_in + 1j*(gamma + g_in)
    A2 = omegaM1 + comb_omega1 - w_in + 1j*(gamma + g_in)
    K1 = (omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return 1e-13*(K1*K2/(A1*A2*B)).sum(axis=(1, 2))


pol_matrix = np.asarray([calculate_pol_a1_12(**m) for m in molecules]).T
# pol_basis = np.linalg.inv(comb_basis.T.dot(comb_basis)).dot(comb_basis.T.dot(pol_matrix))
# print pol_basis.shape

# k = 0
# Q, R = np.linalg.qr(np.delete(pol_basis.real, k, 1), mode='complete')
# result = np.asarray([[Q[j, :].dot(pol_basis.real[:, i]) for i in range(N)] for j in range(N, N_comb*2)])
#
# Q_sum = np.sum(Q[N:], axis=0)
#
# W = result.sum(axis=0)    #  / np.delete(result, k, 1).max(axis=1)
# print W
# print "Wmax = ", W.max(), len(filter(lambda x: x > 0, W)), W.size
# result_sum = [Q_sum.dot(pol_basis.real[:, i]) for i in range(N)]
# print result_sum[k] - np.delete(result_sum, k).max()

# Q_sum_1 = np.zeros_like(Q_sum)
# for i in range(N, N_comb*2):
#     if W[i-N] > 0.0:
#         Q_sum_1 += Q[i, :]
#     else:
#         Q_sum_1 -= Q[i, :]

# print np.asarray([Q_sum_1.dot(pol_basis.real[:, i]) for i in range(N)])
# plt.figure()
# plt.subplot(211)
# plt.plot(comb_basis.dot(Q_sum))
# plt.subplot(212)
# plt.plot(comb_basis.dot(Q_sum_1))
# plt.show()


plt.figure()
plt.plot(field1.sum(axis=(1, 2)), 'y')
plt.plot(field2.sum(axis=(1, 2)), 'g')
plt.plot(comb_basis.sum(axis=(1, 2)), 'c')
plt.plot(pol_matrix[:, 0], 'k')
plt.plot(pol_matrix[:, 1], 'b')
plt.plot(pol_matrix[:, 2], 'r')
plt.show()

# plt.figure()
# color_codes = ['r', 'g', 'b', 'k']

# for i in range(N):
#     plt.subplot(221+i)
#     plt.plot(freq, pol_matrix[:, i], color_codes[i])
#     plt.plot(freq, comb_basis.sum(axis=1), 'k')

# plt.figure()
# for i in range(4):
#     plt.subplot(221+i)
#     [plt.plot(freq, pol_basis[j, i]*comb_basis[:, j]) for j in range(2*N_comb)]
#
# pol_N_minus_1 = np.delete(pol_basis, 0, 1)
#
# Q, R = np.linalg.qr(pol_N_minus_1.real, mode="complete")
#
# sum_comb = np.empty(N_freq)
#
# for i in range(4, 40):
#     sum_comb += comb_basis.dot(Q[:, i])
#
# plt.figure()
# plt.subplot(211)
# plt.plot(comb_basis.dot(Q[:, 0]))
# plt.subplot(212)
# plt.plot(sum_comb, 'r')
# plt.show()