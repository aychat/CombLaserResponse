import numpy as np
import matplotlib.pyplot as plt

N = 2

# w1 = 2.354
# w2 = 4.708

w1 = 2.4149
w2 = 2.4273
molecules = [
    dict(
        w_in=w1 + np.random.uniform(1.0, 1.5)*1e-6,
        g_in=np.random.uniform(0.0, 5.0)*1e-5,
        w_out=w2 + np.random.uniform(1.5, 2.0)*1e-6,
        g_out=np.random.uniform(0.0, 6.0)*1e-5,
    ) for _ in range(N)
]


omegaM1 = w1 + 7e-8
omegaM2 = w1 + 1e-8
gamma = 1.5e-9

N_freq = 1000
freq = np.linspace(2.*w1-4.5e-6, 2.*w1+4.5e-6, N_freq)

omega_del1 = 12e-8
omega_del2 = 12e-8

N_comb = 25
del_omega1 = omega_del1 * np.arange(-N_comb, N_comb)
del_omega2 = omega_del2 * np.arange(-N_comb, N_comb)

omega = freq[:, np.newaxis]
comb_omega1 = del_omega1[np.newaxis, :]
comb_omega2 = del_omega2[np.newaxis, :]

field1 = gamma / ((omega - omegaM1 - comb_omega1)**2 + gamma**2)
field2 = gamma / ((omega - omegaM2 - comb_omega2)**2 + gamma**2)

########################################################################################################################
#                                                                                                                      #
#         ----------------- SECOND ORDER POLARIZATION FUNCTIONS FOR THE OFF-DIAGONAL TERM --------------------         #
#                                                                                                                      #
########################################################################################################################


def calculate_pol_a1_12(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM2 - comb_omega2 - w_in + 1j*(gamma + g_in)
    A2 = omegaM1 + comb_omega1 - w_in + 1j*(gamma + g_in)
    K1 = (omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=1)

########################################################################################################################
#                                                                                                                      #
#                ----------------- PLOT SECOND ORDER POLARIZATION FOR N MOLECULES --------------------                 #
#                                                                                                                      #
########################################################################################################################


plt.figure()
plt.suptitle("$P^{(2)}(\\omega)$ for N molecules")

pol_matrix = np.asarray([calculate_pol_a1_12(**m) for m in molecules]).T*1e-23


for i in range(N):
    plt.subplot(211+i)
    plt.plot(omega.reshape(-1)*1e6, 1e16*pol_matrix[:, i].real, 'k', label='Molecule %d' % (i+1))
    plt.plot(omega.reshape(-1)*1e6, field1.sum(axis=1), 'b')
    plt.plot(omega.reshape(-1)*1e6, field2.sum(axis=1), 'g')
    plt.legend()
    plt.grid()
    plt.xlabel("freq (in GHz)")
    plt.ylabel("Polarization (arbitrary units)")

plt.show()

U_mat, S, V_mat = np.linalg.svd(pol_matrix, full_matrices=True)
print S
np.set_printoptions(precision=4)

# print [[np.vdot(U_mat[i], U_mat[j]) for j in range(N)] for i in range(N)]

plt.figure()
plt.semilogy(S)

detection = U_mat.dot(pol_matrix.conj())[:N, :]
print "determinant =", np.abs(np.linalg.det(detection))

det = [
    np.linalg.det(
        np.asarray([U_mat[j].dot(pol_matrix.conj()) for j in np.random.randint(N, N_freq, size=N)])
    ) for _ in range(10000)
]
print "max null-space determinant =", np.abs(det).max()

x = np.arange(1, N+1)
fig = plt.figure()
ax = fig.add_subplot(111)
[ax.bar(x+0.5-float(j)/(N+1), detection.T[j], 1./(N+1), align='center') for j in range(N)]
ax.set_xticklabels(["Field " + str(i) for i in x])

# print np.abs(np.asarray([U_mat[j].dot(pol_matrix).real*1e-23 for j in range(N, N_freq)])).max()
# plt.show()
