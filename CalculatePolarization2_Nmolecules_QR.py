import numpy as np
import matplotlib.pyplot as plt

N = 6

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

N_freq = 500
freq = np.linspace(2.*2.354-4e-6, 2.*2.354+4e-6, N_freq)

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


def calculate_pol_a1_12(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM2 - comb_omega2 - w_in + 1j*(gamma + g_in)
    A2 = omegaM1 + comb_omega1 - w_in + 1j*(gamma + g_in)
    K1 = (omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))

########################################################################################################################
#                                                                                                                      #
#                ----------------- PLOT SECOND ORDER POLARIZATION FOR N MOLECULES --------------------                 #
#                                                                                                                      #
########################################################################################################################


plt.figure()
plt.suptitle("$P^{(2)}(\\omega)$ for N molecules")

pol_matrix = np.asarray([calculate_pol_a1_12(**m) for m in molecules]).T*1e-19
pol_matrix = pol_matrix.real

for i in range(N):
    plt.plot(omega.reshape(-1)*1e6, pol_matrix[:, i].real, label='Molecule %d' % (i+1))
    plt.legend()
    plt.grid()
    plt.xlabel("freq (in GHz)")
    plt.ylabel("Polarization (arbitrary units)")

# plt.show()

np.set_printoptions(precision=5, suppress=True)
# for i in range(N):
Q, R = np.linalg.qr(np.delete(pol_matrix, 0, 1), mode="complete")
detect = np.abs(np.asarray([[pol_matrix[:, k].dot(Q[:, j]) for k in range(N)] for j in range(N, N+4)]))
print detect[detect[:, 0].argmax()]
plt.figure()
for i in range(4):
    plt.subplot(221+i)
    plt.plot(Q[:, N+i], 'r')
plt.show()