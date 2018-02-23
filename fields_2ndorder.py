import numpy as np
import matplotlib.pyplot as plt

w1 = 2.354
omegaM1 = w1 + 15e-8
omegaM2 = w1 + 25e-8
gamma = 2.5e-9

N_freq = 1000
freq_ord1 = np.linspace(w1-3.8e-6, w1+3.8e-6, N_freq)
freq = np.linspace(2.*w1-7.6e-6, 2.*w1+7.6e-6, N_freq)

omega_del1 = 2e-7
omega_del2 = 2e-7

N_comb = 25
del_omega1 = omega_del1 * np.arange(-N_comb, N_comb)
del_omega2 = omega_del2 * np.arange(-N_comb, N_comb)

omega = freq[:, np.newaxis, np.newaxis]
omega1 = freq_ord1[:, np.newaxis, np.newaxis]
comb_omega1 = del_omega1[np.newaxis, :, np.newaxis]
comb_omega2 = del_omega2[np.newaxis, np.newaxis, :]

field1 = gamma / ((omega1 - omegaM1 - comb_omega1)**2 + gamma**2)
field2 = gamma / ((omega1 - omegaM2 - comb_omega2)**2 + gamma**2)

field_order2 = 2.*np.pi*gamma /((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4*gamma**2)

plt.figure()
plt.subplot(211)
plt.plot(freq_ord1, field1.sum(axis=(1,2)), 'b')
plt.plot(freq_ord1, field2.sum(axis=(1,2)), 'r')
plt.subplot(212)
plt.plot(freq, field_order2.sum(axis=(1,2)), 'k')
plt.show()