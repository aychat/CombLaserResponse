import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
with open("Polarization_order3_data.pickle", "rb") as input_file:
    data = pickle.load(input_file)

freq = data['frequency']
P_f = data['molecules_pol3'].T
field1 = data['field1']
field2 = data['field2']
field3 = data['field3']

P_f /= P_f.max()

plt.figure()
factor = (np.abs(P_f)).max()/field1.max()
plt.subplot(211)
plt.plot(P_f[:, 0].real/factor, 'k')
plt.plot(field1, alpha=0.5)
plt.plot(field2, alpha=0.5)
plt.plot(field3, alpha=0.5)
plt.subplot(212)
plt.plot(P_f[:, 0].imag/factor, 'k')
plt.plot(field1, alpha=0.5)
plt.plot(field2, alpha=0.5)
plt.plot(field3, alpha=0.5)

omega_cb = freq[:, np.newaxis]
del_omega1_cb = data['params']['omega_del_1'] * np.arange(-2*data['params']['N_comb'], 2*data['params']['N_comb'])
comb_omega1_cb = del_omega1_cb[np.newaxis, :]

CB = data['params']['gamma'] / ((omega_cb - data['params']['omega_M1'] - data['params']['omega_M2'] - data['params']['omega_M3'] - comb_omega1_cb) ** 2 + data['params']['gamma'] ** 2)
P_c = CB.T.dot(P_f)


def gaussian(x, x_0, sigma):
    return (1. / (sigma * np.sqrt(np.pi))) * np.exp(-(x - x_0) ** 2 / (2 * sigma ** 2))


colors1 = cm.Reds(np.linspace(0.5, 1, 3))
colors2 = cm.Blues(np.linspace(0.5, 1, 3))

#  --------------------------- PLOT HETERODYNE FIELDS  ----------------------------------------------#

# f, ax = plt.subplots(2, 3, sharex=True)
# f.suptitle(
#     "$3^{rd}$ order non-linear polarization $P^{(3)}(\\omega)$ and corresponding \n heterodyne fields $E^{het}(\\omega)$ for 3 near-identical atomic systems")
#
# for i in range(data['params']['N_molecules']):
#     Q_c, R_c = np.linalg.qr(np.delete(P_c, i, 1), mode='complete')
#     het_fields = Q_c[:, data['params']['N_molecules']:]
#     print Q_c.shape
#     pol3_ = CB.dot(P_c)
#     pol3_ /= pol3_.max()
#     ax[0, i].plot(freq, pol3_[:, i], color=colors1[i])
#     G = gaussian(np.linspace(0., 1., het_fields.shape[1]), 0.5, .035)
#     heterodyne = CB.dot(np.asarray([G[j] * het_fields[:, j] for j in range(het_fields.shape[1])]).sum(axis=0))
#     heterodyne /= heterodyne.max()
#     ax[1, i].plot(freq, heterodyne, color=colors2[i])
#     ax[1, i].set_xlabel("Frequency (in $fs$)")
#     # print [heterodyne.dot(P_f[:, i]) for i in range(3)]
#     print [heterodyne.dot(P_f[:, i]) for i in range(3)]
#
# ax[0, 0].set_ylabel("Normalized Polarization \n $P^{(3)}(\\omega)$")
# ax[1, 0].set_ylabel("Heterodyne fields \n $E^{het}(\\omega)$")
#
# for i in range(2):
#     for j in range(3):
#         for label in ax[i, j].get_xmajorticklabels() + ax[i, j].get_ymajorticklabels():
#             label.set_rotation(30)

# plt.figure()
# plt.plot(G)
plt.show()