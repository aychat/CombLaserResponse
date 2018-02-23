import numpy as np
import matplotlib.pyplot as plt

N = 10
m = np.linspace(0, N-1, 100)
A = np.random.randint(N, size=201)
Z = np.zeros_like(m)
for i in range(100):
    Z[i] = (np.abs(A - m[i])).sum()

plt.figure()
plt.plot(m, Z)
plt.show()