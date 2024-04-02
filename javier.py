import javier
import numpy as np
import matplotlib.pyplot as plt

a = 0.5
eps = 0.05
N=40
x0 = np.zeros(2*N) + np.random.uniform(-0.1, 0.1, 2*N)
coupling_matrix = np.zeros((N, N)) + np.random.uniform(0, 1.1, (N, N))
t, x = javier.solve_coupled_fhn(a, eps, 0.5, 3.0, 1000.0, 0.01, x0, coupling_matrix - np.diag(np.sum(coupling_matrix, axis=1)))

print(t[1:50])
plt.clf()
plt.plot(t, x[:, 0], label='x1')
plt.plot(t, x[:, 2], label='x2')
plt.savefig('x1.png')