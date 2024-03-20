import javier
import numpy as np
import matplotlib.pyplot as plt

a = 0.5
eps = 0.05
x0 = np.array([0.1, 0.1, 0.0, 4.0, 10.0, 1.0])
coupling_matrix = np.zeros((3, 3))
t, x = javier.solve_coupled_fhn(a, eps, 0.5, 3.0, 1000.0, 0.1, x0, coupling_matrix)

plt.plot(t, x[:, 0], label='x1')
plt.show()