import javier
import numpy as np

a = 0.5
eps = 0.05
x0 = np.array([0.1, 0.1, 0.0, 4.0, 10.0, 1.0])
coupling_matrix = np.zeros((3, 3))
print(javier.solve_coupled_fhn(a, eps, 0.5, 3.0, 10.0, x0, coupling_matrix))

# te quiero mucho <3<3 
# besitos

