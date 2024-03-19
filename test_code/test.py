import numpy as np
import hola

# Call the C++ function
result_array = hola.my_function()

print("Result array:" , result_array * np.pi)

