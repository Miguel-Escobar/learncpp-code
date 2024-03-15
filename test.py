import numpy as np
import my_module

# Call the C++ function
result_array = my_module.my_function()

print("Result array:" , result_array * np.pi)