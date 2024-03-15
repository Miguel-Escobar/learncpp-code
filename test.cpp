#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> my_function() {
    // Your C++ code here to generate the numpy array
    // For demonstration, let's create a simple array
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Create a numpy array from the data
    py::array_t<double> result(data.size());
    auto buffer = result.request();
    double *ptr = static_cast<double *>(buffer.ptr);
    std::copy(data.begin(), data.end(), ptr);
    
    return result;
}

PYBIND11_MODULE(my_module, m) {
    m.def("my_function", &my_function);
}
