#include <iostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <functional> // for declaring lambda functions
#include <typeinfo> // for learning purposes

// Define namespaces
namespace py = pybind11;
namespace ode = boost::numeric::odeint;
using std::vector;

// // Define the state type
// typedef std::vector<double> state_type;


/**
 * Calculates the time derivatives of the FitzHugh-Nagumo model equations for a single system.
 *
 * This function computes the time derivatives of the FitzHugh-Nagumo model equations
 * given the current state vector `x`, model parameters `params`, and stores the results
 * in the output vector `dxdt`.
 *
 * @param x The current state vector containing the variables x[0] and x[1].
 * @param params The model parameters, where params[0] represents 'a' and params[1] represents 'eps'.
 * @param dxdt The output vector to store the computed time derivatives.
 */
void fhn_eom(vector<double> &x, vector<double> &dxdt, const double a, const double eps) {
    dxdt[0] = (x[0] - (x[0] * x[0] * x[0]) / 3 - x[1]) / eps;
    dxdt[1] = x[0] + a;
}

// Define the coupling matrix function
/**
 * Calculates the B matrix for a given angle phi and epsilon value.
 *
 * @param phi The angle in radians.
 * @param eps The epsilon value.
 * @return The B matrix as a 2D vector.
 */
vector<vector<double>> bmatrix(double phi, double eps) {
    vector<vector<double>> b = {{-cos(phi) / eps, sin(phi) / eps}, {-sin(phi), cos(phi)}};
    return b;
}


// Define the coupled FHN equations
/**
 * Calculates the coupled FitzHugh-Nagumo equations of motion for a network of neurons.
 *
 * @param x The state vector of the neurons.
 * @param dxdt The derivative of the state vector.
 * @param a The parameter 'a' in the FitzHugh-Nagumo equations.
 * @param eps The parameter 'eps' in the FitzHugh-Nagumo equations.
 * @param coupling_strength The strength of the coupling between neurons.
 * @param coupling_matrix The coupling matrix specifying the connectivity between neurons.
 * @param b A matrix used for intermediate calculations.
 * @param N The number of neurons in the network.
 */
void coupled_fhn_eom(const vector<double> &x, vector<double> &dxdt, const double a, const double eps,
                     const double coupling_strength, const vector<vector<double>> &coupling_matrix,
                     const vector<vector<double>> &b, const size_t N) {

    vector<vector<double>> eachneuron(2, vector<double>(N));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            eachneuron[j][i] = x[2 * i + j];
        }
    }
    
    vector<vector<double>> coupling_terms(N, vector<double>(2));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t r = 0; r < N; ++r) {
                coupling_terms[i][j] += coupling_matrix[i][r] * eachneuron[j][r];
            }
        }
    }
    
    for (size_t i = 0; i < N; ++i) {
        vector<double> thisneuron(2);
        vector<double> thiscoupling(2);
        for (size_t j = 0; j < 2; ++j) {
            thisneuron[j] = eachneuron[j][i];
            thiscoupling[j] = coupling_terms[i][j];
        }
        
        vector<double> dxdt_neuron(2); // This is an unnecessary allocation.
        fhn_eom(thisneuron, dxdt_neuron, a, eps);
        for (size_t j = 0; j < 2; ++j) {
            dxdt[2 * i + j] = dxdt_neuron[j] + coupling_strength * thiscoupling[j];
        }
    }
}

// Define the function to solve the coupled FHN equations
double solve_coupled_fhn(const double a, const double eps, const double phi, const double coupling_strength,
                         const double t_final, const py::array_t<double> &x0_python, const py::array_t<double> &coupling_matrix_python) {
    // Extract the system length and number of neurons                        
    size_t system_length = x0_python.shape(0);
    size_t N = system_length / 2;

    // Extract the data from the input arrays
    auto coupling_matrix_reference = coupling_matrix_python.unchecked<2>(); // Provides a reference to the underlying data (managed by python)
    vector<vector<double>> coupling_matrix(N, vector<double>(N));
    for (py::ssize_t i = 0; i < coupling_matrix_reference.shape(0); i++) {
        for (py::ssize_t j = 0; j < coupling_matrix_reference.shape(1); j++) {
            coupling_matrix[i][j] = coupling_matrix_reference(i, j);  // Access the element at position (i, j)
        }
    }
    
    // Define the B matrix
    vector<vector<double>> b = bmatrix(phi, eps);

    // Define the state vector
    vector<double> x(system_length);
    for (size_t i = 0; i < N; ++i) {
        x[2 * i] = x0_python.data()[2 * i];
        x[2 * i + 1] = x0_python.data()[2 * i + 1];
    }

    // Define the shortened coupled FHN equations of motion to feed into the integrator

    std::function<void( const vector<double>&, vector<double>&, double)> shortened_coupled_fhn_eom = [&](const vector<double> &x, vector<double> &dxdt, double t) {
        coupled_fhn_eom(x, dxdt, a, eps, coupling_strength, coupling_matrix, b, N);
    };

    // // Define the integrator (use adaptive stepper)
    // ode::runge_kutta_dopri5<std::vector<double>> stepper;
    
    // Integrate the system with adaptive timestep
    size_t steps = ode::integrate(shortened_coupled_fhn_eom, x, 0.0, t_final, 0.01);

    // Return the results as a numpy array
    
}

PYBIND11_MODULE(javier, m) {
    m.def("solve_coupled_fhn", &solve_coupled_fhn, "Solve coupled FHN equations with adaptive timestep");
}