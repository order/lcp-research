#ifndef __BINDINGS_INCLUDED__
#define __BINDINGS_INCLUDED__

#include <boost/python.hpp>
#include <armadillo>

#include "simulate.h"

typedef boost::python::object Object;

typedef long long unsigned int ullint;

//===========================================================
// EXPORT
// To python from C++ objects
Object export_cube(const arma::cube & v);
Object export_mat(const arma::mat & A);
Object export_vec(const arma::vec & v);
Object export_base(uint dim, npy_intp * shape, uint n_elem,
		   const double * data);

// Export struct as 
Object export_sim_results(const SimulationOutcome & res);

//===========================================================
// Import
// From python to C++ objects
arma::mat import_mat(PyObject * Obj);
arma::vec import_vec(PyObject * Obj);
arma::uvec import_uvec(PyObject * Obj);

//===========================================================
// Basic function interpolation
Object interpolate(PyObject * py_val,
		   PyObject * py_points,
		   PyObject * py_low,
		   PyObject * py_high,
		   PyObject * py_num_cells);

// Flesh this out more
Object simulate();

// Test function.
Object incr(PyObject * Obj);

//============================================================
// NB: BOOST_PYTHON_MODULE in .cpp

#endif
