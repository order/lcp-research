#ifndef __BINDINGS_INCLUDED__
#define __BINDINGS_INCLUDED__

#include <boost/python.hpp>
#include <armadillo>

typedef boost::python::object Object;

typedef long long unsigned int ullint;

Object export_mat(arma::mat & A);
Object export_vec(arma::vec & v);
//Object export_vec(arma::vec & v);

arma::mat import_mat(PyObject * Obj);
arma::vec import_vec(PyObject * Obj);
arma::uvec import_uvec(PyObject * Obj);

// Basic function interpolation
Object interpolate(PyObject * py_val,
		   PyObject * py_points,
		   PyObject * py_low,
		   PyObject * py_high,
		   PyObject * py_num_cells);

// Test function.
Object incr(PyObject * Obj);

#endif
