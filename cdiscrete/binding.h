#ifndef __BINDINGS_INCLUDED__
#define __BINDINGS_INCLUDED__

#include <boost/python.hpp>
#include <armadillo>

typedef boost::python::object Object;

// Currently only supporting dtype double
Object export_mat(arma::mat & A);
Object export_vec(arma::vec & v);

arma::mat import_mat(PyObject * Obj);
arma::vec import_vec(PyObject * Obj);


#endif
