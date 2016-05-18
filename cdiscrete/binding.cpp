#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include <assert.h>

#include "binding.h"
#include "discrete.h"

namespace bp = boost::python;
using namespace arma;

//=======================================
// EXPORTING

Object export_mat(mat & A){
  npy_intp N = (npy_intp)A.n_rows;
  npy_intp D = (npy_intp)A.n_cols;
  npy_intp shape[2] = { N, D }; // array size

  // Create a ndarray with new memory
  // 
  PyObject* obj = PyArray_New(&PyArray_Type,
			      2, shape, // dimension and shape
			      NPY_DOUBLE, // data type
			      NULL, // strides;
			      NULL, // data pointer
			      0, // item size; information in data type usually
			      NPY_ARRAY_CARRAY, 
			      NULL); // Not sure

  // Get data pointers
  double * dest = (double *) PyArray_DATA((PyArrayObject*)obj);
  double * data = (double *) A.memptr();
  
  // Careful checking of types and sizes
  assert(sizeof(double) == (npy_intp) PyArray_ITEMSIZE((PyArrayObject*)obj));
  assert(N*D == PyArray_SIZE((PyArrayObject*)obj));
  size_t sz = sizeof(double) * N * D;
  assert(sz == (size_t) PyArray_NBYTES((PyArrayObject*)obj));

  // MEMCOPY
  memcpy(dest,data,sz);

  // Not really sure what this does...
  bp::handle<> array( obj );
  return bp::object(array);  
}

Object export_vec(vec & v) {
  npy_intp N = (npy_intp)v.n_elem;  
  npy_intp shape[1] = { N }; // array size

  // Create a ndarray with new memory
  // 
  PyObject* obj = PyArray_New(&PyArray_Type,
			      1, shape, // dimension and shape
			      NPY_DOUBLE, // data type
			      NULL, // strides;
			      NULL, // data pointer
			      0, // item size; information in data type usually
			      NPY_ARRAY_CARRAY, 
			      NULL); // Not sure

  // Memcopy all the data over
  // (Best way of doing this?)
  double * dest = (double *) PyArray_DATA((PyArrayObject*)obj);
  double * data = (double *) v.memptr();
  // More careful checking
  assert(sizeof(double) == (npy_intp) PyArray_ITEMSIZE((PyArrayObject*)obj));
  assert(N == PyArray_SIZE((PyArrayObject*)obj));
  size_t sz = sizeof(double) * N;
  assert(sz == (size_t) PyArray_NBYTES((PyArrayObject*)obj));
  memcpy(dest,data,sz);
    
  bp::handle<> array( obj );
  return bp::object(array);  
}

//==================================================
// IMPORTING

mat import_mat(PyObject * Obj){
  // Imports a matrix
  assert(PyArray_Check(Obj));
  assert(2 == PyArray_NDIM((PyArrayObject*)Obj)); // Is a matrix
  assert(NPY_DOUBLE == PyArray_TYPE((PyArrayObject*)Obj)); // Double

  // Weird; the matrix is transposed coming in; but works
  // as you'd expect during export...
  return mat((const double *)PyArray_DATA((PyArrayObject*)Obj),
	     PyArray_DIM((PyArrayObject*)Obj,1),
	     PyArray_DIM((PyArrayObject*)Obj,0)).t();
  
}

vec import_vec(PyObject * Obj){
  // Imports an vector
  assert(PyArray_Check(Obj));
  assert(1 == PyArray_NDIM((PyArrayObject*)Obj)); // Is a vector
  assert(NPY_DOUBLE == PyArray_TYPE((PyArrayObject*)Obj)); // Double
  
  return vec((const double *)PyArray_DATA((PyArrayObject*)Obj),
	     PyArray_DIM((PyArrayObject*)Obj,0)); // n_rows
  
}

uvec import_uvec(PyObject * Obj){
  // Imports an vector
  assert(PyArray_Check(Obj));
  assert(1 == PyArray_NDIM((PyArrayObject*)Obj)); // Is a vector
  assert(NPY_UINT64 == PyArray_TYPE((PyArrayObject*)Obj)); // Double

  assert(sizeof(ullint) == PyArray_ITEMSIZE((PyArrayObject*)Obj));
  
  return uvec((const ullint *)PyArray_DATA((PyArrayObject*)Obj),
	      PyArray_DIM((PyArrayObject*)Obj,0));
  
}

//================================================
// INTERPOLATE
// Basic interpolation
Object interpolate(PyObject * py_val,
		   PyObject * py_points,
		   PyObject * py_low,
		   PyObject * py_high,
		   PyObject * py_num_cells){
  vec val    = import_vec(py_val);
  mat points = import_mat(py_points);
  
  RegGrid grid;

  grid.low      = import_vec(py_low);
  grid.high     = import_vec(py_high);
  grid.num_cells = import_uvec(py_num_cells);
  
  vec I = interp_fn(val,points,grid);
  return export_vec(I);
}

// Simple debugging function
Object incr(PyObject * Obj){
  mat A = import_mat(Obj);
  A = A+1;
  return export_mat(A);
}

//=====================================
BOOST_PYTHON_MODULE(cDiscrete){
  /*
    Python side: 
   */
  import_array(); // APPARENTLY YOU NEED THIS >:L

  // Create a vector of uints using list operations
  bp::def ("export_mat",export_mat);
  bp::def ("export_vec",export_vec);

  bp::def ("import_mat",import_mat);
  bp::def ("import_vec",import_vec);
  bp::def ("import_uvec",import_uvec);

  bp::def ("interpolate",interpolate);
  
}
