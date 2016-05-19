#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include <assert.h>

#include "binding.h"
#include "discrete.h"

namespace bp = boost::python;
using namespace arma;

//=======================================
// EXPORTING

Object export_cube(const cube & C) {
  
  uint n_elem = C.n_elem;
  uint dim = 3;
  npy_intp shape[3] = { (npy_intp) C.n_rows,
			(npy_intp) C.n_cols,
			(npy_intp) C.n_slices };

  return export_base(dim, shape, n_elem, (double *) C.memptr());
}

Object export_mat(const mat & A) {
  
  uint n_elem = A.n_elem;
  uint dim = 2;
  npy_intp shape[2] = {(npy_intp) A.n_rows,
		       (npy_intp) A.n_cols };

  return export_base(dim, shape, n_elem, (double *) A.memptr());
}

Object export_vec(const vec & v) {
  
  uint n_elem = v.n_elem;
  uint dim = 1;
  npy_intp shape[1] = { (npy_intp) v.n_elem }; // array size

  return export_base(dim, shape, n_elem, (double *) v.memptr());
}

Object export_base(uint dim,
		   npy_intp * shape,
		   uint n_elem,
		   const double * data){
  // Create a ndarray with new memory
  PyObject* obj = PyArray_New(&PyArray_Type,
			      dim, shape, // dimension and shape
			      NPY_DOUBLE, // data type
			      NULL, // strides; uses shape and type
			      NULL, // data pointer; create new mem
			      0, // item size; defaults to using type
			      NPY_ARRAY_CARRAY, 
			      NULL);

  
  // Size checking
  assert(sizeof(double) == (npy_intp) PyArray_ITEMSIZE((PyArrayObject*)obj));
  assert(n_elem == PyArray_SIZE((PyArrayObject*)obj));
  size_t sz = sizeof(double) * n_elem;
  assert(sz == (size_t) PyArray_NBYTES((PyArrayObject*)obj));
  
  // Memcopy all the data over
  double * dest = (double *) PyArray_DATA((PyArrayObject*)obj);
  memcpy(dest,data,sz);
    
  bp::handle<> array( obj );
  return bp::object(array);   
}

Object export_sim_results(const SimulationOutcome & res){
  Object points = export_cube(res.points);
  Object actions = export_cube(res.actions);
  Object costs = export_mat(res.costs);

  return make_tuple(points,actions,costs);
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

Object simulate(){
  SimulationOutcome res;
  uint T = 100;
  
  mat x0 = randn<mat>(1,2);
  mat actions = mat("-1;0;1");
  TransferFunction f = DoubleIntegrator(0.01,5,1e-5,0);
  CostFunction c = BallCosts(0.15,zeros<vec>(2));
  Policy p = DIBangBangPolicy(actions);
  
  simulate(x0,f,c,p,T,res);

  return export_sim_results(res);
  
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
  bp::def ("export_cube",export_cube);  
  bp::def ("export_mat",export_mat);
  bp::def ("export_vec",export_vec);

  // Figure out how to export structures properly
  // Search for "classes" and "boost python converters"
  bp::def ("export_sim_results",export_sim_results);


  bp::def ("import_mat",import_mat);
  bp::def ("import_vec",import_vec);
  bp::def ("import_uvec",import_uvec);

  bp::def ("interpolate",interpolate);
  
}
