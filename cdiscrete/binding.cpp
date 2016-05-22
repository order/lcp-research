
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

  return export_double_base(dim, shape, n_elem, (double *) C.memptr());
}

Object export_mat(const mat & A) {
  
  uint n_elem = A.n_elem;
  uint dim = 2;
  npy_intp shape[2] = {(npy_intp) A.n_rows,
		       (npy_intp) A.n_cols };

  return export_double_base(dim, shape, n_elem, (double *) A.memptr());
}

Object export_vec(const vec & v) {
  
  uint n_elem = v.n_elem;
  uint dim = 1;
  npy_intp shape[1] = { (npy_intp) v.n_elem }; // array size

  return export_double_base(dim, shape, n_elem, (double *) v.memptr());
}

Object export_double_base(uint dim,
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

Object export_uvec(const uvec & v) {
  
  uint n_elem = v.n_elem;
  uint dim = 1;
  npy_intp shape[1] = { (npy_intp) v.n_elem }; // array size

  return export_ullint_base(dim, shape, n_elem, (ullint *) v.memptr());
}

Object export_ullint_base(uint dim,
			  npy_intp * shape,
			  uint n_elem,
			  const ullint * data){
  // Create a ndarray with new memory
  PyObject* obj = PyArray_New(&PyArray_Type,
			      dim, shape, // dimension and shape
			      NPY_ULONGLONG, // data type
			      NULL, // strides; uses shape and type
			      NULL, // data pointer; create new mem
			      0, // item size; defaults to using type
			      NPY_ARRAY_CARRAY, 
			      NULL);

  
  // Size checking
  assert(sizeof(ullint) == (npy_intp) PyArray_ITEMSIZE((PyArrayObject*)obj));
  assert(n_elem == PyArray_SIZE((PyArrayObject*)obj));
  size_t sz = sizeof(ullint) * n_elem;
  assert(sz == (size_t) PyArray_NBYTES((PyArrayObject*)obj));
  
  // Memcopy all the data over
  ullint * dest = (ullint *) PyArray_DATA((PyArrayObject*)obj);
  memcpy(dest,data,sz);
    
  bp::handle<> array( obj );
  return bp::object(array);   
}

Object export_sim_results(const SimulationOutcome & res){
  Object points = export_cube(res.points);
  Object actions = export_cube(res.actions);
  Object costs = export_mat(res.costs);

  return make_tuple<Object,Object,Object>(points,actions,costs);
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

Object argmax_interpolate(PyObject * py_vals,
			  PyObject * py_points,
			  PyObject * py_low,
			  PyObject * py_high,
			  PyObject * py_num_cells){
  mat vals    = import_mat(py_vals);
  mat points = import_mat(py_points);
  
  RegGrid grid;

  grid.low      = import_vec(py_low);
  grid.high     = import_vec(py_high);
  grid.num_cells = import_uvec(py_num_cells);
  
  uvec I = max_interp_fns(vals,points,grid);
  return export_uvec(I);
}


//================================================
// SIMULATION

Object simulate_test_export(){
  SimulationOutcome res;
  simulate_test(res);
  return export_sim_results(res);
}

//===============================================
// DEBUG
Object c_arange(uint n){
  uvec v = uvec(n);
  for(uint i = 0; i < n; i++){v(i) = i;}
  return export_uvec(v);
}

//=====================================
BOOST_PYTHON_MODULE(cDiscrete){
  /*
    Python side: 
   */
  import_array();

  /*
    Should be called from routines in "cdiscrete_wrapper.py"
    which will do type checking on the python side.

    Basic bevahior:
    1) Call using PyObjects (usually wrapping numpy.ndarray)
    2) Convert to C++ objects using "import_*" functions
    3) Call relavent C++ code
    4) Convert results to boost::python::object with "export_*"
   */
  
  bp::def ("interpolate",interpolate);
  bp::def ("simulate_test",simulate_test_export);
  bp::def ("c_arange",c_arange);
  bp::def ("argmax_interpolate", argmax_interpolate);
}
