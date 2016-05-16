#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include <boost/python.hpp>
#include <armadillo>

namespace bp = boost::python;

typedef bp::object Object;

Object export_vec(arma::vec & v) {
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
  memcpy(dest,data,sizeof(double) * N);

  for(uint i = 0; i < N; i++){
    std::cout << *(double *)PyArray_GETPTR1((PyArrayObject*)obj,i) << std::endl;
  }
    
  bp::handle<> array( obj );
  return bp::object(array);  
}

arma::vec import_vec(PyObject * Obj){
  // Imports an nd
  assert(PyArray_Check(Obj));
  assert(1 == PyArray_NDIM((PyArrayObject*)Obj)); // Is a vector
  return arma::vec((double *)PyArray_DATA((PyArrayObject*)Obj), // data pointer
		   PyArray_DIM((PyArrayObject*)Obj,0), // n_rows
		   true, // Copy (safer...)
		   false);
  
}

Object incr(PyObject * Obj){
  arma::vec v = import_vec(Obj);
  std::cout << v << std::endl;
  v = v+1;
  return export_vec(v);
}


//=====================================
BOOST_PYTHON_MODULE(cDiscrete){
  /*
    Python side: 
   */
  import_array(); // APPARENTLY YOU NEED THIS >:L

  // Create a vector of uints using list operations
  bp::def ("incr",incr); 
}
