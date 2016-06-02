#include <assert.h>

#include "misc.h"
#include "transfer.h"
/*
vec TransferFunction::get_next_state(const vec & point,
				     const vec & action) const{
  mat points = conv_to<mat>::from(point.t());
  mat actions = conv_to<mat>::from(action.t());
  mat r = get_next_states(points,actions);
  assert(1 == r.n_rows);
  return r.row(0).t();
}

mat TransferFunction::get_next_states(const mat & points,
				      const vec & action) const{
  uint N = num_states(points);
  mat actions = repmat(action.t(),N,1);
  return get_next_states(points,actions);
}
*/

DoubleIntegrator::DoubleIntegrator(double step_size,
				   uint num_steps,
				   double damping,
				   double jitter){
  _step_size = step_size;
  _sss = 0.5*step_size * step_size;
  _num_steps = num_steps;
  _damping = damping;
  _jitter = jitter;

  // Dynamics
  _Tt = mat(2,2); // <- Defining the transpose
  _Tt(0,0) = 1;
  _Tt(1,0) = _step_size;
  _Tt(0,1) = 0;
  _Tt(1,1) = 1 - _damping;

  _Ut = mat(1,2); // <- Transpose
  _Ut(0,0) = 0.5 * _step_size * _step_size;
  _Ut(0,1) = _step_size;
}

mat DoubleIntegrator::get_next_states(const mat & points,
				      const mat & actions) const{
  // Points are (N,D), actions (N,1)
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(2 == D);
  assert(N == actions.n_rows);
  assert(1 == actions.n_cols);

  vec X = points.col(0);
  vec V = points.col(1);
  mat noise = mat(N,1);
  vec pert_acts = vec(N);
  for(uint t = 0; t < _num_steps; t++){
    noise.randn();
    pert_acts = actions.col(0) + _jitter * noise;
    X += _step_size * V + _sss*pert_acts;
    V *= (1.0 - _damping);
    V += _step_size * pert_acts;
  }
  mat R = mat(N,2);
  R.col(0) = X;
  R.col(1) = V;
  return R;
}
vec DoubleIntegrator::get_next_state(const vec & point,
				     const vec & action) const{
  double x = point[0];
  double v = point[1];
  mat noise = vec(1);

  double pert_acts;
  for(uint t = 0; t < _num_steps; t++){
    noise.randn();
    pert_acts = action[0] + _jitter * noise[0];
    x += _step_size * v + _sss*pert_acts;
    v *= (1.0 - _damping);
    v += _step_size * pert_acts;
  }
  vec X = vec::fixed<2>();
  X[0] = x;
  X[1] = v;
  return X;
}

BoundaryEnforcer:: BoundaryEnforcer(TransferFunction * trans_fn_ptr,
				    const Boundary & boundary){
  _trans_fn_ptr = trans_fn_ptr;
  _boundary.low = boundary.low;
  _boundary.high = boundary.high;
}

BoundaryEnforcer::BoundaryEnforcer(TransferFunction * trans_fn_ptr,
				   const RegGrid & boundary){
  _trans_fn_ptr = trans_fn_ptr;
  _boundary.low = boundary.low;
  _boundary.high = boundary.high;
}

BoundaryEnforcer::~BoundaryEnforcer(){
  delete _trans_fn_ptr;
}

mat BoundaryEnforcer::get_next_states(const mat & points,
				      const mat & actions) const{  
  mat next_states = _trans_fn_ptr->get_next_states(points,actions);
  uint D = points.n_cols;
  assert(D == _boundary.low.n_elem);
  assert(D == _boundary.high.n_elem);

  row_max_inplace(next_states,_boundary.low.t());
  row_min_inplace(next_states,_boundary.high.t());
 
  return next_states;
}

vec BoundaryEnforcer::get_next_state(const vec & points,
				      const vec & actions) const{  
  vec next_state = _trans_fn_ptr->get_next_state(points,actions);
  uint D = points.n_elem;
  assert(D == _boundary.low.n_elem);
  assert(D == _boundary.high.n_elem);

  
  max_inplace(next_state,_boundary.low);
  min_inplace(next_state,_boundary.high);
 
  return next_state;
}
