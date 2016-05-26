#include <assert.h>

#include "misc.h"
#include "transfer.h"

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

DoubleIntegrator::DoubleIntegrator(double step_size,
				   uint num_steps,
				   double damping,
				   double jitter){
  _step_size = step_size;
  _num_steps = num_steps;
  _damping = damping;
  _jitter = jitter;
}

mat DoubleIntegrator::get_next_states(const mat & points,
				      const mat & actions) const{
  // Points are (N,D), actions (N,1)
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(2 == D);
  assert(N == actions.n_rows);
  assert(1 == actions.n_cols);

  /* Dynamics
   Tz = [1    h][x] = [x + hv]
        [0  1-d][v]   [(1-d)v] <- damped
   */
  mat Tt = mat(2,2); // <- Defining the transpose
  Tt(0,0) = 1;
  Tt(1,0) = _step_size;
  Tt(0,1) = 0;
  Tt(1,1) = 1 - _damping;

  /* Effect of action on dynamics
     U = [1/2 h^2; h]
  */
  mat Ut = mat(1,2); // <- Transpose
  Ut(0,0) = 0.5 * _step_size * _step_size;
  Ut(0,1) = _step_size;

  mat X = points;
  for(uint t = 0; t < _num_steps; t++){
    mat noise = randn<mat>(N,1);
    X = X * Tt + (actions + _jitter * noise) * Ut; // TODO: jitter
  }
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
