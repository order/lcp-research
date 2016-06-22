#include <assert.h>

#include "misc.h"
#include "transfer.h"

#include <boost/math/special_functions/sign.hpp>

TransferFunction::~TransferFunction(){}

DoubleIntegrator::DoubleIntegrator(double step_size,
				   uint num_steps,
				   double dampening,
				   double jitter){
  _step_size = step_size;
  _sss = 0.5*step_size * step_size;
  _num_steps = num_steps;
  _dampening = dampening;
  _jitter = jitter;
}

/*
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
    V *= (1.0 - _dampening);
    V += _step_size * pert_acts;   
  }
  mat R = mat(N,2);
  R.col(0) = X;
  R.col(1) = V;
  return R;
  }*/
vec DoubleIntegrator::get_next_state(const vec & point,
				     const vec & action) const{
  double x = point[0];
  double v = point[1];
  vec noise = vec(1);

  double pert_acts;
  for(uint t = 0; t < _num_steps; t++){
    noise.randn();
    pert_acts = action[0] + _jitter * noise[0];
    x += _step_size * v + _sss*pert_acts;
    v *= (1.0 - _dampening);
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
/*
mat BoundaryEnforcer::get_next_states(const mat & points,
				      const mat & actions) const{  
  mat next_states = _trans_fn_ptr->get_next_states(points,actions);
  uint D = points.n_cols;
  assert(D == _boundary.low.n_elem);
  assert(D == _boundary.high.n_elem);

  row_max_inplace(next_states,_boundary.low.t());
  row_min_inplace(next_states,_boundary.high.t());
 
  return next_states;
  }*/

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


//======================================================
// HILLCAR


HillcarTransferFunction::HillcarTransferFunction(double step_size,
						 double num_steps,
						 double dampening,
						 double jitter){
    _step_size = step_size;
    _sss = 0.5 * step_size * step_size;
    _num_steps = num_steps;
    _dampening = dampening;
    _jitter = jitter;
  }

vec HillcarTransferFunction::get_next_state(const vec & point,
					    const vec & action) const{
  assert(2 == point.n_elem);
  assert(1 == action.n_elem);
  
  double x = point[0];
  double v = point[1];
  vec noise = vec(1);

  double slope;
  double hypo2;
  double accel;
  for(uint t = 0; t < _num_steps; t++){
    noise.randn();
    slope = triangle_slope(x);
    hypo2 = 1.0 + slope*slope;
    accel = ((action[0] + noise[0]) / sqrt(hypo2))
      - (GRAVITY*slope / hypo2);

    x += _step_size * v + _sss*accel;
    v *= (1.0 - _dampening);
    v += _step_size * accel;
  }
  vec X = vec::fixed<2>();
  X[0] = x;
  X[1] = v;
  return X;
}

double triangle_wave(double x, double P, double A){
  x /= P;
  return A * ( 2* abs(2 * (x - floor(x + 0.5))) - 1);
}

double triangle_slope(double x){
  double P = 8.0; // Period
  double A = 1.0; // Amplitude

  double T = 0.05; // Threshold for soft thresh
    
  return soft_threshold(triangle_wave(x - P/4,P,A),T);
}

vec triangle_wave(vec x, double P, double A){
  x /= P;
  return A * ( 2* abs(2 * (x - floor(x + 0.5))) - 1);
}


vec triangle_slope(vec x){
  double P = 8.0; // Period
  double A = 1.0; // Amplitude

  double T = 0.05; // Threshold for soft thresh
    
  return soft_threshold(triangle_wave(x - P/4,P,A),T);
}
