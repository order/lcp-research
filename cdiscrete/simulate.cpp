#include <iostream>
#include <assert.h>
#include <armadillo>

#include "simulate.h"

using namespace arma;

//====================================================
// Bang-bang policy for 2D (x,v) double integrator

DIBangBangPolicy::DIBangBangPolicy(const mat & actions){
  // Assume the action matrix is A x 1, and is
  // sorted from most negative (0) to most positive (last)
  assert(1 == actions.n_cols);
  _actions = actions;
  _n_actions = actions.n_rows;
}

uvec DIBangBangPolicy::get_action_indices(const mat & points){
  // Gets the indices of the correct action
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == 2);

  vec x = points.col(0);
  vec v = points.col(1);

  // Decision boundary
  vec dec_fn = v + sign(x) * sqrt(2 * abs(x));
  
  // Assumes 0 is index of most negative,
  // and _n_actions is the index of most positive
  uvec a_idx = uvec(N);
  a_idx(dec_fn > 0).fill(0);
  a_idx(dec_fn < 0).fill(_n_actions);
  
  return a_idx;
}

vec DIBangBangPolicy::get_actions(const mat & points){
  // Converts indicies into actual acceleration number
  // Using provided action list
  uvec a_idx = get_action_indices(points);
  return _actions.rows(a_idx);
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
				      const mat & actions){
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
    X = X * Tt + actions * Ut; // TODO: jitter
  }
  return X;
}
