#include <assert.h>

#include "misc.h"
#include "policy.h"

vec Policy::get_action(const vec & point) const{
  mat points = conv_to<mat>::from(point.t());
  mat actions = get_actions(points);
  assert(1 == actions.n_rows);
  return actions.row(0);
}

uint DiscretePolicy::get_action_index(const vec & point) const{
  mat points = conv_to<mat>::from(point.t());
  uvec a_indices = get_action_indices(points);
  assert(1 == a_indices.n_elem);
  return a_indices(0);
}


//====================================================
// Bang-bang policy for 2D (x,v) double integrator

DIBangBangPolicy::DIBangBangPolicy(const mat & actions){
  // Assume the action matrix is A x 1, and is
  // sorted from most negative (0) to most positive (last)
  assert(1 == actions.n_cols);
  _actions = actions;
  _n_actions = actions.n_rows;
}

uvec DIBangBangPolicy::get_action_indices(const mat & points) const{
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
  if(any(dec_fn>0)){
    a_idx(find(dec_fn > 0)).fill(0);
  }
  if(any(dec_fn<=0)){
    a_idx(find(dec_fn <= 0)).fill(_n_actions-1);
  }

  return a_idx;
}

uint DIBangBangPolicy::get_action_dim() const{
  return 1;
}

mat DIBangBangPolicy::get_actions(const mat & points) const{
  // Converts indicies into actual acceleration number
  // Using provided action list
  uvec a_idx = get_action_indices(points);
  return _actions.rows(a_idx);
}

//====================================================
// Bang-bang policy for 2D (x,v) double integrator

DILinearPolicy::DILinearPolicy(const mat & actions){
  // Assume the action matrix is A x 1, and is
  // sorted from most negative (0) to most positive (last)
  assert(1 == actions.n_cols);
  _actions = actions;
  _n_actions = actions.n_rows;
}

uvec DILinearPolicy::get_action_indices(const mat & points) const{
  // Gets the indices of the correct action
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == 2);

  vec x = points.col(0);
  vec v = points.col(1);

  // Decision boundary
  vec dec_fn = x + v;
  
  // Assumes 0 is index of most negative,
  // and _n_actions is the index of most positive
  uvec a_idx = uvec(N);
  if(any(dec_fn>0)){
    a_idx(find(dec_fn > 0)).fill(0);
  }
  if(any(dec_fn<=0)){
    a_idx(find(dec_fn <= 0)).fill(_n_actions-1);
  }

  return a_idx;
}

uint DILinearPolicy::get_action_dim() const{
  return 1;
}

mat DILinearPolicy::get_actions(const mat & points) const{
  // Converts indicies into actual acceleration number
  // Using provided action list
  uvec a_idx = get_action_indices(points);
  return _actions.rows(a_idx);
}

//====================================================
// Handcrafted policy for 2D (x,v) hillcar

HillcarPolicy::HillcarPolicy(const mat & actions){
  // Assume the action matrix is A x 1, and is
  // sorted from most negative (0) to most positive (last)
  assert(1 == actions.n_cols);
  assert(3 == actions.n_rows);
  assert(actions(0,0) == -actions(2,0));
  assert(0 == actions(1,0));
  
  _actions = actions;
  _n_actions = actions.n_rows;
  _A = 0.9;
  _B = 5.5;
}

uvec HillcarPolicy::get_action_indices(const mat & points) const{
  // Gets the indices of the correct action
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == 2);
  assert(_A < _B);

  vec x = points.col(0);
  vec v = points.col(1);

  uvec a_idx = ones<uvec>(N); // Default to zero

  // If between A and B, match direction of velocity
  uvec mask = find(land(_A < x, x < _B));
  a_idx(mask) = conv_to<uvec>::from(sign(v(mask) + 1e-6) + 1.0);
  // slight nudge is to break deadlock at v = 0

  // If past B, break
  mask = find(x >= _B);
  a_idx(mask).fill(0);

  // If before A try to come into a controlled park
  mask = find(land(x <= _A, sum(points,1) > 1e-6));
  a_idx(mask).fill(0);
  mask = find(land(x <= _A, sum(points,1) < 1e-6));
  a_idx(mask).fill(2);  

  return a_idx;
}

uint HillcarPolicy::get_action_dim() const{
  return 1;
}

mat HillcarPolicy::get_actions(const mat & points) const{
  // Converts indicies into actual acceleration number
  // Using provided action list
  uvec a_idx = get_action_indices(points);
  return _actions.rows(a_idx);
}
