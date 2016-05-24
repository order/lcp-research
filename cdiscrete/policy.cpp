#include <assert.h>

#include "policy.h"

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

vec DIBangBangPolicy::get_actions(const mat & points) const{
  // Converts indicies into actual acceleration number
  // Using provided action list
  uvec a_idx = get_action_indices(points);
  return _actions.rows(a_idx);
}
