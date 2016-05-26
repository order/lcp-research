#include "value.h"

QPolicy::QPolicy(const mat & q,
		 const mat & actions,
		 const RegGrid & grid){
  _q = q;
  _grid = grid;
  _actions = actions;
}

uvec QPolicy::get_action_indices(const mat & points) const{
  return min_interp_fns(_q,points,_grid);
}

mat QPolicy::get_actions(const mat & points) const{
  // Converts indicies into actual acceleration number
  // Using provided action list
  uvec a_idx = get_action_indices(points);
  return _actions.rows(a_idx);
}

uint QPolicy::get_action_dim() const{
  return _actions.n_cols;
}
