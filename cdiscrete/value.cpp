QPolicy::QPolicy(const mat & actions,
		 const vec & values,
		 const RegGrid & grid,
		 const Problem & problem){
  _grid = grid;
  _q_values = generate_q_values(values,grid,problem);
  _actions = actions;
}

uvec QPolicy::get_action_indicies(const mat & points) const{
  return min_interp_fns(_q_values,points,_grid);
}

vec QPolicy::get_actions(const mat & points) const{
  // Converts indicies into actual acceleration number
  // Using provided action list
  uvec a_idx = get_action_indices(points);
  return _actions.rows(a_idx);
}

uint QPolicy::get_action_dim() const{
  return _actions.n_cols;
}
