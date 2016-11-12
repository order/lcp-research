#include "refine.h"
#include <assert.h>

vec bellman_residual_at_nodes(const Discretizer * disc,
				const Simulator * sim,
				const vec & values,
				double gamma,
				int steps,
				uint samples){
  
  uint n = disc->number_of_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == (n+1));
  double max_val = 1.0 / (1.0 - gamma);

  vec V;
  if(n == values.n_elem){
    V = join_vert(values, vec{max_val}); // Pad
  }
  else{
    V = values;
  }

  Points nodes = disc->get_spatial_nodes();
  mat Q = estimate_Q(nodes,disc,sim,V,
                     gamma,steps,samples);
  assert(n == Q.n_rows);
  
  vec v_q_est = min(Q,1);
  return v_q_est - values;
}

vec bellman_residual_at_centers(const Discretizer * disc,
				const Simulator * sim,
				const vec & values,
				double gamma,
				int steps,
				uint samples){
  // Calculate the bellman residual at the cell centers

  uint N = disc->number_of_all_nodes();
  uint C = disc->number_of_cells();
  
  // Pad for the oob node
  double max_val = 1.0 / (1.0 - gamma);
  vec padded_values = join_vert(values, vec{max_val});
   
  Points centers = disc->get_cell_centers();
  // Get the interpolated value at the centers
  vec v_interp = disc->interpolate(centers,padded_values);
  assert(C == v_interp.n_elem);

  // Calculate the 1-step value estimate using dynamics  
  mat Q = estimate_Q(centers,
                     disc,
                     sim,
                     padded_values,
                     gamma,
                     steps,
                     samples);                       
  vec v_q_est = min(Q,1);
  assert(C == v_q_est.n_elem);
  // Absolute error
  return v_q_est - v_interp;
}
vec bellman_residual_at_centers_with_flows(const Discretizer * disc,
					   const Simulator * sim,
					   const vec & values,
					   const mat & flows,
					   double gamma,
					   int steps,
					   uint samples){
  // Calculate the bellman residual at the cell centers
  // Use the flow probabilities rather than max
  uint N = disc->number_of_all_nodes();
  uint C = disc->number_of_cells();
  uint A = sim->num_actions();

  // Pad for the oob node
  double max_val = 1.0 / (1.0 - gamma);
  vec padded_values = join_vert(values, vec{max_val});
  assert(N == padded_values.n_elem);

  // Get interp value at centers
  Points centers = disc->get_cell_centers();
  vec v_interp = disc->interpolate(centers,padded_values);
  assert(C == v_interp.n_elem);
  
  // Compare to 1-step value from center
  mat Q = estimate_Q(centers,
                     disc,
                     sim,
                     padded_values,
                     gamma,
                     steps,
                     samples);
  assert(size(C,A) == size(Q));

  mat padded_flows = join_vert(flows,
                               1.0/((double)A) * ones<rowvec>(A));
  assert(size(N,A) == size(padded_flows));

  mat weights = disc->interpolate(centers,padded_flows);
  weights = normalise(weights,1,1);
  assert(size(C,A) == size(weights));
  assert(all(all(weights >= 0)));
  assert(all(all(weights <= 1)));
  assert((accu(weights) - C) / (double)C < ALMOST_ZERO);
  
  vec v_q_est = sum(Q % weights,1);
  assert(C == v_q_est.n_elem);
  // Absolute error
  return v_q_est - v_interp;
}

vec advantage_function(const Discretizer * disc,
                       const Simulator * sim,
                       const vec & values,
                       double gamma,
                       int steps,
                       uint samples){
  uint N = disc->number_of_all_nodes();
  uint C = disc->number_of_cells();
  
  // Pad for the oob node
  double max_val = 1.0 / (1.0 - gamma);
  vec padded_values = join_vert(values, vec{max_val});
   
  Points centers = disc->get_cell_centers();

  // Calculate the 1-step value estimate using dynamics  
  mat Q = estimate_Q(centers,
                     disc,
                     sim,
                     padded_values,
                     gamma,
                     steps,
                     samples);
  Q = sort(Q,"ascend",1);
  return Q.col(1) - Q.col(0);
}

vec advantage_residual(const Discretizer * disc,
                       const Simulator * sim,
                       const vec & values,
                       double gamma,
                       uint samples){
  vec adv1 = advantage_function(disc, sim, values,
                                gamma,0,samples);
  vec adv2 = advantage_function(disc, sim, values,
                                gamma,1,samples);  
  return abs(adv2 - adv1);
}
  

uvec grad_policy(const Discretizer * disc,
                 const Simulator * sim,
                 const vec & value,
                 uint samples){
  uint C = disc->number_of_cells();  
  mat grad = disc->cell_gradient(value);
  Points centers = disc->get_cell_centers();

  uint A = sim->num_actions();
  mat actions = sim->get_actions();
  
  mat IP = zeros<mat>(C,A);
  for(uint a = 0; a < A; a++){
    vec action = actions.row(a).t();
    for(uint s = 0; s < samples; s++){
      Points p_next = sim->next(centers,action);
      IP.col(a) += sum(grad % p_next,1);
    }
  }

  uvec policy = col_argmin(IP);
  assert(C == policy.n_elem);
  return policy;
}

uvec q_policy(const Discretizer * disc,
              const Simulator * sim,
              const vec & values,
              double gamma,
              uint samples){
  uint N = disc->number_of_all_nodes();
  uint C = disc->number_of_cells();
  
  // Pad for the oob node
  double max_val = 1.0 / (1.0 - gamma);
  vec padded_values = join_vert(values, vec{max_val});
   
  Points centers = disc->get_cell_centers();
  // Get the interpolated value at the centers
  vec v_interp = disc->interpolate(centers,padded_values);
  assert(C == v_interp.n_elem);

  // Calculate the 1-step value estimate using dynamics  
  mat Q = estimate_Q(centers,
                     disc,
                     sim,
                     padded_values,
                     gamma,
                     0,
                     samples);

  uvec policy = col_argmin(Q);
  assert(C == policy.n_elem);
  return policy;
}

// Simplest policy; return action with max flow
uvec flow_policy(const Discretizer * disc,
                 const mat & flows){
  uint A = flows.n_cols;
  uint C = disc->number_of_cells();
  Points centers = disc->get_cell_centers();
  // Pad with 0 flow at oob node
  mat padded_flows = join_vert(flows,zeros<rowvec>(A));
 
  mat interp = disc->interpolate(centers,padded_flows);
  uvec policy = col_argmax(interp);
  assert(C == policy.n_elem);
  return policy;
}


// Simplest policy; return action with max flow
uvec flow_policy_diff(const Discretizer * disc,
                      const mat & flows){
  uint N = disc->number_of_spatial_nodes();
  uint C = disc->number_of_cells();
  uvec policy = col_argmax(flows);

  uvec diff = zeros<uvec>(C);
  umat cell_node_idx = disc->get_cell_node_indices();
  uint V = cell_node_idx.n_cols;
  uvec policy0 = policy(cell_node_idx.col(0));
  uvec policyi;
  for(uint i = 1; i < V; i++){
    policyi = policy(cell_node_idx.col(i));
    diff(find(policy0 != policyi)) += 1;
  }
  return diff;
}

uvec policy_agg(const Discretizer * disc,
                     const Simulator * sim,
                     const vec & values,
                     const mat & flows,
                     const double gamma,
                     const uint samples){
  uint N = values.n_elem;
  
  uvec gp = grad_policy(disc,sim,values,samples);
  uvec qp = q_policy(disc,sim,values,gamma,samples);
  uvec fp = flow_policy(disc,flows);

  return gp + qp + fp;
}

vec agg_flow_at_centers(const Discretizer * disc,
                         const mat & flows){
  uint n = disc->number_of_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(n == flows.n_rows or N == flows.n_rows);
  
  uint C = disc->number_of_cells();
  Points centers = disc->get_cell_centers();
  
  // Pad with 0 flow at oob node
  vec agg_flows = zeros<vec>(N);
  agg_flows.head(flows.n_rows) = sum(flows,1);
   
  vec interp = disc->interpolate(centers,agg_flows);
  return interp;
}
