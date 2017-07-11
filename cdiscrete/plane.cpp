#include "plane.h"
#include "misc.h"

RelativePlanesSimulator::RelativePlanesSimulator(const mat & bbox,
						 const mat & actions,
						 double noise_std,
						 double step) :
  m_bbox(bbox),m_actions(actions),m_step(step),m_noise_std(noise_std),
  m_damp(1e-4){
  assert(THREE_DIM == m_bbox.n_rows);
  assert(TWO_ACTIONS == dim_actions());  // Ownship + othership
}

vec RelativePlanesSimulator::get_state_weights(const TypedPoints & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(THREE_DIM == 3);
  
  vec weight = ones<vec>(N) / (double) N; // Uniform
  return weight;
}

mat RelativePlanesSimulator::get_costs(const TypedPoints & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint A = num_actions();
  
  vec l2_norm = lp_norm(points.m_points,2,1);
  assert(N == l2_norm.n_elem);

  // Any state within the NMAC_RADIUS gets unit cost
  vec cost = zeros(N);
  cost(l2_norm < NMAC_RADIUS).fill(1);

  // Make sure that OOB is heaven
  uvec special_idx = points.get_special_mask();
  cost(special_idx).fill(0);
  
  mat costs = repmat(cost,1,A);
  return costs;
}

mat RelativePlanesSimulator::get_actions() const{
  return m_actions;
}

TypedPoints RelativePlanesSimulator::next(const TypedPoints & points,
					  const vec & action) const{
  assert(THREE_DIM == points.n_cols);
  assert(dim_actions() == action.n_elem);
    
  double h = m_step;
  vec noise = randn<vec>(2);
  double u = action[0] + m_noise_std*noise(0); // Own action
  double v = action[1] + m_noise_std*noise(1); // Other action

  vec theta = points.m_points.col(2);

  // New points after translating so the own ship is at (0,0,h*u)
  Points t_points = Points(points.n_rows, points.n_cols);
  t_points.col(0) = points.m_points.col(0) + h * (cos(theta) - 1);
  t_points.col(1) = points.m_points.col(1) + h * sin(theta);

  // New points after rotating so the own ship is a (0,0,0)
  double phi = -h * u;
  Points r_points = Points(points.n_rows, points.n_cols);
  r_points.col(0) = cos(phi) * t_points.col(0) - sin(phi) * t_points.col(1);
  r_points.col(1) = sin(phi) * t_points.col(0) + cos(phi) * t_points.col(1);
  r_points.col(2) = theta + h * (v - u);

  return TypedPoints(r_points, points.m_reg);
}

mat RelativePlanesSimulator::q_mat(const TypedDiscretizer * disc) const{
  Points points = disc->get_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == points.n_rows + 1); // One OOB node
  uint A = num_actions();
  
  mat Q = mat(N,A+1);
  
  // Set the state-weight component
  Q.col(0).head_rows(N) = -get_state_weights(points);
  Q(N-1,0) = 0;  // No weight for the OOB node

  // Fill in costs. OOB has 0 cost (success!).
  Q.tail_cols(A) = join_cols(get_costs(points), zeros<mat>(1,A));
  return Q;
}

sp_mat RelativePlanesSimulator::transition_matrix(const TypedDiscretizer * disc,
						  const vec & action,
						  bool include_oob) const{
  Points points = disc->get_spatial_nodes();
  uint n = disc->number_of_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == points.n_rows + 1); // single oob node
  
  Points p_next = next(points,action);
  ElementDist P = disc->points_to_element_dist(p_next);
  assert(size(N,n) == size(P));

  
  // Final row is the OOB row
  assert(!include_oob);
  if(!include_oob){
    // Make sure we aren't pruning off important transitions
    assert(accu(P.tail_rows(1)) < ALMOST_ZERO);
    P = resize(P,n,n);
  }
  else{
    // Add an all-zero column
    P = resize(P,N,N);
    assert(accu(P.tail_cols(1)) < ALMOST_ZERO);
  }  
  return P;
}

vector<sp_mat> RelativePlanesSimulator::transition_blocks(const TypedDiscretizer * disc,
                                                            uint num_samples) const{
  vector<sp_mat> blocks;
  uint A = num_actions();

  // Swap out if there are oob
  bool include_oob = false;
  uint n = disc->number_of_spatial_nodes();
  for(uint a = 0; a < A; a++){
    sp_mat T = sp_mat(n,n);
    for(uint s = 0; s < num_samples;s++){
      T += transition_matrix(disc,m_actions.row(a).t(),include_oob);
    }
    T /= (double)num_samples;
    blocks.push_back(T);
  }
  return blocks;
}

vector<sp_mat> RelativePlanesSimulator::lcp_blocks(const typedDiscretizer * disc,
						   const double gamma,
						   uint num_samples) const{
  uint A = num_actions();
  uint N = disc->number_of_spatial_nodes(); // Not using oob
  vector<sp_mat> blocks = transition_blocks(disc,num_samples);
  assert(A == blocks.size());

  for(uint a = 0; a < A; a++){
    assert(size(N,N) == size(blocks[a]));
    blocks[a] = speye(N,N) - gamma * blocks[a];
  }
  return blocks;
}

uint RelativePlanesSimulator::num_actions() const{
  return m_actions.n_rows;
}
uint RelativePlanesSimulator::dim_actions() const{
  return m_actions.n_cols;
}
mat RelativePlanesSimulator::get_bounding_box() const{
  return m_bbox;
}
