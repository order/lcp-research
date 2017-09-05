#include "plane.h"
#include "misc.h"

RelativePlanesSimulator::RelativePlanesSimulator(const mat & bbox,
						 const mat & actions,
						 double noise_std,
						 double step,
						 double nmac_radius) :
  m_bbox(bbox),m_actions(actions),m_step(step),m_noise_std(noise_std),
  m_damp(1e-4), m_nmac_radius(nmac_radius){
  assert(THREE_DIM == m_bbox.n_rows);
  assert(TWO_ACTIONS == dim_actions());  // Ownship + othership
}

vec RelativePlanesSimulator::get_state_weights(const TypedPoints & points) const{
  
  uint n_spatial = points.num_spatial_nodes();
  uint n_points = points.num_all_nodes();
  uint D = points.n_cols;
  assert(THREE_DIM == 3);
  
  vec weight = ones<vec>(n_points) / (double) n_spatial; // Uniform
  weight(points.get_special_mask()).fill(0);
  assert(abs(accu(weight) - 1.0) / n_points < ALMOST_ZERO);
  return weight;
}

mat RelativePlanesSimulator::get_costs(const TypedPoints & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint A = num_actions();

  // x,y distance
  vec l2_norm = lp_norm(points.m_points.head_cols(2),2,1);
  assert(N == l2_norm.n_elem);

  // Any state within the NMAC_RADIUS gets unit cost
  vec cost = zeros(N);
  uvec ball_mask = find(l2_norm < m_nmac_radius);
  cost(ball_mask).fill(1);
  
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

  //Just copy registry over; remapping happens in points_to_element_dist.
  return TypedPoints(r_points, points.m_reg);
}

mat RelativePlanesSimulator::q_mat(const TypedDiscretizer * disc) const{
  TypedPoints points = disc->get_all_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == disc->number_of_spatial_nodes() + 1);
  uint A = num_actions();
  
  mat Q = mat(N,A+1);
  Q.col(0) = -get_state_weights(points);
  Q.tail_cols(A) = get_costs(points);
  
  return Q;
}

sp_mat RelativePlanesSimulator::transition_matrix(
						  const TypedDiscretizer * disc,
						  const vec & action,
						  bool include_oob) const{
  TypedPoints points = disc->get_spatial_nodes();
  uint n = disc->number_of_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == n + 1); // single oob node
  
  TypedPoints p_next = next(points, action);
  ElementDist P = disc->points_to_element_dist(p_next);  
  assert(size(N,n) == size(P));

  // Final row is the OOB row
  if(!include_oob){
    P.resize(n,n); // crop; sub probability matrix
  }
  else{
    P.resize(N,N);
    P(N-1,N-1) = 1.0;

#ifndef NDEBUG
    for(uint i = 0; i < N; i++){
      assert(abs(1.0 - accu(P.col(i))) < PRETTY_SMALL); 
    }
  }
#endif
  
  return P;
}

vector<sp_mat> RelativePlanesSimulator::transition_blocks(
							  const TypedDiscretizer * disc,
                                                            uint num_samples) const{
  vector<sp_mat> blocks;
  uint A = num_actions();

  bool include_oob = true;  // Promote to a MACRO
  cout << "Include out-of-bound nodes: " << include_oob << endl;
  uint n = disc->number_of_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == n+1);
  
  for(uint a = 0; a < A; a++){
    cout << "Forming transition block for action " << a << "..." << endl;
    sp_mat T;
    if(include_oob){
      T = sp_mat(N,N);
    }
    else{
      T = sp_mat(n,n);
    }
    for(uint s = 0; s < num_samples;s++){
      cout << "\tSample " << s << "..." << endl;
      T += transition_matrix(disc,m_actions.row(a).t(), include_oob);
    }
    T /= (double)num_samples;
    if(include_oob){
      assert(size(N,N) == size(T));
    }
    else{
      assert(size(n,n) == size(T));
    }
    blocks.push_back(T);
  }
  return blocks;
}

vector<sp_mat> RelativePlanesSimulator::lcp_blocks(const TypedDiscretizer * disc,
						   const double gamma,
						   uint num_samples) const{
  uint A = num_actions();
  uint N = disc->number_of_all_nodes(); // Not using oob
  vector<sp_mat> blocks = transition_blocks(disc, num_samples);
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
