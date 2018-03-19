#include <assert.h>
#include "di.h"
#include "misc.h"

DoubleIntegratorSimulator::DoubleIntegratorSimulator(const mat & bbox,
                                                     const mat &actions,
                                                     double noise_std,
                                                     double step) :
  m_bbox(bbox),m_actions(actions),m_step(step),m_noise_std(noise_std),
  m_damp(1e-4){}

vec DoubleIntegratorSimulator::get_state_weights(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(DOUBLE_INT_DIM == D);
  
  vec weight = ones<vec>(N) / (double) N;
  return weight;
}

vec DoubleIntegratorSimulator::get_state_weights(const TypedPoints & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(DOUBLE_INT_DIM == D);

  vec weight = ones<vec>(N) / (double) N;
  return weight;
}

mat DoubleIntegratorSimulator::get_costs(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint A = num_actions();
  assert(DOUBLE_INT_DIM == D);

  vec l2_norm = lp_norm(points, 2, 1);
  assert(N == l2_norm.n_elem);
  assert(all(l2_norm >= 0));

  vec cost = l2_norm - 0.5;
  cost = max(zeros<vec>(N), cost);
  cost = min(ones<vec>(N), cost);
  
  mat costs = repmat(cost, 1, A);
  return costs;
}

mat DoubleIntegratorSimulator::get_costs(const TypedPoints & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint A = num_actions();
  assert(DOUBLE_INT_DIM == D);

  uvec spatial_mask = points.get_spatial_mask();
  uint n_spatial = spatial_mask.n_elem;
  mat spatial_points = points.m_points.rows(spatial_mask);

  mat _cost = get_costs(spatial_points);
  assert(size(n_spatial, A) == size(_cost));
  
  mat costs = ones<mat>(N, A);  // All special nodes are 1
  costs.rows(spatial_mask) = _cost;
  return costs;
}

mat DoubleIntegratorSimulator::get_actions() const{
  return m_actions;
}

Points DoubleIntegratorSimulator::next(const Points & points,
                                       const vec & actions) const{
  assert(DOUBLE_INT_DIM == points.n_cols);
  assert(dim_actions() == actions.n_elem);
  
  Points new_points = Points(size(points));
  assert(size(points) == size(new_points));
  double t = m_step;
  vec noise = m_noise_std * randn<vec>(1);
  double u = actions(0) + noise(0);
  
  new_points.col(0) = points.col(0) + t * points.col(1) + 0.5 * t * t * u;
  new_points.col(1) = (1.0 - m_damp) * points.col(1) + t * u;

  saturate(new_points,
           uvec{0, 1},
           m_bbox);
  
  return new_points;
}

TypedPoints DoubleIntegratorSimulator::next(const TypedPoints & points,
					    const vec & actions) const{
  // TODO
  assert(false);
}

mat DoubleIntegratorSimulator::q_mat(const TypedDiscretizer * disc) const{
  TypedPoints points = disc->get_all_nodes();
  uint N = disc->number_of_all_nodes();
  uint A = num_actions();
  
  mat Q = mat(N, A + 1);
  vec w = get_state_weights(points);
  assert(N == w.n_elem);
  Q.col(0) = -w;

  mat costs = get_costs(points);
  assert(size(N, A) == size(costs));
  Q.tail_cols(A) = costs;
  return Q;
}

mat DoubleIntegratorSimulator::q_mat(const Discretizer * disc) const{
  TypedPoints points = disc->get_spatial_nodes();
  uint N = disc->number_of_spatial_nodes();
  uint A = num_actions();
  
  mat Q = mat(N,A+1);
  Q.col(0) = -get_state_weights(points);
  Q.tail_cols(A) = get_costs(points);
  return Q;
}

sp_mat DoubleIntegratorSimulator::transition_matrix(const Discretizer * disc,
                                                    const vec & action,
                                                    bool include_oob) const{
  Points points = disc->get_spatial_nodes();
  uint n = disc->number_of_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == n+1); // single oob node
  
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

sp_mat DoubleIntegratorSimulator::transition_matrix(const TypedDiscretizer * disc,
						    const vec & action) const{
  TypedPoints points = disc->get_all_nodes();
  uint n = disc->number_of_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == n+1); // single oob node
  
  TypedPoints p_next = next(points, action);
  ElementDist P = disc->points_to_element_dist(p_next);
  assert(size(N,N) == size(P));
  return P;
}

vector<sp_mat> DoubleIntegratorSimulator::transition_blocks(const Discretizer * disc,
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

vector<sp_mat> DoubleIntegratorSimulator::transition_blocks(const TypedDiscretizer * disc,
							    uint num_samples) const{

  vector<sp_mat> blocks;
  uint A = num_actions();

  // Swap out if there are oob
  uint n = disc->number_of_all_nodes();
  for(uint a = 0; a < A; a++){
    sp_mat T = sp_mat(n,n);
    for(uint s = 0; s < num_samples;s++){
      T += transition_matrix(disc,m_actions.row(a).t());
    }
    T /= (double)num_samples;
    blocks.push_back(T);
  }
  return blocks;
}

vector<sp_mat> DoubleIntegratorSimulator::lcp_blocks(const Discretizer * disc,
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

vector<sp_mat>  DoubleIntegratorSimulator::lcp_blocks(const TypedDiscretizer * disc,
						      const double gamma,
						      uint num_samples) const{
  uint A = num_actions();
  uint N = disc->number_of_all_nodes(); // Using OOB
  vector<sp_mat> blocks = transition_blocks(disc,num_samples);
  assert(A == blocks.size());

  for(uint a = 0; a < A; a++){
    assert(size(N,N) == size(blocks[a]));
    blocks[a] = speye(N,N) - gamma * blocks[a];
  }
  return blocks;
}


uint DoubleIntegratorSimulator::num_actions() const{
  return m_actions.n_rows;
}
uint DoubleIntegratorSimulator::dim_actions() const{
  return m_actions.n_cols;
}
mat DoubleIntegratorSimulator::get_bounding_box() const{
  return m_bbox;
}

