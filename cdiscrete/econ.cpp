#include "di.h"
#include "misc.h"

using namespace tri_mesh;

StockSimulator::StockSimulator(const mat & bbox,
			       double step) :
  m_bbox(bbox),m_step(step){
}


mat StockSimulator::get_actions() const{
  return {{0.0},{1.0}};  // List initialization
}

mat StockSimulator::get_costs(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(TRI_NUM_DIM == D);
  /*
    Assumption: first time is the immediate spot price
   */

  mat costs = mat(N,2);
  costs.col(0).fill(0); // First action: don't sell. Cost = 0.
  costs.col(1) = -points.col(0); // Second action: sell.
  
  return costs;
}


vec StockSimulator::get_state_weights(const Points & points) const{
  // State weights: uniform seems like a bad assumption. Put all weight on
  // state closest to (0,0)?
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(TRI_NUM_DIM == D);
  
  vec weight = ones<vec>(N) / (double) N;
  return weight;
}


Points StockSimulator::next(const Points & points,
			    const vec & actions) const{
  assert(TRI_NUM_DIM == points.n_cols);
  assert(dim_actions() == actions.n_elem);
  
  Points new_points = Points(size(points));
  assert(size(points) == size(new_points));
  
  double t = m_step;
  vec noise = m_noise_std*randn<vec>(1);  // TODO: replace with empirical
  double u = actions(0) + noise(0);
  
  new_points.col(0) = points.col(0) + t*points.col(1) + 0.5*t*t*u;
  new_points.col(1) = (1.0 - m_damp)*points.col(1) + t*u;

  saturate(new_points,
           uvec{0,1},
           m_bbox);
  
  return new_points;
}

mat StockSimulator::q_mat(const Discretizer * disc) const{
  Points points = disc->get_spatial_nodes();
  uint N = disc->number_of_spatial_nodes();
  uint A = num_actions();
  
  mat Q = mat(N,A+1);
  Q.col(0) = -get_state_weights(points);
  Q.tail_cols(A) = get_costs(points);
  return Q;
}

sp_mat StockSimulator::transition_matrix(const Discretizer * disc,
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

vector<sp_mat> StockSimulator::transition_blocks(const Discretizer * disc,
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

vector<sp_mat> StockSimulator::lcp_blocks(const Discretizer * disc,
                                                     const double gamma,
                                                     uint num_samples) const{
  // TODO: Move to more general file
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

uint StockSimulator::num_actions() const{
  return m_actions.n_rows;
}
uint StockSimulator::dim_actions() const{
  return m_actions.n_cols;
}
mat StockSimulator::get_bounding_box() const{
  return m_bbox;
}


TriMesh generate_initial_mesh(double angle, double length, const mat & bbox){
  // TODO: move to more general file
  TriMesh mesh;
  mesh.build_box_boundary(bbox);
  
  cout << "Refining based on (" << angle
       << "," << length <<  ") criterion ..."<< endl;
  mesh.refine(angle,length);
  
  cout << "Optimizing (25 rounds of Lloyd)..."<< endl;
  mesh.lloyd(25);
  
  cout << "Re-refining.."<< endl;
  mesh.refine(angle,length);

  mesh.freeze();
  return mesh;
}
