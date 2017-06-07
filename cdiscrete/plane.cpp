#include "planes.h"
#include "misc.h"

using namespace tet_mesh;

RelativePlanesSimulator::RelativePlanesSimulator(const mat & bbox,
                                                     const mat &actions,
                                                     double noise_std,
                                                     double step) :
  m_bbox(bbox),m_actions(actions),m_step(step),m_noise_std(noise_std),
  m_damp(1e-4){
  assert(TET_NUM_DIM == m_bbox.n_rows);
  assert(TWO_ACTIONS == dim_actions());  // Ownship + othership
}

vec RelativePlanesSimulator::get_state_weights(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(TET_NUM_DIM == D);
  
  vec weight = ones<vec>(N) / (double) N; 
  return weight;
}

mat RelativePlanesSimulator::get_costs(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint A = num_actions();
  
  assert(TET_NUM_DIM == D);

  vec l2_norm = lp_norm(points,2,1);
  assert(N == l2_norm.n_elem);
  assert(all(l2_norm >= 0));

  // Any state within the NMAC_RADIUS gets unit cost
  vec cost = zeros(N);
  cost(l2_norm < NMAC_RADIUS).fill(1);
  mat costs = repmat(cost,1,A);
  return costs;
}

mat RelativePlanesSimulator::get_actions() const{
  return m_actions;
}

Points RelativePlanesSimulator::next(const Points & points,
                                       const vec & actions) const{
  assert(TET_NUM_DIM == points.n_cols);
  assert(dim_actions() == actions.n_elem);
    
  double h = m_step;
  double u = action[0] + m_noise_std*randn<vec>(1); // Own action
  double v = action[1] + m_noise_std*randn<vec>(1); // Other action

  vec theta = points.col(2);

  // New points after translating so the own ship is at (0,0,h*u)
  Points t_points = Points(size(points));
  t_points.col(0) = points.col(0) + h * (cos(theta) - 1);
  t_points.col(1) = points.col(1) + h * sin(theta);

  // New points after rotating so the own ship is a (0,0,0)
  double phi = -h * u;
  Points r_points = Points(size(points));
  r_points.col(0) = cos(phi) * t_points.col(0) - sin(phi) * t_points.col(1);
  r_points.col(1) = sin(phi) * t_points.col(0) + cos(phi) * t_points.col(1);
  r_points.col(2) = theta + h * (v - u);

  // Wrap the angle
  uvec angle_col = {2};
  mat angle_bbox = {
    {-datum::inf,datum:inf},
    {-datum::inf,datum:inf},
    {-datum::pi,datum::pi}
  };
  wrap(r_points, angle_col, angle_bbox);
  
  return r_points;
}

mat RelativePlanesSimulator::q_mat(const Discretizer * disc) const{
  Points points = disc->get_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  assert(N == points.n_rows + 1); // One OOB node
  uint A = num_actions();
  
  mat Q = mat(N,A+1);
  
  // Set the state-weight component
  Q.col(0).head_rows(N) = -get_state_weights(points);
  Q(N-1,0) = 0;  // No weight for the OOB node

  // Fill in costs. OOB has 0 cost regardless.
  Q.tail_cols(A) = join_cols(get_costs(points), zeros<mat>(1,A));
  return Q;
}

sp_mat RelativePlanesSimulator::transition_matrix(const Discretizer * disc,
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

vector<sp_mat> RelativePlanesSimulator::transition_blocks(const Discretizer * disc,
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

vector<sp_mat> RelativePlanesSimulator::lcp_blocks(const Discretizer * disc,
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

void RelativePlanesSimulator::add_bang_bang_curve(TriMesh & mesh,
                                                    uint num_curve_points) const{
  VertexHandle v_zero = mesh.locate_vertex(Point(0,0));

  VertexHandle v_old = v_zero;
  VertexHandle v_new;  
  double x,y;
  double N = num_curve_points;

  vec lb = m_bbox.col(0);
  vec ub = m_bbox.col(1);
  
  // Figure out the max y within boundaries
  assert(lb(0) < 0);
  double max_y = min(ub(1),std::sqrt(-lb(0)));
  assert(max_y > 0);

  //Insert +ve y, -ve x points
  for(double i = 1; i < N; i++){
    y = max_y * i / N; // Uniform over y
    assert(y > 0);
    x = - y * y;
    if(x <= lb(0)) break;
    v_new = mesh.insert(Point(x,y));
    mesh.insert_constraint(v_old,v_new);
    v_old = v_new;
  }

  //Insert -ve y, +ve x points
  v_old = v_zero;
  assert(ub(0) > 0);
  double min_y = max(lb(1),-std::sqrt(ub(0)));
  assert(min_y < 0);
  
  for(double i = 1; i < N; i++){
    y = min_y * i / N;
    assert(y < 0);
    x = y * y;
    if(x >= ub(0)) break;
    v_new = mesh.insert(Point(x,y));
    mesh.insert_constraint(v_old,v_new);
    v_old = v_new;
  }
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
