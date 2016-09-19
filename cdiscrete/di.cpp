#include "di.h"
#include "misc.h"

DoubleIntegratorSimulator::DoubleIntegratorSimulator(const mat & bbox,
                                                     const mat &actions,
                                                     double noise_std,
                                                     double step) :
  m_bbox(bbox),m_actions(actions),m_step(step),m_noise_std(noise_std){}

vec DoubleIntegratorSimulator::get_state_weights(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(TRI_NUM_DIM == D);

  vec l2_norm = lp_norm(points,2,1);
  assert(N == l2_norm.n_elem);
  assert(all(l2_norm >= 0));

  vec weight = max(l2_norm) - l2_norm + 1e-3;
  assert(all(weight >= 0));
  weight /= accu(weight);
 
  return weight;
}

mat DoubleIntegratorSimulator::get_costs(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(TRI_NUM_DIM == D);

  vec l2_norm = lp_norm(points,2,1);
  assert(N == l2_norm.n_elem);
  assert(all(l2_norm >= 0));

  vec cost = min(ones<vec>(N),max(zeros<vec>(N),l2_norm - 0.5));
  mat costs = repmat(cost,1,2);
  return costs;
}

mat DoubleIntegratorSimulator::get_actions() const{
  return m_actions;
}

Points DoubleIntegratorSimulator::next(const Points & points,
                                       const vec & actions) const{
  assert(TRI_NUM_DIM == points.n_cols);
  assert(dim_actions() == actions.n_elem);
  
  Points new_points = Points(size(points));
  assert(size(points) == size(new_points));
  double t = m_step;
  vec noise = m_noise_std*randn<vec>(1);
  double u = actions(0) + noise(0);
  
  new_points.col(0) = points.col(0) + t*points.col(1) + 0.5*t*t*u;
  new_points.col(1) = points.col(1) + t*u;

  saturate(new_points,
           uvec{0,1},
           m_bbox);
  
  return new_points;
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

void DoubleIntegratorSimulator::add_bang_bang_curve(TriMesh & mesh,
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

uint DoubleIntegratorSimulator::num_actions() const{
  return m_actions.n_rows;
}
uint DoubleIntegratorSimulator::dim_actions() const{
  return m_actions.n_cols;
}
mat DoubleIntegratorSimulator::get_bounding_box() const{
  return m_bbox;
}

