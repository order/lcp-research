#include "hillcar.h"
#include "misc.h"

#include "tri_mesh.h"
using namespace tri_mesh;

HillcarSimulator::HillcarSimulator(const mat & bbox,
                                                     const mat &actions,
                                                     double noise_std,
                                                     double step) :
  m_bbox(bbox),m_actions(actions),m_step(step),m_noise_std(noise_std){}

vec HillcarSimulator::get_state_weights(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(TRI_NUM_DIM == D);

  rowvec center = {-4.0,0.0};
  vec l2_norm = lp_norm(points - repmat(center,N,1),2,1);
  assert(N == l2_norm.n_elem);
  assert(all(l2_norm >= 0));

  vec weight = max(l2_norm) - l2_norm + 1e-3;
  assert(all(weight >= 0));
  weight /= accu(weight);
  
  return weight;
}

mat HillcarSimulator::get_costs(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(2 == D);

  vec l2_norm = lp_norm(points,2,1);
  assert(N == l2_norm.n_elem);
  assert(all(l2_norm >= 0));

  vec cost = min(ones<vec>(N),max(zeros<vec>(N),l2_norm - 0.1));
  mat costs = repmat(cost,1,2);
  costs.col(0) += 1e-6;
  return costs;
}

mat HillcarSimulator::get_actions() const{
  return m_actions;
}

void HillcarSimulator::enforce_boundary(Points & points) const{
  // Special hillcar boundary. If out of bounds in X coords,
  // Saturate X and set V to 0.
  assert(2 == points.n_cols);
  assert(is_finite(points));
  
  saturate(points,
           uvec{V_VAR},
           m_bbox);
}

Points HillcarSimulator::next(const Points & points,
                                       const vec & actions) const{
  assert(2 == points.n_cols);
  assert(1 == actions.n_elem);
  
  Points new_points = Points(size(points));
  assert(size(points) == size(new_points));
  double t = m_step;
  vec noise = m_noise_std*randn<vec>(1);
  double u = actions(0) + noise(0);

  vec slope = triangle_slope(points.col(X_VAR));
  vec hypo2 = 1.0 + pow(slope,2.0);
  vec accel = u / sqrt(hypo2) - (GRAVITY*slope) / hypo2;
  
  new_points.col(X_VAR) = points.col(X_VAR)
    + t*points.col(V_VAR)
    + 0.5*t*t*accel;
  new_points.col(V_VAR) = points.col(V_VAR) + t*accel;

  enforce_boundary(new_points);
  
  return new_points;
}

sp_mat HillcarSimulator::transition_matrix(const Discretizer * disc,
                                           const vec & action,
                                           bool include_oob) const{
  assert(include_oob);
  
  Points points = disc->get_spatial_nodes();
  uint N = disc->number_of_all_nodes();
  uint n = disc->number_of_spatial_nodes();
  
  Points p_next = next(points,action);
  ElementDist P = disc->points_to_element_dist(p_next);
  // Final row is the OOB row
  assert(size(N,n) == size(P));
  P = resize(P,N,N);
  
  return P;
}

uint HillcarSimulator::num_actions() const{
  return m_actions.n_rows;
}
uint HillcarSimulator::dim_actions() const{
  return m_actions.n_cols;
}
mat HillcarSimulator::get_bounding_box() const{
  return m_bbox;
}

vec triangle_wave(vec x, double P, double A){
  x /= P;
  return A * ( 2* abs(2 * (x - floor(x + 0.5))) - 1);
}

vec triangle_slope(vec x){
  double P = 8.0; // Period
  double A = 1.0; // Amplitude
  double T = 0.05; // Threshold for soft thresh
    
  return soft_threshold(triangle_wave(x - P/4,P,A),T);
}
