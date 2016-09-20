#include "simulator.h"
#include "dubins.h"

using namespace dubins;

DubinsCarSimulator::DubinsCarSimulator(const mat &actions,
                                       double noise_std,
                                       double step):
  m_actions(actions), m_noise_std(noise_std), m_step(step){}

mat DubinsCarSimulator::get_costs(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(DUBINS_DIM == D);
  assert(points.is_finite());

  uint A = m_actions.n_rows;

  vec l2_norm = lp_norm(points,2,1);
  assert(all(l2_norm >= 0));

  vec cost = min(ones<vec>(N),max(zeros<vec>(N),l2_norm - 0.1));
  mat costs = repmat(cost,1,A);
  return costs;
}

vec DubinsCarSimulator::get_state_weights(const Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(DUBINS_DIM == D);

  vec l2_norm = lp_norm(points,2,1);
  assert(all(l2_norm >= 0));

  vec weight = max(l2_norm) - l2_norm + 1e-3;
  assert(all(weight >= 0));
  weight /= accu(weight);
  
  return weight;
}

mat DubinsCarSimulator::get_actions() const{
  return m_actions;
}

Points DubinsCarSimulator::next(const Points & points,
                                const vec & actions) const{
  assert(DUBINS_DIM == points.n_cols);
  assert(points.is_finite());
  assert(DUBINS_ACTION_DIM == actions.n_elem);

  Points new_points = Points(points);
  double u1 = actions(0); // linear velocity
  double u2 = actions(1); // angular velocity
  
  new_points.col(0) += m_step * u1 * cos(points.col(2)); // x
  new_points.col(1) += m_step * u1 * sin(points.col(2)); // y
  new_points.col(2) += m_step * u1*u2; // theta

  // Angle wrap
  uvec wrap_idx = uvec{2};
  mat bbox = {{datum::nan,datum::nan},
              {datum::nan,datum::nan},
              {-datum::pi,datum::pi}};
  wrap(new_points,wrap_idx,bbox);

  // May be out of bounds
  return new_points;
}

sp_mat DubinsCarSimulator::transition_matrix(const Discretizer * disc,
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

uint DubinsCarSimulator::num_actions() const{
  return m_actions.n_rows;
}
uint DubinsCarSimulator::dim_actions() const{
  return m_actions.n_cols;
}
