#include "simulator.h"
#include "dubins.h"

using namespace dubins;

DubinsCarSimulator::DubinsCarSimulator(const mat &actions,
                                       double noise_std,
                                       double step):
  m_actions(actions), m_noise_std(noise_std), m_step(step){}

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


RoundaboutDubinsCarSimulator::RoundaboutDubinsCarSimulator(const mat &actions,
                                                           double noise_std,
                                                           double step):
  DubinsCarSimulator(actions,noise_std,step){}

mat RoundaboutDubinsCarSimulator::get_costs(const Points & points) const{

  
  uint N = points.n_rows;

  // Cost is 1 unless lowered elsewhere
  vec cost = ones<vec>(N);
  double pi_2 = datum::pi / 2.0;

  ////////////////////////////////////
  // Add in legal moves, which cost 0.1
  
  // Top left lane
  bvec lane_tl = ones<bvec>(N);
  lane_tl(find(points.col(0) > 0)) *= 0; // Zero out right
  lane_tl(find(points.col(1) < 2.5)) *= 0; // Zero out bottom
  // If you're in the top left lane you can legally go down,
  // which we're calling any thing in [-pi,0]
  lane_tl(find(points.col(2) > 0)) *= 0; // Zero out +vec angles
  cost(find(lane_tl)).fill(0.1);

  bvec lane_tr = ones<bvec>(N);
  lane_tr(find(points.col(0) < 0)) *= 0;
  lane_tr(find(points.col(1) < 2.5)) *= 0;
  // Legal moves: [0,pi]
  lane_tr(find(points.col(2) < 0)) *= 0;
  cost(find(lane_tr)).fill(0.1);

  // Right top lane
  bvec lane_rt = ones<bvec>(N);
  lane_rt(find(points.col(0) < 2.5)) *= 0;
  lane_rt(find(points.col(1) < 0)) *= 0;
  // Legal moves: [-pi,-0.5 pi] and [0.5 pi, pi]
  lane_rt(find(abs(points.col(2)) < pi_2)) *= 0;
  cost(find(lane_rt)).fill(0.1);

  // Right bottom lane
  bvec lane_rb = ones<bvec>(N);
  lane_rt(find(points.col(0) < 2.5)) *= 0;
  lane_rt(find(points.col(1) > 0)) *= 0;
  // Legal moves: [-0.5 pi, 0.5 pi]
  lane_rt(find(abs(points.col(2)) > pi_2)) *= 0;
  cost(find(lane_rt)).fill(0.1);

  //////////////////////////
  // Roundabout part
  rowvec origin = zeros<rowvec>(2);
  vec d = dist(points.head_cols(2),origin);
  
  // Need special cases in case of x,y = 0
  // Work on these when they crop up
  assert(not any(abs(points.col(0)) < ALMOST_ZERO));
  assert(not any(abs(points.col(1)) < ALMOST_ZERO)); 
  vec angle = atan(points.col(1) / points.col(0));

  for(uint i = 0; i < N; i++){
    if(d(i) > 3)
      continue;

    double a = angle(i); // Angle on roundabout
    double o = points(i,2); // Point's orientation

    // Legal orientation
    assert(o < datum::pi);
    assert(o > -datum::pi);

    // Simple if orientation is -ve: should be in [a,a+pi]
    if(o <= 0 and a <= o and o <= a + datum::pi)
      cost(i) = 0.1;

    // Two segments to check if orientation is +ve
    // Is it in [a,pi]?
    if(o > 0 and a <= o)
      cost(i) = 0.1;

    // Is it in [-pi,a-pi]?
    if(o > 0 and o < a - datum::pi)
      cost(i) = 0.1;
  }

  /////////////////////////////
  // Goal region is a box:
  // [4,5] x [0.7,1.5] x {a | |a| > 11/6 pi}
  bvec goal = ones<bvec>(N);
  goal(find(points.col(0) > 5)) *= 0;
  goal(find(points.col(0) < 4)) *= 0;
  
  goal(find(points.col(1) > 1.5)) *= 0;
  goal(find(points.col(1) < 0.75)) *= 0;
  
  double thresh = 11.0 / 6.0 * datum::pi;
  goal(find(abs(points.col(2)) < thresh)) *= 0;

  cost(find(goal)).fill(0);

  // Duplicate
  mat costs = repmat(cost,1,num_actions());
  return costs;
}
vec RoundaboutDubinsCarSimulator::get_state_weights(const Points & points) const{
  uint N = points.n_rows;

  //A little bit of weight everywhere
  vec weights = 0.01*ones<vec>(N);

  // More weight in a small cylinder around a starting point
  rowvec start = {-0.75, 7.75};
  vec d = dist(points.head_cols(2),start);
  double pi_2 = datum::pi / 2.0;
  
  bvec start_mask = zeros<bvec>(N);
  start_mask(find(d <= 0.75)).fill(1);
  start_mask(find(points.col(3) < -pi_2 - 0.5)).fill(0);
  start_mask(find(points.col(3) > pi_2 + 0.5)).fill(0);
  
  weights(find(1 == start_mask)).fill(1);
  weights /= accu(weights);
  
  return weights;
}
